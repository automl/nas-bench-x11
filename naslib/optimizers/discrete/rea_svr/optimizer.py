import collections
import logging
import torch
import copy
import time
import numpy as np

from sklearn.svm import NuSVR
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from scipy import stats

from naslib.optimizers.core.metaclasses import MetaOptimizer

from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils.utils import AttrDict, count_parameters_in_MB
from naslib.utils.logging import log_every_n_seconds

logger = logging.getLogger(__name__)

def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))

class REA_SVR(MetaOptimizer):
    
    # training the models is not implemented
    using_step_function = False
    
    def __init__(self, config, metric=Metric.VAL_ACCURACY,
                 all_curve=True, model_name='svr',
                 best_hyper=None, n_hypers=1000):
        super().__init__()
        self.n_hypers = n_hypers
        self.all_curve = all_curve
        self.model_name = model_name
        self.best_hyper = best_hyper
        self.name = 'rea-svr'
        self.metric = metric
        self.info = []
        self.y_train = []
        self.fidelity = config.search.single_fidelity
        if config.search_space == 'nasbench101':
            self.extrapolation = config.search.fidelity
            self.top_n_percent = 0.5
        elif config.search_space in ["nasbench201", "nasbench211"]:
            self.extrapolation = config.search.fidelity // 2
            self.top_n_percent = 0.5
        elif config.search_space == 'darts':
            self.extrapolation = config.search.fidelity // 2
            self.top_n_percent = 0.2
        elif config.search_space == 'nlp':
            self.extrapolation = config.search.fidelity
            self.top_n_percent = 0.2
        else:
            raise NotImplementedError('{} is not yet implemented yet'.format(config.search_space))

        self.train_svr = True

        self.config = config
        self.epochs = config.search.epochs
        self.population_size = config.search.population_size
        self.sample_size = config.search.sample_size

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        self.population = collections.deque(maxlen=self.population_size)
        self.history = torch.nn.ModuleList()


    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert search_space.QUERYABLE, "Regularized evolution is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api

    def collate_inputs(self, VC_all_archs_list, AP_all_archs_list):
        """
        Args:
            VC_all_archs_list: a list of validation accuracy curves for all archs
            AP_all_archs_list: a list of architecture features for all archs

        Returns:
            X: an collated array of all input information used for extrapolation model

        """
        VC = np.vstack(VC_all_archs_list)  # dimension: n_archs x n_epochs
        DVC = np.diff(VC, n=1, axis=1)
        DDVC = np.diff(DVC, n=1, axis=1)

        mVC = np.mean(VC, axis=1)[:, None]
        stdVC = np.std(VC, axis=1)[:, None]
        mDVC = np.mean(DVC, axis=1)[:, None]
        stdDVC = np.std(DVC, axis=1)[:, None]
        mDDVC = np.mean(DDVC, axis=1)[:, None]
        stdDDVC = np.std(DDVC, axis=1)[:, None]

        if self.all_curve:
            TS_list = [VC, DVC, DDVC, mVC, stdVC]
        else:
            TS_list = [mVC, stdVC, mDVC, stdDVC, mDDVC, stdDDVC]

        if self.metric == Metric.TRAIN_LOSS:
            sumVC = np.sum(VC, axis=1)[:, None]
            TS_list += [sumVC]

        TS = np.hstack(TS_list)

        if len(AP_all_archs_list) != 0:
            AP = np.vstack(AP_all_archs_list)
            X = np.hstack([AP, TS])
        else:
            X = TS
        return X

    def get_data_reqs(self):
        """
        Returns a dictionary with info about whether the predictor needs
        extra info to train/query.
        """
        reqs = {'requires_partial_lc':True,
                'metric':self.metric,
                'requires_hyperparameters':True,
                'hyperparams':['flops', 'latency', 'params']
                }
        return reqs

    def prepare_data(self, info):
        # todo: this can be added at the top of collate_inputs
        val_acc_curve = []
        arch_params = []

        for i in range(len(info)):
            acc_metric = info[i]
            val_acc_curve.append(acc_metric)
        return self.collate_inputs(val_acc_curve, arch_params)

    def fit(self, ytrain, info, learn_hyper=True):

        # prepare training data
        xtrain_data = self.prepare_data(info)  # dimension: n_archs x n_epochs
        y_train = np.array(ytrain)

        # learn hyperparameters of the extrapolator by cross validation
        if self.best_hyper is None or learn_hyper:
            # specify model hyper-parameters
            if self.model_name == 'svr':
                C = loguniform(1e-5, 10, self.n_hypers)
                nu = np.random.uniform(0, 1, self.n_hypers)
                gamma = loguniform(1e-5, 10, self.n_hypers)
                hyper = np.vstack([C, nu, gamma]).T
            elif self.model_name == 'blr':
                alpha_1 = np.random.uniform(1e-7, 1e-5, self.n_hypers)
                alpha_2 = np.random.uniform(1e-7, 1e-5, self.n_hypers)
                lambda_1 = np.random.uniform(1e-7, 1e-5, self.n_hypers)
                lambda_2 = np.random.uniform(1e-7, 1e-5, self.n_hypers)
                hyper = np.vstack([alpha_1, alpha_2, lambda_1, lambda_2]).T
            elif self.model_name == 'rf':
                n_trees = np.random.randint(10, 800, self.n_hypers)
                frac_feature = np.random.uniform(0.1, 0.5, self.n_hypers)
                hyper = np.vstack([n_trees, frac_feature]).T

            print(f'start CV on {self.model_name}')
            mean_score_list = []
            t_start = time.time()
            for i in range(self.n_hypers):
                # define model
                if self.model_name == 'svr':
                    model = NuSVR(C=hyper[i, 0], nu=hyper[i, 1], gamma=hyper[i, 2], kernel='rbf')
                    # model = SVR(C=hyper[i, 0], nu=hyper[i, 1], gamma= ,kernel='linear')
                elif self.model_name == 'blr':
                    model = BayesianRidge(alpha_1=hyper[i, 0], alpha_2=hyper[i, 1],
                                          lambda_1=hyper[i, 2], lambda_2=hyper[i, 3])
                elif self.model_name == 'rf':
                    model = RandomForestRegressor(n_estimators=int(hyper[i, 0]), max_features=hyper[i, 1])
                # perform cross validation to learn the best hyper value
                scores = cross_val_score(model, xtrain_data, y_train, cv=3)
                mean_scores = np.mean(scores)
                mean_score_list.append(mean_scores)
                # print(f'hper={hyper[i]}, score={mean_scores}')
            t_end = time.time()
            best_hyper_idx = np.argmax(mean_score_list)
            best_hyper = hyper[best_hyper_idx]
            max_score = np.max(mean_score_list)
            time_taken = t_end - t_start
            print(f'{self.model_name}'
                  f'best_hyper={best_hyper}, score={max_score}, time={time_taken}')
            self.best_hyper = best_hyper

        # fit the extrapolator with the best hyperparameters to the training data
        if self.model_name == 'svr':
            best_model = NuSVR(C=self.best_hyper[0], nu=self.best_hyper[1], gamma=self.best_hyper[2], kernel='rbf')
            # model = SVR(C=hyper[i, 0], nu=hyper[i, 1], gamma= ,kernel='linear')
        elif self.model_name == 'blr':
            best_model = BayesianRidge(alpha_1=self.best_hyper[0], alpha_2=self.best_hyper[1],
                                       lambda_1=self.best_hyper[2], lambda_2=self.best_hyper[3])
        elif self.model_name == 'rf':
            best_model = RandomForestRegressor(n_estimators=int(self.best_hyper[0]), max_features=self.best_hyper[1])

        best_model.fit(xtrain_data, y_train)
        self.best_model = best_model

    def query(self, info):
        data = self.prepare_data(info)
        pred_on_test_set = self.best_model.predict(data)
        return pred_on_test_set

    def new_epoch(self, epoch):
        # We sample as many architectures as we need 
        if epoch < self.population_size:
            logger.info("Start sampling architectures to fill the population")
            # If there is no scope defined, let's use the search space default one
            
            model = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
            model.arch = self.search_space.clone()
            model.arch.sample_random_architecture(dataset_api=self.dataset_api)
            model.epoch = model.arch.get_max_epochs()
            model.full_lc = model.arch.query(self.performance_metric,
                                            self.dataset,
                                            epoch=model.epoch+1,
                                            dataset_api=self.dataset_api,
                                            full_lc=True)
            model.accuracy = model.full_lc[-1]
            self.info.append(model.full_lc[:self.fidelity])
            self.y_train.append(model.full_lc[self.extrapolation])

            self.population.append(model)
            self._update_history(model)
            log_every_n_seconds(logging.INFO, "Population size {}".format(len(self.population)))
        else:
            if self.train_svr:
                self.fit(self.y_train, self.info)
                self.train_svr = False
            sample = []
            while len(sample) < self.sample_size:
                candidate = np.random.choice(list(self.population))
                sample.append(candidate)
            
            parent = max(sample, key=lambda x: x.accuracy)

            child = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
            child.arch = self.search_space.clone()
            child.arch.mutate(parent.arch, dataset_api=self.dataset_api)
            child.epoch = self.fidelity
            child.partial_lc = child.arch.query(self.performance_metric,
                                              self.dataset,
                                              epoch=child.epoch,
                                              dataset_api=self.dataset_api,
                                              full_lc=True)
            child.accuracy = child.partial_lc[-1]
            prediction = self.query(np.array(child.partial_lc).reshape(1, -1))
            topk = np.sort(np.array(self.y_train))[-int(len(self.y_train) * self.top_n_percent):]
            if prediction > min(topk):
                child.epoch = child.arch.get_max_epochs()
                child.full_lc = child.arch.query(self.performance_metric,
                                                 self.dataset,
                                                 epoch=child.epoch+1,
                                                 dataset_api=self.dataset_api,
                                                 full_lc=True)
                self.info.append(child.full_lc[:self.fidelity])
                self.y_train.append(child.full_lc[self.extrapolation])
                child.accuracy = child.full_lc[-1]
                self.train_svr = True

                self.population.append(child) # maintain the non-proxy population
            self._update_history(child)
        
    def _update_history(self, child):
        self.history.append(child)

    def train_statistics(self):
        best_arch, best_arch_epoch = self.get_final_architecture()
        latest_arch, latest_arch_epoch = self.get_latest_architecture()
        return (
            best_arch.query(Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api, epoch=best_arch_epoch-1),
            best_arch.query(Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api, epoch=best_arch_epoch),
            best_arch.query(Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api, epoch=best_arch_epoch),
            latest_arch.query(Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api, epoch=latest_arch_epoch),
        )

    def test_statistics(self):
        best_arch, epoch = self.get_final_architecture()
        return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api, epoch=epoch)


    def get_final_architecture(self):

        # Returns the sampled architecture with the lowest validation error.
        best_arch = max(self.history, key=lambda x: x.accuracy)
        return best_arch.arch, best_arch.epoch


    def get_latest_architecture(self):

        # Returns the architecture from the most recent epoch
        latest_arch = self.history[-1]
        return latest_arch.arch, latest_arch.epoch


    def get_op_optimizer(self):
        raise NotImplementedError()

    
    def get_checkpointables(self):
        return {'model': self.history}
    

    def get_model_size(self):
        return count_parameters_in_MB(self.history)