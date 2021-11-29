import collections
import logging
import torch
import copy
import numpy as np

from naslib.predictors.lce.parametric_model import model_name_list, model_config, construct_parametric_model
from naslib.predictors.lce.parametric_ensemble import ParametricEnsemble
from naslib.optimizers.core.metaclasses import MetaOptimizer

from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils.utils import AttrDict, count_parameters_in_MB
from naslib.utils.logging import log_every_n_seconds

logger = logging.getLogger(__name__)
    
        
class REA_LCE(MetaOptimizer):
    
    # training the models is not implemented
    using_step_function = False
    
    def __init__(self, config, metric=Metric.VAL_ACCURACY):
        super().__init__()
        self.config = config
        self.epochs = config.search.epochs
        self.sample_size = config.search.sample_size
        self.population_size = config.search.population_size
        self.fidelity = self.config.search.single_fidelity
        self.default_guess = 0.0
        self.N = 10
        if self.config.search_space == 'nasbench101':
            self.extrapolation = self.config.search.fidelity // 3
            self.top_n_percent = 0.2
        elif self.config.search_space in ["nasbench201", "nasbench211"]:
            self.extrapolation = self.config.search.fidelity // 2
            self.top_n_percent = 0.5
        elif self.config.search_space == 'darts':
            self.extrapolation = self.config.search.fidelity // 2
            self.top_n_percent = 0.2
        elif config.search_space == 'nlp':
            self.extrapolation = config.search.fidelity
            self.top_n_percent = 0.5
        else:
            raise NotImplementedError('{} is not yet implemented yet'.format(config.search_space))

        self.info = []

        self.performance_metric = metric
        self.dataset = config.dataset

        self.population = collections.deque(maxlen=self.population_size)
        self.history = torch.nn.ModuleList()


    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert search_space.QUERYABLE, "Regularized evolution is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
        
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
            self.info.append(model.full_lc[self.extrapolation])
            self.population.append(model)
            self._update_history(model)
            log_every_n_seconds(logging.INFO, "Population size {}".format(len(self.population)))
        else:
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
            ensemble = ParametricEnsemble([construct_parametric_model(model_config, name) for name in model_name_list])
            learning_curves = np.array([inf / 100 for inf in child.partial_lc])
            ensemble.mcmc(learning_curves, N=self.N)
            prediction = ensemble.mcmc_sample_predict([self.extrapolation]) * 100.
            if np.isnan(prediction) or not np.isfinite(prediction):
                prediction = self.default_guess + np.random.rand()
            child.accuracy = child.partial_lc[-1]
            topk = np.sort(np.array(self.info))[-int(len(self.info) * self.top_n_percent):]
            if prediction > min(topk):
                child.epoch = child.arch.get_max_epochs()
                child.full_lc = child.arch.query(self.performance_metric,
                                                self.dataset,
                                                epoch=child.epoch+1,
                                                dataset_api=self.dataset_api,
                                                full_lc=True)
                child.accuracy = child.full_lc[-1]
                self.info.append(child.full_lc[self.extrapolation])
                self.population.append(child)

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
