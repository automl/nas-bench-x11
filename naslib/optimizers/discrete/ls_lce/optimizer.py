import collections
import logging
import torch
import copy
import random
import numpy as np

from naslib.predictors.lce.parametric_model import model_name_list, model_config, construct_parametric_model
from naslib.predictors.lce.parametric_ensemble import ParametricEnsemble

from naslib.optimizers.core.metaclasses import MetaOptimizer

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench201.graph import NasBench201SearchSpace

from naslib.utils.utils import AttrDict, count_parameters_in_MB
from naslib.utils.logging import log_every_n_seconds

logger = logging.getLogger(__name__)

class LS_LCE(MetaOptimizer):
    
    # training the models is not implemented
    using_step_function = False
    def __init__(self, config, metric=Metric.VAL_ACCURACY):

        super().__init__()
        self.config = config
        self.epochs = config.search.epochs
        self.fidelity = self.config.search.single_fidelity
        self.default_guess = 0.0
        self.N = 30
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
            self.top_n_percent = 0.2
        else:
            raise NotImplementedError('{} is not yet implemented yet'.format(config.search_space))

        self.info = []

        self.performance_metric = metric
        self.dataset = config.dataset
        
        self.num_init = config.search.num_init
        self.nbhd = []
        self.chosen = None
        self.best_arch = None
        
        self.history = torch.nn.ModuleList()


    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert search_space.QUERYABLE, "Local search is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api

    def new_epoch(self, epoch):

        if epoch < self.num_init:
            # randomly sample initial architectures 
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

            if not self.best_arch or model.accuracy > self.best_arch.accuracy:
                self.best_arch = model
            self._update_history(model)

        else:
            if len(self.nbhd) == 0 and self.chosen and self.best_arch.accuracy <= self.chosen.accuracy:
                logger.info('Reached local minimum. Starting from new random architecture.')
                model = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
                model.arch = self.search_space.clone()
                model.arch.sample_random_architecture(dataset_api=self.dataset_api)
                model.epoch = model.arch.get_max_epochs()
                model.full_lc = model.arch.query(self.performance_metric,
                                                 self.dataset,
                                                 epoch=model.epoch + 1,
                                                 dataset_api=self.dataset_api,
                                                 full_lc=True)
                model.accuracy = model.full_lc[-1]
                self.info.append(model.full_lc[self.extrapolation])

                self.chosen = model
                self.best_arch = model
                self.nbhd = self.chosen.arch.get_nbhd(dataset_api=self.dataset_api)

            else:
                if len(self.nbhd) == 0:
                    logger.info('Start a new iteration. Pick the best architecture and evaluate its neighbors.')
                    self.chosen = self.best_arch
                    self.nbhd = self.chosen.arch.get_nbhd(dataset_api=self.dataset_api)

                model = self.nbhd.pop()
                model.epoch = self.fidelity
                model.partial_lc = model.arch.query(self.performance_metric,
                                              self.dataset,
                                              epoch=model.epoch,
                                              dataset_api=self.dataset_api,
                                              full_lc=True)
                ensemble = ParametricEnsemble([construct_parametric_model(model_config, name) for name in model_name_list])
                learning_curves = np.array([inf / 100 for inf in model.partial_lc])
                ensemble.mcmc(learning_curves, N=self.N)
                prediction = ensemble.mcmc_sample_predict([self.extrapolation]) * 100.
                if np.isnan(prediction) or not np.isfinite(prediction):
                    prediction = 0.0
                model.accuracy = model.partial_lc[-1]
                topk = np.sort(np.array(self.info))[-int(len(self.info) * self.top_n_percent):]
                if prediction > min(topk):
                    model.epoch = model.arch.get_max_epochs()
                    model.full_lc = model.arch.query(self.performance_metric,
                                                     self.dataset,
                                                     epoch=model.epoch+1,
                                                     dataset_api=self.dataset_api,
                                                     full_lc=True)
                    self.info.append(model.full_lc[self.extrapolation])
                    model.accuracy = model.full_lc[-1]
                    if model.accuracy > self.best_arch.accuracy:
                        self.best_arch = model
                        logger.info('Found new best architecture.')
                self._update_history(model)           
                        
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
