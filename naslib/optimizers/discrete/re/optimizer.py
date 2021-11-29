import collections
import logging
import torch
import copy
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer

from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils.utils import AttrDict, count_parameters_in_MB
from naslib.utils.logging import log_every_n_seconds

logger = logging.getLogger(__name__)
    
        
class RegularizedEvolution(MetaOptimizer):
    
    # training the models is not implemented
    using_step_function = False
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.epochs = config.search.epochs
        self.sample_size = config.search.sample_size
        self.population_size = config.search.population_size

        self.performance_metric = Metric.VAL_ACCURACY
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
            model.accuracy = model.arch.query(self.performance_metric, 
                                              self.dataset, 
                                              dataset_api=self.dataset_api,
                                              epoch=model.epoch)
            
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
            child.epoch = child.arch.get_max_epochs()
            child.accuracy = child.arch.query(self.performance_metric,
                                              self.dataset, 
                                              dataset_api=self.dataset_api,
                                              epoch=child.epoch)

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