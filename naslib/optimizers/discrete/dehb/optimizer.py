import collections
import math
import logging
import torch
import copy
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer

from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils.utils import AttrDict, count_parameters_in_MB
from naslib.utils.logging import log_every_n_seconds

logger = logging.getLogger(__name__)
    
        
class DEHB(MetaOptimizer):
    
    # training the models is not implemented
    using_step_function = False
    
    def __init__(self, config):
        super().__init__()
        # Hyperband related stuff
        self.config = config
        self.rounds = []
        self.round_sizes = []
        self.fidelities = []
        if self.config.search_space in ["nasbench201", "nasbench211"]:
            self.max_budget = 512 # SH according to cosine decay
            self.min_budget = 12
            self.eta = 3  # defines downsampling rate (default=3)
            self.enc_dim = 6
            self.max_mutations = 1
            self.crossover_prob = 0.5
            self.top_n_percent = 0.3
            self.mutate_prob = 0.1
        elif self.config.search_space == "nasbench101":
            self.max_budget = 108
            self.min_budget = 4
            self.eta = 3  # defines downsampling rate (default=3)
            self.enc_dim = 26
            self.max_mutations = 1
            self.crossover_prob = 0.5
            self.top_n_percent = 0.3
            self.mutate_prob = 0.1
        elif self.config.search_space == "darts":
            self.max_budget = 98
            self.min_budget = 4
            self.eta = 3  # defines downsampling rate (default=3)
            self.enc_dim = 32
            self.max_mutations = 5
            self.crossover_prob = 0.5
            self.top_n_percent = 0.3
            self.mutate_prob = 0.1
        elif self.config.search_space == "nlp":
            self.max_budget = 50
            self.min_budget = 3
            self.eta = 3  # defines downsampling rate (default=3)
            self.enc_dim = 185
            self.max_mutations = 8
            self.crossover_prob = 0.8
            self.top_n_percent = 0.3
            self.mutate_prob = 0.2
        else:
            raise NotImplementedError()
        self.de = dict()
        self._epsilon = 1e-6
        self.pop_size = {}
        self.counter = 0
        s_max = math.floor(math.log(self.max_budget / self.min_budget, self.eta) + self._epsilon)
        # set up round sizes, fidelities, and list of arches
        for s in reversed(range(s_max + 1)):
            self.rounds.append(s)
            round_sizes = []
            fidelities = []
            n = math.ceil((s_max + 1) * self.eta ** s / (s + 1) - self._epsilon) # initial number of configurations
            r = self.max_budget / self.eta**s # initial number of iterations to run configurations for
            for i in range(s + 1):
                n_i = math.floor(n / self.eta ** i + self._epsilon)
                r_i = min(math.floor(r * self.eta ** i + self._epsilon), config.search.fidelity)
                round_sizes.append(n_i)
                fidelities.append(r_i)
                self.pop_size[r_i] = self.pop_size.get(r_i, 0) + n_i
            self.round_sizes.append(round_sizes)
            self.fidelities.append(fidelities)
        for budget in self.fidelities[0][1:]:
            budget = min(budget, config.search.fidelity)
            self.de[budget] = {}
            self.de[budget]['promotions'] = collections.deque(maxlen=100)

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset
        self.history = torch.nn.ModuleList()


        self.epochs = self.compute_epochs()
        self.current_round = []
        self.next_round = []
        self.round_number = 0
        self.prev_round = 0
        self.counter = 0
        self.process = 0

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert search_space.QUERYABLE, "Hyperband_simple is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
        self.max_training_epoch = self.search_space.get_max_epochs()

    def compute_epochs(self):
        return self.round_sizes, self.rounds[::-1]

    def new_epoch(self, epoch, round, i):
        if self.process < i: # re-init for each new process
            self.current_round = []
            self.next_round = []
            self.round_number = 0
            self.prev_round = 0
            self.counter = 0
            self.process = i

        if self.prev_round < round:  # reset round_number for each new round
            self.prev_round = round
            self.round_number = 0

        if epoch < self.round_sizes[round][0]:
            # sample random architectures
            model = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
            model.arch = self.search_space.clone()
            budget = self.fidelities[round][0]
            if round == 0:
                model.arch.sample_random_architecture(dataset_api=self.dataset_api)
            else:
                if len(self.de[budget]['promotions']) > 0:
                    print('promotion from budget: {}, length: {}'.format(budget, len(self.de[budget]['promotions'])))
                    model = self.de[budget]['promotions'].pop()
                    model = copy.deepcopy(model)
                    arch = self.search_space.clone()
                    arch.mutate(model.arch, dataset_api=self.dataset_api)
                    model.arch = arch
                else:
                    model.arch.sample_random_architecture(dataset_api=self.dataset_api)

            model.epoch = min(self.fidelities[round][0], self.max_training_epoch)
            model.accuracy = model.arch.query(self.performance_metric,
                                              self.dataset,
                                              epoch=model.epoch,
                                              dataset_api=self.dataset_api)

            self._update_history(model)
            self.next_round.append(model)

        else:
            if len(self.current_round) == 0:
                # if we are at the end of a round of hyperband, continue training only the best
                logger.info("Starting a new round: continuing to train the best arches")
                self.counter = 0
                cutoff = math.ceil(self.round_sizes[round][self.round_number] * self.top_n_percent)
                self.current_round = sorted(self.next_round, key=lambda x: -x.accuracy)[:cutoff]
                self.round_number += 1
                self.round_number = min(self.round_number, len(self.fidelities[round]) - 1)
                self.next_round = []

            # train the next architecture
            model = self.current_round.pop()
            self.counter += 1
            """
            Note: technically we would just continue training this arch, but right now,
            just for simplicity, we treat it as if we start to train it again from scratch
            """
            model = copy.deepcopy(model)

            if np.random.rand(1) < self.mutate_prob:
                candidate = model.arch.clone()
                for _ in range(self.max_mutations):
                    arch_ = self.search_space.clone()
                    arch_.mutate(candidate, dataset_api=self.dataset_api)
                    candidate = arch_
                mutant = candidate
                arch = self.search_space.clone()
                arch.crossover_bin(model.arch, mutant, self.enc_dim, self.crossover_prob, dataset_api=self.dataset_api)
                model.arch = arch
            model.epoch = min(self.fidelities[round][self.round_number], self.max_training_epoch)
            model.accuracy = model.arch.query(self.performance_metric,
                                              self.dataset,
                                              epoch=model.epoch,
                                              dataset_api=self.dataset_api)
            self.de[self.fidelities[round][self.round_number]]['promotions'].append(model)
            self._update_history(model)
            self.next_round.append(model)

    def _update_history(self, child):
        self.history.append(child)

    def get_final_architecture(self):
        
        # Returns the sampled architecture with the lowest validation error.
        best_arch = max(self.history, key=lambda x: x.accuracy)
        return best_arch.arch, best_arch.epoch

    def get_latest_architecture(self):

        # Returns the architecture from the most recent epoch
        latest_arch = self.history[-1]
        return latest_arch.arch, latest_arch.epoch

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

    
    def get_op_optimizer(self):
        raise NotImplementedError()

    def get_checkpointables(self):
        return {'model': self.history}

    def get_model_size(self):
        return count_parameters_in_MB(self.history)