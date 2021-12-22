import json
import pickle
import logging
import os
import sys
from abc import ABC, abstractmethod
import numpy as np
import pathvalidate
import torch
import torch.backends.cudnn as cudnn

from nas_bench_x11.encodings.encoding import encode
from nas_bench_x11.utils.data_loaders.nb101_data import get_nb101_data
from nas_bench_x11.utils.data_loaders.nbnlp_data import get_nbnlp_data
from nas_bench_x11.utils.data_loaders.darts_data import load_darts_strings, load_darts_data


class SurrogateModel(ABC):
    def __init__(self, data_root, log_dir, seed, model_config, data_config, search_space, nb101_api):
        self.data_root = data_root
        self.log_dir = log_dir
        self.model_config = model_config
        self.data_config = data_config
        self.seed = seed
        self.search_space = search_space
        self.nb101_api = nb101_api
        self.verbose = False

        # set random seeds
        np.random.seed(seed)
        cudnn.benchmark = True
        torch.manual_seed(seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(seed)

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            # Add logger
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)

            # todo: unify data configs
            if self.verbose:
                logging.info('MODEL CONFIG: {}'.format(model_config))
                logging.info('DATA CONFIG: {}'.format(data_config))

            with open(os.path.join(log_dir, 'model_config.json'), 'w') as fp:
                json.dump(model_config, fp)

            with open(os.path.join(log_dir, 'data_config.json'), 'w') as fp:
                json.dump(data_config, fp)
            
    def load_dataset(self, dataset_type='train', use_full_lc=True, nlp_max_nodes=12):
        """
        Returns specified dataset type for a search space
        TODO: unify the way the darts and nb201/nlp/nb101 search spaces are loaded.
        """
        if self.search_space == 'darts':
            train_strings, val_strings, test_strings = load_darts_strings(self.data_root, self.seed)
            if dataset_type == 'train':
                return load_darts_data(train_strings, use_full_lc=use_full_lc, extra_feats=False)
            elif dataset_type == 'val':
                return load_darts_data(val_strings, use_full_lc=use_full_lc, extra_feats=False)
            elif dataset_type == 'test':
                return load_darts_data(test_strings, use_full_lc=use_full_lc, extra_feats=False)

        elif self.search_space in ['nb201', 'nlp', 'nb101']:
            if self.search_space == 'nb201':
                with open(os.path.join(self.data_root, 'nb201_cifar10_full_training.pickle'), 'rb') as f:
                    data = pickle.load(f)
            elif self.search_space == 'nlp':
                data = get_nbnlp_data(self.data_root, nlp_max_nodes)
            elif self.search_space == 'nb101':
                data = get_nb101_data(data_root=self.data_root)

            np.random.seed(0)
            arch_strings = list(data.keys())
            n = len(arch_strings)
            random_order = [i for i in range(n)]
            np.random.shuffle(random_order)
            split_indices = [int(0.8*n), int(0.9*n), -1] # 0.9
            if dataset_type == 'train':
                train_strings = [arch_strings[i] for i in random_order[:split_indices[0]]]
                return encode(train_strings, data, 
                              search_space=self.search_space, 
                              nlp_max_nodes=nlp_max_nodes, 
                              nb101_api=self.nb101_api)
            elif dataset_type == 'val':
                val_strings = [arch_strings[i] for i in random_order[split_indices[0]:split_indices[1]]]
                return encode(val_strings, data, 
                              search_space=self.search_space, 
                              nlp_max_nodes=nlp_max_nodes, 
                              nb101_api=self.nb101_api)
            elif dataset_type == 'test':
                test_strings = [arch_strings[i] for i in random_order[split_indices[1]:split_indices[2]]]
                return encode(test_strings, data, 
                              search_space=self.search_space, 
                              nlp_max_nodes=nlp_max_nodes, 
                              nb101_api=self.nb101_api)

        else:
            raise NotImplementedError()

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def validate(self):
        raise NotImplementedError()

    @abstractmethod
    def test(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self):
        raise NotImplementedError()

    @abstractmethod
    def load(self, model_path):
        raise NotImplementedError()

    @abstractmethod
    def query(self, config_dict):
        raise NotImplementedError()
