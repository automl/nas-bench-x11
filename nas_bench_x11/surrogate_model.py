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

from nas_bench_x11.utils import utils
from nas_bench_x11.encodings.encoding import encode
from nas_bench_x11.utils.data_loaders.nb101_data import get_nb101_data
from nas_bench_x11.utils.data_loaders.nbnlp_data import get_nbnlp_data


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

        # Seeding
        np.random.seed(seed)
        cudnn.benchmark = True
        torch.manual_seed(seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(seed)

        # Create config loader
        root = utils.get_project_root()
        self.config_loader = utils.ConfigLoader(os.path.join(root, 'configs/data_configs/nb301_configspace.json'))

        # Load the data
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            # Add logger
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)

            # Dump the config of the run to log_dir
            self.data_config['seed'] = seed

            logging.info('Loading dataset')
            if self.verbose:
                logging.info('MODEL CONFIG: {}'.format(model_config))
                logging.info('DATA CONFIG: {}'.format(data_config))

            if self.search_space == 'darts':
                self._load_data()
                if self.verbose:
                    logging.info(
                        'DATA: No. train data {}, No. val data {}, No. test data {}'.format(len(self.train_paths),
                                                                                            len(self.val_paths),
                                                                                            len(self.test_paths)))
            else:
                self.train_paths, self.val_paths, self.test_paths = [], [], []

            with open(os.path.join(log_dir, 'model_config.json'), 'w') as fp:
                json.dump(model_config, fp)

            with open(os.path.join(log_dir, 'data_config.json'), 'w') as fp:
                json.dump(data_config, fp)
            
    def load_dataset(self, dataset_type='train', use_full_lc=True, nlp_max_nodes=12):
        """
        Returns specified dataset type for a search space
        TODO: Since darts is different from the other search spaces, we should put
        load_results_from_result_paths into its own file, data_loaders/darts_data.py, just like the other search spaces
        """
        if self.search_space == 'darts':
            if dataset_type == 'train':
                return self.load_results_from_result_paths(self.train_paths, use_full_lc=use_full_lc, extra_feats=False)
            elif dataset_type == 'val':
                return self.load_results_from_result_paths(self.val_paths, use_full_lc=use_full_lc, extra_feats=False)
            elif dataset_type == 'test':
                return self.load_results_from_result_paths(self.test_paths, use_full_lc=use_full_lc, extra_feats=False)

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

    def load_results_from_result_paths(self, result_paths, use_full_lc=False, extra_feats=False):
        """
        Read in the result paths and extract hyperparameters and validation accuracy
        result_paths: list of files containing trained architecture results
        returns list of architecture encodings, val accs, and test accs
        """
        # Get the train/test data
        hyps, val_accuracies, test_accuracies, full_lcs = [], [], [], []

        for result_path in result_paths:
            config_space_instance, val_accuracy, test_accuracy, _, full_lc = self.config_loader[result_path]
            enc = config_space_instance.get_array()
            if len(full_lc) != 98:
                # if the learning curve is less than the maximum length, extend the final accuracy
                full_lc = [*full_lc, *[full_lc[-1]]*(98-len(full_lc))]
            if extra_feats:
                enc = enc.tolist()
                enc.extend(full_lc[:3])
            hyps.append(enc)
            val_accuracies.append(val_accuracy)
            full_lcs.append(full_lc)
            test_accuracies.append(test_accuracy)

        X = np.array(hyps)

        if use_full_lc:
            y = np.array(full_lcs)
        else:
            y = np.array(val_accuracies)

        # Impute none and nan values
        # Essential to prevent segmentation fault with robo
        idx = np.where(y is None)
        y[idx] = 100

        idx = np.isnan(X)
        X[idx] = -1

        return X, y, test_accuracies
            
    def _load_data(self):
        # Get the result train/val/test split
        train_paths = []
        val_paths = []
        test_paths = []
        for key, data_config in self.data_config.items():
            if type(data_config) == dict:
                result_loader = utils.ResultLoader(
                    self.data_root, filepath_regex=data_config['filepath_regex'],
                    train_val_test_split=data_config, seed=self.seed)
                train_val_test_split = result_loader.return_train_val_test()
                # Save the paths
                for paths, filename in zip(train_val_test_split, ['train_paths', 'val_paths', 'test_paths']):
                    file_path = os.path.join(self.log_dir,
                                             pathvalidate.sanitize_filename('{}_{}.json'.format(key, filename)))
                    json.dump(paths, open(file_path, 'w'))

                train_paths.extend(train_val_test_split[0])
                val_paths.extend(train_val_test_split[1])
                test_paths.extend(train_val_test_split[2])

        '''
        # Add extra paths to test
        # Increased ratio of skip-connections.
        matching_files = lambda dir: [str(path) for path in Path(os.path.join(self.data_root, dir)).rglob('*.json')]
        test_paths.extend(matching_files('groundtruths/low_parameter/'))

        # Extreme hyperparameter settings
        # Learning rate
        test_paths.extend(matching_files('groundtruths/hyperparameters/learning_rate/'))
        test_paths.extend(matching_files('groundtruths/hyperparameters/weight_decay/'))

        # Load the blacklist to filter out those elements
        if self.model_config["model"].endswith("_time"):
            blacklist = json.load(open('surrogate_models/configs/data_configs/blacklist_runtimes.json'))
        else:
            blacklist = json.load(open('surrogate_models/configs/data_configs/blacklist.json'))
        filter_out_black_list = lambda paths: list(filter(lambda path: path not in blacklist, paths))
        train_paths, val_paths, test_paths = map(filter_out_black_list, [train_paths, val_paths, test_paths])
        '''
        # Shuffle the total file paths again
        rng = np.random.RandomState(6)
        rng.shuffle(train_paths)
        rng.shuffle(val_paths)
        rng.shuffle(test_paths)

        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths

    def _get_labels_and_preds(self, result_paths):
        """Get labels and predictions from json paths"""
        labels = []
        preds = []
        for result_path in result_paths:
            config_space_instance, val_accuracy_true, test_accuracy_true, _ = self.config_loader[result_path]
            val_pred = self.query(config_space_instance.get_dictionary())
            labels.append(val_accuracy_true)
            preds.append(val_pred)

        return labels, preds

    def _log_predictions(self, result_paths, labels, preds, identifier):
        """Log paths, labels and predictions for one split"""
        if not isinstance(preds[0], float):
            preds = [p[0] for p in preds]

        logdir = os.path.join(self.log_dir, identifier+"_preds.json")

        dump_dict = {"paths": result_paths, "labels": labels, "predictions": preds}
        with open(logdir, "w") as f:
            json.dump(dump_dict, f)

    def log_dataset_predictions(self):
        """Log paths, labels and predictions for train, val, test splits"""
        data_splits = {"train": self.train_paths, "val": self.val_paths, "test": self.test_paths}

        for split_identifier, result_paths in data_splits.items():
            print("==> Logging predictions of %s split" % split_identifier)
            labels, preds = self._get_labels_and_preds(result_paths)
            self._log_predictions(result_paths, labels, preds, split_identifier)

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
