"""
This file contains code based on
https://github.com/automl/nasbench301/
Authors: Julien Siems, Lucas Zimmer, Arber Zela, Jovita Lukasik, Margret Keuper, Frank Hutter
"""

import glob
import itertools
import json
import os
import re
from math import isclose
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import seaborn as sns
from ConfigSpace.read_and_write import json as config_space_json_r_w
from scipy.stats import norm, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from pathlib import Path

from nas_bench_x11.models.lgboost import LGBModel, LGBModelTime
from nas_bench_x11.models.xgboost import XGBModel, XGBModelTime
from nas_bench_x11.models.svd_lgb import SVDLGBModel
from nas_bench_x11.models.svd_xgb import SVDXGBModel
from nas_bench_x11.models.svd_nn import SVDNNModel
from nas_bench_x11.models.vae_nn import VAENNModel
from nas_bench_x11.models.vae_lgb import VAELGBModel
from nas_bench_x11.models.vae_xgb import VAEXGBModel

sns.set_style('whitegrid')

model_dict = {

    # NOTE: RUNTIME MODELS SHOULD END WITH "_time"

    'xgb': XGBModel,
    'svd_lgb': SVDLGBModel,
    'svd_xgb': SVDXGBModel,
    'svd_nn': SVDNNModel,
    'vae_lgb': VAELGBModel,
    'vae_xgb': VAEXGBModel,
    'vae_nn': VAENNModel,
    'xgb_time': XGBModelTime,
    'lgb': LGBModel,
    'lgb_time': LGBModelTime,
}


def get_project_root() -> Path:
    """
    Returns the root path of the project.
    """
    return Path(__file__).parent.parent


def evaluate_learning_curve_metrics(y_true, y_pred, prediction_is_first_arg, reduction='mean'):
    """
    Create a dict with all evaluation metrics
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    reduce_fn = {'mean': np.mean}[reduction]

    if prediction_is_first_arg:
        y_true, y_pred = y_pred, y_true

    def kendall_tau_lc(y_true, y_pred, decimals=None):
        corrs, p_vals = [], []

        for yt, yp in zip(y_true.T, y_pred.T):
            if decimals is None:
                corr, p_val = kendalltau(yt, yp)
            else:
                corr, p_val = kendalltau(yt, np.round(np.array(yp), decimals=decimals))
            corrs.append(corr)
            p_vals.append(p_val)
        return corrs

    metrics_dict = dict()
    metrics_dict["mse"] = reduce_fn(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    metrics_dict["rmse"] = reduce_fn(mean_squared_error(y_true, y_pred, multioutput='raw_values', squared=False))
    metrics_dict["r2"] = reduce_fn(r2_score(y_true, y_pred, multioutput='raw_values'))
    metrics_dict["kendall_tau"] = reduce_fn(kendall_tau_lc(y_true, y_pred))
    metrics_dict["kendall_tau_2_dec"] = reduce_fn(kendall_tau_lc(y_true, y_pred, decimals=2))
    metrics_dict["kendall_tau_1_dec"] = reduce_fn(kendall_tau_lc(y_true, y_pred, decimals=1))

    metrics_dict["spearmanr"] = reduce_fn([spearmanr(yt, yp).correlation for yt, yp in zip(y_true.T, y_pred.T)])

    # Last epoch metrics

    metrics_dict["last_epoch_mse"] = mean_squared_error(y_true[:, -1], y_pred[:, -1])
    metrics_dict["last_epoch_rmse"] = np.sqrt(metrics_dict["last_epoch_mse"])
    metrics_dict["last_epoch_r2"] = r2_score(y_true[:, -1], y_pred[:, -1])
    metrics_dict["last_epoch_kendall_tau"], p_val = kendalltau(y_true[:, -1], y_pred[:, -1])
    metrics_dict["last_epoch_kendall_tau_2_dec"], p_val = kendalltau(y_true[:, -1], np.round(np.array(y_pred[:, -1]), decimals=2))
    metrics_dict["last_epoch_kendall_tau_1_dec"], p_val = kendalltau(y_true[:, -1], np.round(np.array(y_pred[:, -1]), decimals=1))

    metrics_dict["last_epoch_spearmanr"] = spearmanr(y_true[:, -1], y_pred[:, -1]).correlation

    return metrics_dict


def evaluate_metrics(y_true, y_pred, prediction_is_first_arg):
    """
    Create a dict with all evaluation metrics
    """

    if prediction_is_first_arg:
        y_true, y_pred = y_pred, y_true

    metrics_dict = dict()
    metrics_dict["mse"] = mean_squared_error(y_true, y_pred)
    metrics_dict["rmse"] = np.sqrt(metrics_dict["mse"])
    metrics_dict["r2"] = r2_score(y_true, y_pred)
    metrics_dict["kendall_tau"], p_val = kendalltau(y_true, y_pred)
    metrics_dict["kendall_tau_2_dec"], p_val = kendalltau(y_true, np.round(np.array(y_pred), decimals=2))
    metrics_dict["kendall_tau_1_dec"], p_val = kendalltau(y_true, np.round(np.array(y_pred), decimals=1))

    metrics_dict["spearmanr"] = spearmanr(y_true, y_pred).correlation

    return metrics_dict


def get_model_configspace(model):
    """
    Retrieve the model_config
    :param model: Name of the model for which you want the default config
    :return:
    """
    # Find matching config for the model name
    model_config_regex = re.compile(".*{}_configspace.json".format(model))
    root = get_project_root()
    matched_model_config_paths = list(
        filter(model_config_regex.match, glob.glob(str(root) + '/configs/model_configs/*')))

    print(matched_model_config_paths)
    # Make sure we only matched exactly one config
    assert len(matched_model_config_paths) == 1, 'Multiple or no configs matched with the requested model.'
    model_config_path = matched_model_config_paths[0]

    # Load the configspace object
    model_configspace = config_space_json_r_w.read(open(model_config_path, 'r').read())
    return model_configspace


def convert_array_to_list(a):
    """Converts a numpy array to list"""

    if isinstance(a, np.ndarray):
        return a.tolist()
    else:
        return a


class ConfigLoader:
    def __init__(self, config_space_path):
        self.config_space = self.load_config_space(config_space_path)

        # The exponent to scale the fidelity with.
        # Used to move architectures across the fidelity budgets
        # Default at None, hence the fidelity values are not changed
        self.fidelity_exponent = None

        # The number of skip connections to have in the cell
        # If this set to None (default) No skip connections will be added to the cell
        # Maximum is the maximum number of operations.
        self.parameter_free_op_increase_type = None
        self.ratio_parameter_free_op_in_cell = None

        # Manually adjust a certain set of hyperparameters
        self.parameter_change_dict = None

        # Save predefined fidelity multiplier
        self.fidelity_multiplier = {
            'SimpleLearningrateSchedulerSelector:cosine_annealing:T_max': 1.762734383267615,
            'NetworkSelectorDatasetInfo:darts:init_channels': 1.3572088082974532,
            'NetworkSelectorDatasetInfo:darts:layers': 1.2599210498948732
        }
        self.fidelity_starts = {
            'SimpleLearningrateSchedulerSelector:cosine_annealing:T_max': 50,
            'NetworkSelectorDatasetInfo:darts:init_channels': 8,
            'NetworkSelectorDatasetInfo:darts:layers': 5
        }

    def __getitem__(self, path, return_full_lc=True):
        """
        Load the results from results.json
        :param path: Path to results.json
        :return:
        """
        json_file = json.load(open(path, 'r'))
        config_dict = json_file['optimized_hyperparamater_config']
        config_space_instance = self.query_config_dict(config_dict)
        val_accuracy = json_file['info'][0]['val_accuracy']
        test_accuracy = json_file['test_accuracy']
        if return_full_lc:
            full_lc = json_file['learning_curves']['Train/val_accuracy']
            return config_space_instance, val_accuracy, test_accuracy, json_file, full_lc
        else:
            # this is the original nasbench301 return
            return config_space_instance, val_accuracy, test_accuracy, json_file

    def get_runtime(self, path):
        """
        Load the runtime from results.json
        :param path: Path to results.json
        return:
        """
        json_file = json.load(open(path, 'r'))
        config_dict = json_file['optimized_hyperparamater_config']

        config_space_instance = self.query_config_dict(config_dict)
        runtime = json_file['runtime']
        return config_space_instance, runtime

    def query_config_dict(self, config_dict):
        # Evaluation methods
        # Scale the hyperparameters if needed
        if self.fidelity_exponent is not None:
            config_dict = self.scale_fidelity(config_dict)

        # Add selected parameter free op
        if self.ratio_parameter_free_op_in_cell is not None:
            config_dict = self.add_selected_parameter_free_op(config_dict)

        # Change a selection of parameters
        if self.parameter_change_dict is not None:
            config_dict = self.change_parameter(config_dict)

        # Create the config space instance based on the config space
        config_space_instance = \
            self.convert_config_dict_to_configspace_instance(self.config_space, config_dict=config_dict)

        return config_space_instance

    def add_selected_parameter_free_op(self, config_dict):
        """
        Add selected parameter free operation to the config dict
        :param config_dict:
        :return:
        """
        assert self.parameter_free_op_increase_type in ['max_pool_3x3',
                                                        'avg_pool_3x3',
                                                        'skip_connect'], 'Unknown parameter-free op was selected.'
        # Dictionary containing operations
        cell_op_dict_sel_param_free = {'normal': {}, 'reduce': {}}
        cell_op_dict_non_sel_param_free = {'normal': {}, 'reduce': {}}

        for cell_type in ['normal']:
            for edge in range(0, 14):
                key = 'NetworkSelectorDatasetInfo:darts:edge_{}_{}'.format(cell_type, edge)
                op = config_dict.get(key, None)
                if op is not None:
                    if op == self.parameter_free_op_increase_type:
                        cell_op_dict_sel_param_free[cell_type][key] = op
                    else:
                        cell_op_dict_non_sel_param_free[cell_type][key] = op

        # Select random subset of operations which to turn to selected parameter-free op
        for cell_type in ['normal', 'reduce']:
            num_sel_param_free_ops = len(cell_op_dict_sel_param_free[cell_type].values())
            num_non_sel_param_free_ops = len(cell_op_dict_non_sel_param_free[cell_type].values())

            num_ops = num_sel_param_free_ops + num_non_sel_param_free_ops
            desired_num_sel_param_free_ops = np.round(num_ops * self.ratio_parameter_free_op_in_cell).astype(np.int)
            remaining_num_sel_param_free_op = desired_num_sel_param_free_ops - num_sel_param_free_ops

            if remaining_num_sel_param_free_op > 0:
                # There are still more selected parameter free operations to add to satisfy the ratio of
                # sel param free op. Therefore override some of the other operations to be parameter free op.
                sel_param_free_idx = np.random.choice(num_non_sel_param_free_ops, remaining_num_sel_param_free_op,
                                                      replace=False)
                for idx, (key, value) in enumerate(cell_op_dict_non_sel_param_free[cell_type].items()):
                    if idx in sel_param_free_idx:
                        config_dict[key] = self.parameter_free_op_increase_type
        return config_dict

    def scale_fidelity(self, config_dict):
        """
        Scale the fidelity of the current sample
        :param config_dict:
        :return:
        """
        for name, value in self.fidelity_multiplier.items():
            config_dict[name] = int(config_dict[name] * value ** self.fidelity_exponent)
        return config_dict

    def change_parameter(self, config_dict):
        for name, value in self.parameter_change_dict.items():
            config_dict[name] = value
        return config_dict

    def convert_config_dict_to_configspace_instance(self, config_space, config_dict):
        """
        Convert a config dictionary to configspace instace
        :param config_space:
        :param config_dict:
        :return:
        """

        def _replace_str_bool_with_python_bool(input_dict):
            for key, value in input_dict.items():
                if value == 'True':
                    input_dict[key] = True
                elif value == 'False':
                    input_dict[key] = False
                else:
                    pass
            return input_dict

        # Replace the str true with python boolean type
        config_dict = _replace_str_bool_with_python_bool(config_dict)
        config_instance = CS.Configuration(config_space, values=config_dict)
        return config_instance

    @staticmethod
    def load_config_space(path):
        """
        Load ConfigSpace object
        As certain hyperparameters are not denoted as optimizable but overriden later,
        they are manually overriden here too.
        :param path:
        :return:
        """
        with open(os.path.join(path), 'r') as fh:
            json_string = fh.read()
            config_space = config_space_json_r_w.read(json_string)

        # Override the constant hyperparameters for num_layers, init_channels and
        config_space._hyperparameters.pop('NetworkSelectorDatasetInfo:darts:layers', None)
        num_layers = CSH.UniformIntegerHyperparameter(name='NetworkSelectorDatasetInfo:darts:layers', lower=1,
                                                      upper=10000)
        config_space._hyperparameters.pop('SimpleLearningrateSchedulerSelector:cosine_annealing:T_max', None)
        t_max = CSH.UniformIntegerHyperparameter(name='SimpleLearningrateSchedulerSelector:cosine_annealing:T_max',
                                                 lower=1, upper=10000)
        config_space._hyperparameters.pop('NetworkSelectorDatasetInfo:darts:init_channels', None)
        init_channels = CSH.UniformIntegerHyperparameter(name='NetworkSelectorDatasetInfo:darts:init_channels', lower=1,
                                                         upper=10000)
        config_space._hyperparameters.pop('SimpleLearningrateSchedulerSelector:cosine_annealing:eta_min', None)
        eta_min_cosine = CSH.UniformFloatHyperparameter(
            name='SimpleLearningrateSchedulerSelector:cosine_annealing:eta_min', lower=0, upper=10000)

        config_space.add_hyperparameters([num_layers, t_max, init_channels, eta_min_cosine])
        return config_space

    def get_config_without_architecture(self, config_instance):
        """
        Remove the architecture parameters from the config.
        Currently this function retrieves the 5 parameters which are actually changed throughout the results:
        num_epochs, num_layers, num_init_channels (3 fidelities) + learning_rate, weight_decay
        :param config_instance:
        :return:
        """
        non_arch_hyperparameters_list = [
            config_instance._values['SimpleLearningrateSchedulerSelector:cosine_annealing:T_max'],
            config_instance._values['NetworkSelectorDatasetInfo:darts:init_channels'],
            config_instance._values['NetworkSelectorDatasetInfo:darts:layers'],
            config_instance._values['OptimizerSelector:sgd:learning_rate'],
            config_instance._values['OptimizerSelector:sgd:weight_decay']]

        return non_arch_hyperparameters_list


class ResultLoader:
    def __init__(self, root, filepath_regex, train_val_test_split, seed):
        self.root = root
        self.filepath_regex = filepath_regex
        self.train_val_test_split = train_val_test_split
        np.random.seed(seed)

    def return_train_val_test(self):
        """
        Get the result train/val/test split.
        :return:
        """
        if self.train_val_test_split['type'] == 'all_result_paths':
            paths_split = self.all_result_paths()
        elif self.train_val_test_split['type'] == 'filtered_result_paths':
            paths_split = self.filtered_result_paths()
        elif self.train_val_test_split['type'] == 'per_budget_equal_result_paths':
            paths_split = self.per_budget_equal_result_paths()
        elif self.train_val_test_split['type'] == 'per_subfolder_equal_ratio':
            paths_split = self.per_subfolder_equal_ratio()
        elif self.train_val_test_split['type'] == 'no_data':
            paths_split = [], [], []
        else:
            raise ValueError('Unknown train/val/test split.')
        train_paths, val_paths, test_paths = paths_split
        return train_paths, val_paths, test_paths

    def filter_duplicate_dirs(self, paths_to_json):
        """
        Checks to configurations in the results.json files and returns paths such that none contains
        duplicate configurations.
        :param paths_to_json: List of dir/results.json
        :return: unique list of dir/results.json w.r.t. configuration
        """
        config_hashes = []

        for path_to_json in paths_to_json:
            with open(path_to_json, "r") as f:
                results = json.load(f)
            config_hash = hash(results["optimized_hyperparamater_config"].__repr__())
            config_hashes.append(config_hash)

        _, unique_indices = np.unique(config_hashes, return_index=True)

        return list(np.array(paths_to_json)[unique_indices])

    def get_splits(self, paths, ratios=None):
        """
        Divide the paths into train/val/test splits.
        :param paths:
        :param ratios:
        :return:
        """
        if ratios is None:
            train_ratio, val_ratio, test_ratio = self.train_val_test_split['train'], self.train_val_test_split['val'], \
                self.train_val_test_split['test']
        else:
            train_ratio, val_ratio, test_ratio = ratios
        assert isclose(train_ratio + val_ratio + test_ratio, 1.0,
                       abs_tol=1e-8), 'The train/val/test split should add up to 1.'

        # Randomly shuffle the list
        rng = np.random.RandomState(6)
        rng.shuffle(paths)

        # Extract the train/val/test splits
        train_upper_idx = int(train_ratio * len(paths))
        val_upper_idx = int((train_ratio + val_ratio) * len(paths))

        train_paths = paths[:train_upper_idx]
        val_paths = paths[train_upper_idx:val_upper_idx]
        test_paths = paths[val_upper_idx:-1]
        return train_paths, val_paths, test_paths

    def all_result_paths(self, verbose=False):
        """
        Return the paths of all results
        :return: result paths
        """
        all_results_paths = glob.glob(os.path.join(self.root, self.filepath_regex))
        if verbose:
            print("==> Found %i results paths. Filtering duplicates..." % len(all_results_paths))
        all_results_paths.sort()
        all_results_paths_filtered = self.filter_duplicate_dirs(all_results_paths)
        if verbose:
            print("==> Finished filtering. Found %i unique architectures, %i duplicates" % (len(all_results_paths_filtered),
                                                                                            len(all_results_paths) - len(
                                                                                                all_results_paths_filtered)))
        train_paths, val_paths, test_paths = self.get_splits(all_results_paths_filtered)
        return train_paths, val_paths, test_paths

    def per_subfolder_equal_ratio(self):
        """
        :return:
        """
        train_paths, val_paths, test_paths = [], [], []
        for subdir in os.listdir(os.path.join(self.root, self.filepath_regex)):
            subdir_path = os.path.join(self.root, self.filepath_regex, subdir)

            # For each subdir split according to the train_val_test_ratios
            files_in_subdir = glob.glob(os.path.join(subdir_path, '*'))
            files_in_subdir.sort()
            train, val, test = self.get_splits(files_in_subdir)

            # Add the train paths
            train_paths.extend(train)
            val_paths.extend(val)
            test_paths.extend(test)
        return train_paths, val_paths, test_paths

    def filtered_result_paths(self):
        """
        Return only the paths of the results that match the filter
        :return: result paths
        """
        # Check result filters have been specified
        assert self.train_val_test_split.get('filters', None) is not None, 'Can\'t filter without a result filter.'
        # Train/val and test split should not be the same filter
        assert self.train_val_test_split['filters']['train_val_filter'] != self.train_val_test_split['filters'][
            'test_filter'], 'Train/Val filter should not be the same as the test filter.'
        all_results_paths = glob.glob(os.path.join(self.root, 'run_*/results_fidelity_*/results_*.json'))
        all_results_paths.sort()

        results_per_filter = {result_filter: [] for result_filter in self.train_val_test_split.get('filters').keys()}
        for result_path in tqdm(all_results_paths, desc='Filtering results'):
            result_json = json.load(open(result_path, 'r'))
            # Go through all elements to be filtered
            for result_filter_name, result_filter_path in self.train_val_test_split.get('filters').items():
                result_filter = json.load(open(result_filter_path, 'r'))
                results = []
                for filter_key, filter_details in result_filter.items():
                    # Retrieve the element to be checked
                    filtered_value = list(find_key_value(filter_key, result_json))
                    if len(filtered_value):
                        if filter_details['type'] == "interval":
                            # Check if the configuration matches the filter interval
                            lower_filter_val, high_filter_val = filter_details['data']
                            if lower_filter_val <= filtered_value[0] <= high_filter_val:
                                results.append(result_path)
                            else:
                                continue
                        elif filter_details['type'] == "list":
                            # Check whether the value is in a list of pre-specified values
                            if filtered_value[0] in filter_details['data']:
                                results.append(result_path)
                            else:
                                continue
                        else:
                            pass
                if len(results) == len(result_filter.keys()):
                    results_per_filter[result_filter_name].append(results[0])
        # Split the train/val split
        new_train_ratio = self.train_val_test_split['train'] / (
            self.train_val_test_split['train'] + self.train_val_test_split['val'])
        new_val_ratio = self.train_val_test_split['val'] / (
            self.train_val_test_split['train'] + self.train_val_test_split['val'])
        train_paths, val_paths, _ = self.get_splits(results_per_filter['train_val_filter'],
                                                    (new_train_ratio, new_val_ratio, 0.0))
        test_paths = results_per_filter['test_filter']
        assert len(set(results_per_filter['train_val_filter']).intersection(
            set(test_paths))) == 0, 'Train/val and test set are not disjoint.'
        return train_paths, val_paths, test_paths

    def per_budget_equal_result_paths(self):
        """
        Here train/val/test split is performed such that *per fidelity* the ratio of train/val/test is consistent.
        :return: result_paths
        """
        train_paths_dict, val_paths_dict, test_paths_dict = self.per_budget_data()
        def flat_list_from_list_of_lists(list_of_list): return list(itertools.chain.from_iterable(list_of_list))
        train_paths, val_paths, test_paths = [flat_list_from_list_of_lists(dict.values()) for dict in
                                              [train_paths_dict, val_paths_dict, test_paths_dict]]

        rng = np.random.RandomState(6)
        rng.shuffle(train_paths)
        rng.shuffle(val_paths)
        val_paths(test_paths)
        return train_paths, val_paths, test_paths

    def per_budget_data(self):
        """
        Extract the train/val/test split for each budget
        :return: Dictionaries containing the data for each fidelity
        """
        train_paths_dict, val_paths_dict, test_paths_dict = {}, {}, {}
        for fidelity_num in range(7):
            results_in_fidelity = glob.glob(
                os.path.join(self.root, 'run_*/results_fidelity_{}/results_*.json').format(fidelity_num))
            results_in_fidelity.sort()
            # Split the fidelity based on the train/val/test portions
            train_paths_in_fidelity, val_paths_in_fidelity, test_paths_in_fidelity = self.get_splits(
                results_in_fidelity)
            train_paths_dict[fidelity_num] = train_paths_in_fidelity
            val_paths_dict[fidelity_num] = val_paths_in_fidelity
            test_paths_dict[fidelity_num] = test_paths_in_fidelity
        return train_paths_dict, val_paths_dict, test_paths_dict


def find_key_value(key, dictionary):
    """
    Check if key is contained in dictionary in a nested way
    Source: https://gist.github.com/douglasmiranda/5127251#file-gistfile1-py-L2
    :param key:
    :param dictionary:
    :return:
    """
    for k, v in dictionary.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find_key_value(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                for result in find_key_value(key, d):
                    yield result


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
