import os
import json
import numpy as np

from nas_bench_x11.utils import utils

def load_darts_strings(data_root, seed):

    # Load config
    root = utils.get_project_root()
    data_config_path = os.path.join(root, 'configs/data_configs/nb301_splits.json')
    data_config = json.load(open(data_config_path, 'r'))

    # Get the result train/val/test split
    train_paths = []
    val_paths = []
    test_paths = []
    for key, data_config in data_config.items():
        if type(data_config) == dict:
            result_loader = utils.ResultLoader(
                data_root, filepath_regex=data_config['filepath_regex'],
                train_val_test_split=data_config, seed=seed)
            train_val_test_split = result_loader.return_train_val_test()

            train_paths.extend(train_val_test_split[0])
            val_paths.extend(train_val_test_split[1])
            test_paths.extend(train_val_test_split[2])

    # Shuffle the total file paths again
    rng = np.random.RandomState(6)
    rng.shuffle(train_paths)
    rng.shuffle(val_paths)
    rng.shuffle(test_paths)

    return train_paths, val_paths, test_paths


def load_darts_data(result_paths, use_full_lc=False, extra_feats=False):
    """
    Read in the result paths and extract hyperparameters and validation accuracy
    result_paths: list of files containing trained architecture results
    returns list of architecture encodings, val accs, and test accs
    """

    # Create config loader
    root = utils.get_project_root()
    config_loader = utils.ConfigLoader(os.path.join(root, 'configs/data_configs/nb301_configspace.json'))

    # Get the train/test data
    hyps, val_accuracies, test_accuracies, full_lcs = [], [], [], []
    
    for result_path in result_paths:
        config_space_instance, val_accuracy, test_accuracy, _, full_lc = config_loader[result_path]
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
