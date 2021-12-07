"""
This file contains code based on
https://github.com/automl/nasbench301/
Authors: Julien Siems, Lucas Zimmer, Arber Zela, Jovita Lukasik, Margret Keuper, Frank Hutter
"""

import json
import os
import time
import click
import numpy as np
from sklearn.model_selection import StratifiedKFold

from nas_bench_x11.utils import utils
from nas_bench_x11.ensemble import Ensemble


@click.command()
@click.option('--model', type=click.Choice(list(utils.model_dict.keys())), default='svd_lgb',
              help='which surrogate model to fit')

@click.option('--model_config_path', type=click.STRING, help='Leave None to use the default config.', default=None)
@click.option('--log_dir', type=click.STRING, help='Experiment directory', default=None)
@click.option('--seed', type=click.INT, help='seed for numpy, python, pytorch', default=6)
@click.option('--ensemble', help='wether to use an ensemble', default=True)
@click.option('--search_space', help='darts, nb101, nlp, nb201', default='nb101')
@click.option('--data_path', type=click.STRING, default=None, help='path to data directory')
@click.option('--data_splits_root', type=click.STRING, help='path to directory containing data splits', default=None)

def train_surrogate_model(model, model_config_path, 
                          log_dir, seed, ensemble, search_space, data_path, data_splits_root):
    # Load config
    root = utils.get_project_root()
    data_config_path = os.path.join(root, 'configs/data_configs/nb301_splits.json')
    data_config = json.load(open(data_config_path, 'r'))

    # Create log directory
    if log_dir is None:
        log_dir = os.path.join('checkpoints', 'surrogate-' + time.strftime("%Y%m%d-%H%M%S") + '-' + search_space)
    os.makedirs(log_dir)

    # Select model config to use
    if model_config_path is None:
        # Get model configspace
        model_configspace = utils.get_model_configspace(model)

        # Use default model config
        model_config = model_configspace.get_default_configuration().get_dictionary()
    else:
        model_config = json.load(open(model_config_path, 'r'))
    model_config['model'] = model

    # set up default data path
    if data_path is None:
        data_path = os.path.join(root, '../data')
    
    nb101_api = None
    if search_space == 'nb101':
        # load nasbench101 api here, so that it doesn't reload for every surrogate in the ensemble
        from nasbench import api
        nb101_api_folder = os.path.join(utils.get_project_root(), '../data')
        nb101_api = api.NASBench(os.path.join(nb101_api_folder, 'nasbench_full.tfrecord'))
    elif search_space == 'darts':
        # todo: make this more general
        data_path = os.path.join(data_path, 'nb_301_v13_lc_iclr_final')

    # Instantiate surrogate model
    if ensemble:
        surrogate_model = Ensemble(member_model_name=model, data_root=data_path, log_dir=log_dir,
                                   starting_seed=seed,
                                   model_config=model_config, data_config=data_config, ensemble_size=5,
                                   search_space=search_space, nb101_api=nb101_api)
    else:
        surrogate_model = utils.model_dict[model](data_root=data_path, log_dir=log_dir, seed=seed,
                                                  model_config=model_config, data_config=data_config,
                                                  search_space=search_space, nb101_api=nb101_api)

    # Override train/val/test splits if specified
    if data_splits_root is not None:
        train_paths = json.load(open(os.path.join(data_splits_root, "train_paths.json"), "r"))
        val_paths = json.load(open(os.path.join(data_splits_root, "val_paths.json"), "r"))
        test_paths = json.load(open(os.path.join(data_splits_root, "test_paths.json"), "r"))

        cross_val_paths = train_paths + val_paths
        optimzier_identifiers = [path.split("/")[-2] for path in cross_val_paths]
        k_fold = StratifiedKFold(n_splits=10, shuffle=False, random_state=6)
        splits = list(k_fold.split(cross_val_paths, optimzier_identifiers))

        train_inds, val_inds = splits[seed % len(splits)]

        surrogate_model.train_paths = list(np.array(cross_val_paths)[train_inds])
        surrogate_model.val_paths = list(np.array(cross_val_paths)[val_inds])
        surrogate_model.test_paths = test_paths

    # Train and validate the model on the available data
    surrogate_model.train()

    # Test the model
    if len(surrogate_model.test_paths) > 0:
        surrogate_model.test()

    # Save the model
    surrogate_model.save()


if __name__ == "__main__":
    train_surrogate_model()
