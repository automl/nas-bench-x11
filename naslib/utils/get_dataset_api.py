import os
import pickle

from naslib.utils.utils import get_project_root
from nas_bench_x11.api import load_ensemble

"""
This file loads any dataset files or api's needed by the Trainer or PredictorEvaluator object.
They must be loaded outside of the search space object, because search spaces are copied many times
throughout the discrete NAS algos, which would lead to memory errors.
"""

def get_nasbench101_api(dataset=None, full_data=True,
                        nb111_model_path='checkpoints/nb111-v0.5'):
        
    # load nas-bench-111 surrogate
    nb311_root = get_project_root()
    print('nas-bench-111 path', os.path.join(nb311_root, nb111_model_path))
    performance_model = load_ensemble(os.path.join(nb311_root, nb111_model_path))

    # load nasbench101
    from nasbench import api
    if full_data:
        nb101_data = api.NASBench(os.path.join(get_project_root(), 'data', 'nasbench_full.tfrecord'))
    else:
        nb101_data = api.NASBench(os.path.join(get_project_root(), 'data', 'nasbench_only108.tfrecord'))
    
    return {'api': api, 'nb101_data':nb101_data, 'nb111_model':performance_model}

def get_nasbench201_api(dataset=None):
    """
    Load the original nasbench201 dataset (which does not include full LC info)
    TODO: this is a subset of the full LC datasets, so it is possible to get rid of this dataset.
    """
    with open(os.path.join(get_project_root(), 'data', 'nb201_all.pickle'), 'rb') as f:
        nb201_data = pickle.load(f)

    """
    Now load the full LC info. These files are large, so we only load one for the specific dataset.
    """
    if dataset == 'cifar10':
        with open(os.path.join(get_project_root(), 'data', 'nb201_cifar10_full_training.pickle'), 'rb') as f:
            full_lc_data = pickle.load(f)

    elif dataset == 'cifar100':
        with open(os.path.join(get_project_root(), 'data', 'nb201_cifar100_full_training.pickle'), 'rb') as f:
            full_lc_data = pickle.load(f)

    elif dataset == 'ImageNet16-120':
        with open(os.path.join(get_project_root(), 'data', 'nb201_ImageNet16_full_training.pickle'), 'rb') as f:
            full_lc_data = pickle.load(f)

    return {'raw_data':nb201_data, 'full_lc_data':full_lc_data}

def get_nasbench211_api(dataset=None, 
                        nb211_model_path=os.path.join('checkpoints/nb211-v0.5')):
    # get the datasets from nasbench201
    full_api = get_nasbench201_api(dataset=dataset)

    # load the nb211 surrogate
    nb211_root = get_project_root()
    print('nb211 path', os.path.join(nb211_root, nb211_model_path))

    nb211_model = load_ensemble(os.path.join(nb211_root, nb211_model_path))
    full_api['nb211_model'] = nb211_model
    return full_api

def get_darts_api(dataset=None, learning_curves=True,
                  nb311_model_path='checkpoints/nb311-v0.5',
                  nb301_runtime_path=os.path.expanduser('nasbench301/nb_models/lgb_runtime_v1.0')):
    """
    Load the nb301/nb311 training data (which contains full learning curves) and the nb301 models
    """
    if not learning_curves:
        print('This version currently does not support the original nasbench301')
        raise NotImplementedError()

    nb311_root = get_project_root()
    print('nb311 path', os.path.join(nb311_root, nb311_model_path))

    performance_model = load_ensemble(os.path.join(nb311_root, nb311_model_path))
    runtime_model = load_ensemble(nb301_runtime_path)
    nb311_model = [performance_model, runtime_model]
    return {'nb311_model':nb311_model}


def get_nlp_api(dataset=None, nlp_model_path='checkpoints/nbnlp-v0.5'):
    """
    Load the nas-bench-nlp surrogate model, which contains full training data
    """

    nb311_root = get_project_root()
    print('nas-bench-nlp path', os.path.join(nb311_root, nlp_model_path))

    performance_model = load_ensemble(os.path.join(nb311_root, nlp_model_path))
    return {'nlp_model':performance_model}


def get_dataset_api(search_space=None, dataset=None):

    if search_space == 'nasbench101':
        return get_nasbench101_api(dataset=dataset)

    elif search_space == 'nasbench201':
        return get_nasbench201_api(dataset=dataset)
    
    elif search_space == 'nasbench211':
        return get_nasbench211_api(dataset=dataset)

    elif search_space == 'darts':
        return get_darts_api(dataset=dataset)
    
    elif search_space == 'nlp':
        return get_nlp_api(dataset=dataset)

    else:
        raise NotImplementedError()
