import os
import pickle

def get_nbnlp_data(data_root, nlp_max_nodes):
    with open(os.path.join(data_root, 'nb_nlp.pickle'), 'rb') as f:
        data = pickle.load(f)
    for arch in list(data.keys()):
        if len(arch[1]) > nlp_max_nodes:
            data.pop(arch)

    return data