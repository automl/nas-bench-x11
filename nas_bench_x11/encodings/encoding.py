import numpy as np

from nas_bench_x11.encodings.encodings_nb101 import encode_nb101
from nas_bench_x11.encodings.encodings_nb201 import encode_nb201
from nas_bench_x11.encodings.encodings_nlp import encode_nlp


def encode(arch_strings, data, search_space, nlp_max_nodes, nb101_api):

    if search_space == 'nb201':
        x_enc = [encode_nb201(arch_str) for arch_str in arch_strings]
        y = [np.array(data[arch_str]['cifar10-valid']['eval_acc1es']) for arch_str in arch_strings]

    elif search_space == 'nlp':
        x_enc = []
        epoch = 3
        for arch_str in arch_strings:
            lc_acc = np.array([100.0 - loss for loss in data[arch_str]['val_losses']])
            accs = lc_acc[:epoch]
            enc = encode_nlp(compact=arch_str, max_nodes=nlp_max_nodes, accs=accs, one_hot=False, lc_feature=True,
                             only_accs=False)
            x_enc.append(enc)
        y = []
        for arch_str in arch_strings:
            lc_acc = np.array([100.0 - loss for loss in data[arch_str]['val_losses']])
            y.append(lc_acc)
            
    elif search_space == 'nb101':
        x_enc = [encode_nb101(nb101_api, arch_str=arch_str, lc_feature=True, only_accs=False) for arch_str in arch_strings]
        y = [np.array(data[arch_str]) for arch_str in arch_strings]

    return x_enc, y, None
