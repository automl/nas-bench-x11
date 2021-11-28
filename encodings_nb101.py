import numpy as np
from nasbench import api


INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


def convert_to_cell(matrix, ops):

    if len(matrix) < 7:
        # the nasbench spec can have an adjacency matrix of n x n for n<7,
        # but in the nasbench api, it is always 7x7 (possibly containing blank rows)
        # so this method will add a blank row/column

        new_matrix = np.zeros((7, 7), dtype='int8')
        new_ops = []
        n = matrix.shape[0]
        for i in range(7):
            for j in range(7):
                if j < n - 1 and i < n:
                    new_matrix[i][j] = matrix[i][j]
                elif j == n - 1 and i < n:
                    new_matrix[i][-1] = matrix[i][j]

        for i in range(7):
            if i < n - 1:
                new_ops.append(ops[i])
            elif i < 6:
                new_ops.append('conv3x3-bn-relu')
            else:
                new_ops.append('output')
        return {
            'matrix': new_matrix,
            'ops': new_ops
        }

    else:
        return {
            'matrix': matrix,
            'ops': ops
        }


def encode_adj(matrix, ops):
    """ 
    compute the "standard" encoding,
    i.e. adjacency matrix + op list encoding 
    """
    encoding_length = (NUM_VERTICES ** 2 - NUM_VERTICES) // 2 + OP_SPOTS
    encoding = np.zeros((encoding_length))
    dic = {CONV1X1: 0., CONV3X3: 0.5, MAXPOOL3X3: 1.0}
    n = 0
    for i in range(NUM_VERTICES - 1):
        for j in range(i+1, NUM_VERTICES):
            encoding[n] = matrix[i][j]
            n += 1
    for i in range(1, NUM_VERTICES - 1):
        encoding[n + i - 1] = dic[ops[i]]
    return tuple(encoding)


def get_paths(matrix, ops):
    """ 
    return all paths from input to output
    """
    paths = []
    for j in range(0, NUM_VERTICES):
        paths.append([[]]) if matrix[0][j] else paths.append([])

    # create paths sequentially
    for i in range(1, NUM_VERTICES - 1):
        for j in range(1, NUM_VERTICES):
            if matrix[i][j]:
                for path in paths[i]:
                    paths[j].append([*path, ops[i]])
    return paths[-1]


def get_path_indices(matrix, ops):
    """
    compute the index of each path
    There are 3^0 + ... + 3^5 paths total.
    (Paths can be length 0 to 5, and for each path, for each node, there
    are three choices for the operation.)
    """
    paths = get_paths(matrix, ops)
    mapping = {CONV3X3: 0, CONV1X1: 1, MAXPOOL3X3: 2}
    path_indices = []

    for path in paths:
        index = 0
        for i in range(NUM_VERTICES - 1):
            if i == len(path):
                path_indices.append(index)
                break
            else:
                index += len(OPS) ** i * (mapping[path[i]] + 1)

    path_indices.sort()
    return tuple(path_indices)


def encode_paths(matrix, ops):
    """ output one-hot encoding of paths """
    path_indices = get_path_indices(matrix, ops)
    num_paths = sum([len(OPS) ** i for i in range(OP_SPOTS + 1)])
    encoding = np.zeros(num_paths)
    for index in path_indices:
        encoding[index] = 1
    return encoding


def encode_nb101(nb101_api, arch_str, encoding_type='adj', lc_feature=False, only_accs=False):

    fix, comp = nb101_api.get_metrics_from_hash(arch_str)
    cell = convert_to_cell(fix['module_adjacency'], fix['module_operations'])

    if encoding_type == 'adj':
        encoding = encode_adj(cell['matrix'], cell['ops'])
    elif encoding_type == 'path':
        encoding = encode_paths(cell['matrix'], cell['ops'])

    if lc_feature:
        accs = []
        for e in [4, 12, 36, 108]:
            accs.append(np.mean([comp[e][i]['final_validation_accuracy'] for i in range(3)]))
        encoding = np.array([*encoding, *accs])
    if only_accs:
        encoding = np.array([*accs])
    return encoding
