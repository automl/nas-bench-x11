import numpy as np

def get_adj_matrix(compact, max_nodes):
    # this method returns the flattened adjacency matrix only
    last_idx = len(compact[1]) - 1
    assert last_idx <= max_nodes
    def extend(idx):
        if idx == last_idx:
            return max_nodes
        return idx 

    adj_matrix = np.zeros((max_nodes+1, max_nodes+1))
    for edge in compact[0]:
        adj_matrix[extend(edge[0]), extend(edge[1])] = 1

    return adj_matrix

def get_categorical_ops(compact, max_nodes):
    """
    This returns the set of ops, extended to account for the
    max number of nodes in the search space, so that it's the
    same size for all ops.
    """
    last_idx = len(compact[1]) - 1
    assert last_idx <= max_nodes
    return [*compact[1][:-1], *[0]*(max_nodes - last_idx), compact[1][-1]]

def get_categorical_hidden_states(compact, max_hidden_states=3):
    assert len(compact[2]) <= max_hidden_states
    return [*compact[2], *[0]*(max_hidden_states - len(compact[2]))]

def encode_nlp(compact, max_nodes, accs, one_hot=False, lc_feature=True, only_accs=False):
    """
    this method returns the adjacency one hot encoding,
    which is a flattened adjacency matrix + one hot op encoding
    + flag for is_hidden_state on each node.
    """
    adj_matrix = get_adj_matrix(compact, max_nodes=max_nodes)
    flattened = [int(i) for i in adj_matrix.flatten()]
    assert len(flattened) == (max_nodes + 1) ** 2

    # add ops and hidden states
    ops = get_categorical_ops(compact, max_nodes=max_nodes)
    assert len(ops) == max_nodes + 1
    hidden_states = get_categorical_hidden_states(compact)
    assert len(hidden_states) == 3
    if not one_hot:
        if only_accs:
            return [*accs]
        if lc_feature:
            return [*flattened, *ops, *hidden_states, *accs]
        return [*flattened, *ops, *hidden_states]

    ops_onehot = []
    last_idx = len(compact[1]) - 1
    for i, op in enumerate(ops):
        onehot = [1 if op == i else 0 for i in range(8)]
        ops_onehot.extend(onehot)
        if i in compact[2]:
            ops_onehot.append(1)
        elif i == max_nodes and last_idx in compact[2]:
            ops_onehot.append(1)
        else:
            ops_onehot.append(0)

    if only_accs:
        return [*accs]
    if lc_feature:
        return [*flattened, *ops, *hidden_states, *accs]
    return [*flattened, *ops_onehot]