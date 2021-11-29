import numpy as np
import copy
import random
import torch

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench101.conversions import convert_naslib_to_spec, \
    convert_spec_to_tuple, convert_vector_to_spec
from naslib.predictors.utils.encodings_nb101 import encode_adj_naszilla, encode_101

from .primitives import ReLUConvBN

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


class NasBench101SearchSpace(Graph):
    """
    Contains the interface to the tabular benchmark of nasbench 101.
    """

    OPTIMIZER_SCOPE = [
        "stack_1",
        "stack_2",
        "stack_3",
    ]

    QUERYABLE = True

    def __init__(self):
        super().__init__()
        self.num_classes = self.NUM_CLASSES if hasattr(self, 'NUM_CLASSES') else 10
        self.max_epoch = 107
        
        #
        # Cell definition
        #
        node_pair = Graph()
        node_pair.name = "node_pair"    # Use the same name for all cells with shared attributes

        # need to add subgraphs on the nodes, each subgraph has option for 3 ops
        # Input node
        node_pair.add_node(1)
        node_pair.add_node(2)
        node_pair.add_edges_from([(1,2)])

        cell = Graph()
        cell.name = 'cell'

        cell.add_node(1)    # input node
        cell.add_node(2, subgraph=node_pair.set_scope("stack_1").set_input([1]))
        cell.add_node(3, subgraph=node_pair.copy().set_scope("stack_1"))
        cell.add_node(4, subgraph=node_pair.copy().set_scope("stack_1"))
        cell.add_node(5, subgraph=node_pair.copy().set_scope("stack_1"))
        cell.add_node(6, subgraph=node_pair.copy().set_scope("stack_1"))
        cell.add_node(7)    # output

        # Edges
        cell.add_edges_densly()

        #
        # dummy Makrograph definition for RE for benchmark queries
        #
        channels = [128, 256, 512]
        
        self.name = "makrograph"

        total_num_nodes = 3
        self.add_nodes_from(range(1, total_num_nodes+1))
        self.add_edges_from([(i, i+1) for i in range(1, total_num_nodes)])

        self.edges[1, 2].set('op', ops.Stem(channels[0]))
        self.edges[2, 3].set('op', cell.copy().set_scope('stage_1'))
        
        node_pair.update_edges(
            update_func=lambda current_edge_data: _set_node_ops(current_edge_data, C=channels[0]),
            scope="node",
            private_edge_data=True
        )
        
        cell.update_edges(
            update_func=lambda current_edge_data: _set_cell_ops(current_edge_data, C=channels[0]),
            scope="cell",
            private_edge_data=True
        )

    def query(self, metric=None, dataset='cifar10', path=None, epoch=-1, full_lc=False, dataset_api=None):

        assert isinstance(metric, Metric)
        assert dataset in ['cifar10', None], "Unknown dataset: {}".format(dataset)
        if metric in [Metric.ALL, Metric.HP]:
            raise NotImplementedError()
        if dataset_api is None:
            raise NotImplementedError('Must pass in dataset_api')

        # note: these are only used for querying the nasbench101 api
        metric_to_nb101 = {
            Metric.TRAIN_ACCURACY: 'train_accuracy',
            Metric.VAL_ACCURACY: 'validation_accuracy',
            Metric.TEST_ACCURACY: 'test_accuracy',
            Metric.TRAIN_TIME: 'training_time',
            Metric.PARAMETERS: 'trainable_parameters',
        }

        spec = self.get_spec()


        api_spec = dataset_api['api'].ModelSpec(**spec)
        if not dataset_api['nb101_data'].is_valid(api_spec):
            raise NotImplementedError('Invalid spec')

        # create the encoding used in nas-bench-111
        encoding = encode_adj_naszilla(spec)
        fix, comp = dataset_api['nb101_data'].get_metrics_from_spec(api_spec)
        accs = []
        for e in [4, 12, 36, 108]:
            accs.append(np.mean([comp[e][i]['final_validation_accuracy'] for i in range(3)]))
        encoding = np.array([*encoding, *accs])

        if metric == Metric.TRAIN_TIME:
            # todo: create an nb111 train time model. Alternatively, use nb101 train times
            if epoch == -1:
                return np.mean([comp[108][i]['final_training_time'] for i in range(3)])
            else:
                return np.mean([comp[108][i]['final_training_time'] for i in range(3)]) * epoch/self.max_epoch

        lc = dataset_api['nb111_model'].predict(config=encoding,
                                                representation='compact',
                                                search_space='nb101')
        if full_lc and epoch == -1:
            return lc
        elif full_lc and epoch != -1:
            return lc[:epoch]
        else:
            # return the value of the metric only at the specified epoch
            return lc[epoch]


    def get_spec(self):
        if self.spec is None:
            self.spec = convert_naslib_to_spec(self)
        return self.spec
    
    def get_hash(self):
        return convert_spec_to_tuple(self.get_spec())

    def get_max_epochs(self):
        # Return the max number of epochs that can be queried
        return 107

    def set_spec(self, spec):
        # TODO: convert the naslib object to this spec
        # convert_spec_to_naslib(spec, self)
        self.spec = spec

    def sample_random_architecture(self, dataset_api):
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        From the NASBench repository:
        one-hot adjacency matrix
        draw [0,1] for each slot in the adjacency matrix
        """
        while True:
            matrix = np.random.choice(
                [0, 1], size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT
            spec = dataset_api['api'].ModelSpec(matrix=matrix, ops=ops)
            if dataset_api['nb101_data'].is_valid(spec):
                break
                
        self.set_spec({'matrix':matrix, 'ops':ops})

    def model_based_sample_architecture(self, dataset_api=None, minimize_me=None, good_kde=None, vartypes=None):
        """
        This will perform a model-based architecture sampling and update the edges in the
        naslib object accordingly.
        """
        num_samples = 128
        random_fraction = 0.2
        while True:
            best = np.inf
            best_vector = None
            for i in range(num_samples):
                idx = np.random.randint(0, len(good_kde.data))
                datum = good_kde.data[idx]
                vector = []
                for m, bw, t in zip(datum, good_kde.bw,
                                    vartypes):
                    if np.random.rand() < (1 - bw):
                        vector.append(int(m))
                    else:
                        vector.append(np.random.randint(t))
                val = minimize_me(vector)
                if val < best:
                    best = val
                    best_vector = vector
            if best_vector is None or np.random.rand() < random_fraction:
                self.sample_random_architecture(dataset_api=dataset_api)
            else:
                best_matrix, best_ops = convert_vector_to_spec(best_vector)
                spec = dataset_api['api'].ModelSpec(matrix=best_matrix, ops=best_ops)
                if dataset_api['nb101_data'].is_valid(spec):
                    best_matrix, best_ops = convert_vector_to_spec(best_vector)
                    self.set_spec({'matrix': best_matrix, 'ops': best_ops})
                    break

    def mutate(self, parent, dataset_api, edits=1):
        """
        This will mutate the parent architecture spec.
        Code inspird by https://github.com/google-research/nasbench
        """
        parent_spec = parent.get_spec()
        spec = copy.deepcopy(parent_spec)
        matrix, ops = spec['matrix'], spec['ops']
        for _ in range(edits):
            while True:
                new_matrix = copy.deepcopy(matrix)
                new_ops = copy.deepcopy(ops)
                for src in range(0, NUM_VERTICES - 1):
                    for dst in range(src+1, NUM_VERTICES):
                        if np.random.random() < 1 / NUM_VERTICES:
                            new_matrix[src][dst] = 1 - new_matrix[src][dst]
                for ind in range(1, NUM_VERTICES - 1):
                    if np.random.random() < 1 / len(OPS):
                        available = [op for op in OPS if op != new_ops[ind]]
                        new_ops[ind] = np.random.choice(available)
                new_spec = dataset_api['api'].ModelSpec(new_matrix, new_ops)
                if dataset_api['nb101_data'].is_valid(new_spec):
                    break
        
        self.set_spec({'matrix':new_matrix, 'ops':new_ops})

    def crossover_bin(self, parent, mutant, dim, prob, dataset_api=None):
        '''Performs the binomial crossover of DE
        '''
        parent_enc = np.array(encode_101(parent, encoding_type='adjacency_cat'))
        mutant_enc = np.array(encode_101(mutant, encoding_type='adjacency_cat'))
        while True:
            cross_points = np.random.rand(dim) < prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True

            offspring = np.where(cross_points, mutant_enc, parent_enc)
            matrix, ops = convert_vector_to_spec(offspring)
            spec = dataset_api['api'].ModelSpec(matrix, ops)
            if dataset_api['nb101_data'].is_valid(spec):
                self.set_spec({'matrix':matrix, 'ops':ops})
                break


    def get_nbhd(self, dataset_api=None):
        # return all neighbors of the architecture
        spec = self.get_spec()
        matrix, ops = spec['matrix'], spec['ops']
        nbhd = []
        
        def add_to_nbhd(new_matrix, new_ops, nbhd):
            new_spec = {'matrix':new_matrix, 'ops':new_ops}
            model_spec = dataset_api['api'].ModelSpec(new_matrix, new_ops)
            if dataset_api['nb101_data'].is_valid(model_spec):
                nbr = NasBench101SearchSpace()
                nbr.set_spec(new_spec)
                nbr_model = torch.nn.Module()
                nbr_model.arch = nbr
                nbhd.append(nbr_model)
            return nbhd
        
        # add op neighbors
        for vertex in range(1, OP_SPOTS + 1):
            if is_valid_vertex(matrix, vertex):
                available = [op for op in OPS if op != ops[vertex]]
                for op in available:
                    new_matrix = copy.deepcopy(matrix)
                    new_ops = copy.deepcopy(ops)
                    new_ops[vertex] = op
                    nbhd = add_to_nbhd(new_matrix, new_ops, nbhd)

        # add edge neighbors
        for src in range(0, NUM_VERTICES - 1):
            for dst in range(src+1, NUM_VERTICES):
                new_matrix = copy.deepcopy(matrix)
                new_ops = copy.deepcopy(ops)
                new_matrix[src][dst] = 1 - new_matrix[src][dst]
                new_spec = {'matrix':new_matrix, 'ops':new_ops}
            
                if matrix[src][dst] and is_valid_edge(matrix, (src, dst)):
                    nbhd = add_to_nbhd(new_matrix, new_ops, nbhd)

                if not matrix[src][dst] and is_valid_edge(new_matrix, (src, dst)):
                    nbhd = add_to_nbhd(new_matrix, new_ops, nbhd)

        random.shuffle(nbhd)
        return nbhd

    def get_type(self):
        return 'nasbench101'
    
def _set_node_ops(current_edge_data, C):
    current_edge_data.set('op', [
        ReLUConvBN(C, C, kernel_size=1),
        # ops.Zero(stride=1),    #! recheck about the hardcoded second operation
        ReLUConvBN(C, C, kernel_size=3),
        ops.MaxPool1x1(kernel_size=3, stride=1),
    ])

def _set_cell_ops(current_edge_data, C):
    current_edge_data.set('op', [
        ops.Identity(),
        ops.Zero(stride=1), 
    ])
    

def get_utilized(matrix):
    # return the sets of utilized edges and nodes
    # first, compute all paths
    n = np.shape(matrix)[0]
    sub_paths = []
    for j in range(0, n):
        sub_paths.append([[(0, j)]]) if matrix[0][j] else sub_paths.append([])
    
    # create paths sequentially
    for i in range(1, n - 1):
        for j in range(1, n):
            if matrix[i][j]:
                for sub_path in sub_paths[i]:
                    sub_paths[j].append([*sub_path, (i, j)])
    paths = sub_paths[-1]

    utilized_edges = []
    for path in paths:
        for edge in path:
            if edge not in utilized_edges:
                utilized_edges.append(edge)

    utilized_nodes = []
    for i in range(NUM_VERTICES):
        for edge in utilized_edges:
            if i in edge and i not in utilized_nodes:
                utilized_nodes.append(i)

    return utilized_edges, utilized_nodes

def num_edges_and_vertices(matrix):
    # return the true number of edges and vertices
    edges, nodes = get_utilized(matrix)
    return len(edges), len(nodes)

def is_valid_vertex(matrix, vertex):
    edges, nodes = get_utilized(matrix)
    return (vertex in nodes)

def is_valid_edge(matrix, edge):
    edges, nodes = get_utilized(matrix)
    return (edge in edges)