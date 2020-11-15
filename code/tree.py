import numpy as np
from copy import deepcopy
import warnings

warnings.filterwarnings("error")


def get_bounds(X):
    n_features = X.shape[1]
    out = []
    for feature in range(n_features):
        out.append([np.min(X[:, feature]), np.max(X[:, feature])])
    return np.array(out, dtype=float)


class Tree:

    def __init__(self, x, init_size=10, max_depth=None, min_samples_per_node=1):
        """
        _______
        params:
            * x = numpy.ndarray : [#_samples, #_features]
            * init_size : The initial size of the node array.
            * min_samples_per_node : The minimum samples that each node can have.
            * min_gain : The minimum Gini gain. Gains below this value will contribute to the counter for early stopping.
            * counter_threshold : The number on the counter before early stopping is triggered.
            * offset_value : The size of the increment for building the list of split points.
        """
        self.x = x
        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]
        self.nodes = np.empty([init_size], dtype=object)
        self.node_count = 1
        self.capacity = init_size
        self.max_depth = max_depth
        self.min_samples_per_node = min_samples_per_node
        self.decision_path = np.zeros([self.n_samples, self.capacity], dtype=bool)
        self.decision_path[:, 0] = np.ones([self.n_samples])

    def predict(self, x):
        """
        Returns the node for each sample in x.
        _______
        params:
            * x = numpy.ndarray : [#_samples, #_features]
        _______
        returns:
            * out = numpy.ndarray : [#_samples]
        _______
        """
        out = np.empty(x.shape[0], dtype=int)
        for i in range(x.shape[0]):
            node = self.nodes[0]  # Start from root node

            if any([(x[i, j] < node.bounds[j, 0] or x[i, j] > node.bounds[j, 1]) for j in range(x.shape[1])]):
                # Sample is outside the root node
                out[i] = -1
                continue

            while not node.is_terminal:
                if x[i, node.feature] <= node.threshold:
                    node = self.nodes[node.lesser_child]
                else:
                    node = self.nodes[node.greater_child]
            out[i] = node.node_id
        return out

    def split_node(self, node_id, feature, threshold, n_samples_lesser):
        """
        This method will split a node by constructing two new nodes and assigning them to the child attributes of the
        parent.
        _______
        params:
            * node_id = int : Node identifier.
            * feature = int : Feature to split on.
            * threshold = float : Threshold to split at.
            * n_samples_lesser = int : Number of samples in lesser child (number in greater child inferred).
        """
        node = self.nodes[node_id]
        node.feature = feature
        node.threshold = threshold
        node.is_terminal = False
        node.splittable=False

        parent_bounds = node.get_bounds()

        lesser_bounds = deepcopy(parent_bounds)
        lesser_bounds[feature, 1] = threshold
        greater_bounds = deepcopy(parent_bounds)
        greater_bounds[feature, 0] = threshold

        self.add_node(node_id, lesser_bounds, n_samples_lesser, node.depth + 1, is_lesser=True)
        self.add_node(node_id, greater_bounds, (node.n_samples - n_samples_lesser), node.depth + 1, is_lesser=False)
        self.update_decision_path()

    def add_node(self, parent_id, bounds, n_samples, depth, is_lesser):
        """
        Method for adding a node to the self.nodes attribute of the tree. Will automatically expand the capacity of the
        tree.
        _______
        params:
            * parent_id = int : identifier of parent (not parent Node object).
            * bounds = np.ndarray : [#features, [lower bound, upper bound]].
            * n_samples = int : number of samples in node.
            * is_lesser = bool : if True then node is the lesser child of it's parent.
        """

        node_id = self.node_count
        if node_id >= self.capacity:
            new_capacity = self.capacity * 2

            new_nodes = np.empty([new_capacity], dtype=object)
            new_nodes[:self.node_count] = self.nodes

            self.capacity = new_capacity
            self.nodes = new_nodes

        node = Node(node_id, parent_id, bounds, n_samples, depth)
        self.nodes[node_id] = node
        self.node_count += 1

        if parent_id is None:
            pass  # node is root node
        elif is_lesser:
            self.nodes[parent_id].lesser_child = node_id
        else:
            self.nodes[parent_id].greater_child = node_id

    def update_decision_path(self):
        """
        Method for updating the decision path attribute of the tree.
        _______
        notes:
            * self.decision_path = numpy.ndarray : [#nodes, #samples]
                ** A boolean array where 'decision_path[i, j] = True' means that node 'i' contains sample 'j'.
        """
        new_decision_path = np.zeros([self.n_samples, self.node_count], dtype=bool)
        for i in range(self.n_samples):
            node = self.nodes[0]
            new_decision_path[i, 0] = True
            while not node.is_terminal:
                if self.x[i, node.feature] <= node.threshold:
                    node = self.nodes[node.lesser_child]
                else:
                    node = self.nodes[node.greater_child]
                new_decision_path[i, node.node_id] = True

        self.decision_path = new_decision_path

    def get_decision_path(self):
        """
        Retrieves decision path. Use this method rather than accessing class attribute as it strips off the trailing
        empty elements of the decision path.
        _______
        returns:
            * decision_path = numpy.ndarray : [#_nodes, #_samples]
                ** A boolean array where 'decision_path[i, j] = True' means that node 'i' contains sample 'j'.
        """
        return self.decision_path[:, :self.node_count]

    def get_nodes(self):
        """
        Retrieves all nodes of the tree. Use this method rather than accessing class attribute as it strips off the
        trailing empty elements of node.
        """
        return self.nodes[:self.node_count]

    def get_max_depth(self):
        max_depth = 0
        for i in range(self.node_count):
            d = self.nodes[i].depth
            if d > max_depth:
                max_depth = d
        return max_depth

    def set_max_depth(self, max_depth):
        self.max_depth = max_depth

    def set_nodes(self, nodes, node_count):
        self.nodes = nodes
        self.node_count = node_count


class Node:
    def __init__(self, node_id, parent_id, bounds, n_samples, depth, lesser_child=None,
                 greater_child=None, feature=None, threshold=None, is_terminal=True, splittable=True, counter=0):

        self.node_id = node_id
        self.parent_id = parent_id
        self.lesser_child = lesser_child
        self.greater_child = greater_child
        self.bounds = bounds
        self.feature = feature
        self.threshold = threshold
        self.n_samples = n_samples
        self.depth = depth
        self.is_terminal = is_terminal
        self.splittable = splittable
        self.counter = counter

    def get_bounds(self):
        return self.bounds

    def get_volume(self):
        return np.prod(self.bounds[:, 1] - self.bounds[:, 0])

    def set_counter(self, counter):
        self.counter = counter