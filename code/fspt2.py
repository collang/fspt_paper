import numpy as np
import math
import warnings
from tree import Tree, Node

warnings.filterwarnings("error")


def get_bounds(X):
    n_features = X.shape[1]
    out = []
    for feature in range(n_features):
        out.append([np.min(X[:, feature]), np.max(X[:, feature])])
    return np.array(out, dtype=float)


def gini_index(p, e):
    n = p+e
    if n == 0:
        return 0.5
    else:
        return (p/n)*(1-p/n) + (e/n)*(1-e/n)


class FspTree(Tree):

    def __init__(self, x, init_size=4, min_gain=0.15, counter_threshold=3, max_depth=None,
                 offset_value=1e-6, min_samples_per_node=1, feature_importance=None):
        """
        _______
        params:
            * x = numpy.ndarray, dtype = any numeric
                ** [#_samples, #_features]
            * init_size = int
                ** The initial size of the node array.
            * min_gain = float
                ** The minimum Gini gain. Gains below this value will add to the counter.
            * counter_threshold = int
                **The number on the counter before a node is no longer splittable.
            * max_depth = int
                ** The maximum depth of the tree.
            * offset_value = float
                ** The size of increment for building the list of split points.
            * min_samples_per_node = int
                ** The minimum number of training examples that must be in both child nodes after splitting.
            * feature_importance = numpy.ndarray, dtype = any numeric
                ** A measure of feature importance.
        """
        super(FspTree, self).__init__(x, init_size, max_depth, min_samples_per_node)

        root = Node(node_id = 0, parent_id=None, bounds=get_bounds(x), n_samples=x.shape[0], depth=1)
        self.nodes[0] = root
        self.min_gain = min_gain
        self.counter_threshold = counter_threshold
        self.offset_value = offset_value

        if feature_importance is not None:
            if feature_importance.size != self.n_features:
                raise ValueError("Feature importance vector invalid size.")
            self.feature_importance = feature_importance
        else:
            self.feature_importance = np.ones(self.n_features)

    def find_rejects(self, x, level):
        scores = self.predict_scores(x)
        return scores < level

    def predict_scores(self, x, normalise=True, e=None):
        nodes = self.predict(x)
        scores = self.get_scores(normalise=False, e=e)[nodes]
        scores[nodes == -1] = 0.0
        scores = (scores - np.min(scores))/(np.max(scores) - np.min(scores))
        return scores

    def build_tree(self):

        stopping_condition = False
        while not stopping_condition:
            found = False

            for node in self.get_nodes():

                if not node.splittable:
                    continue

                if self.max_depth is not None:
                    if node.depth >= self.max_depth:
                        node.splittable=False
                        continue

                if self.min_samples_per_node is not None:
                    if node.n_samples <= self.min_samples_per_node:
                        node.splittable=False
                        continue

                if node.counter > self.counter_threshold:
                    node.splittable = False
                    continue

                score, gain, feature, threshold, p_lesser = self.node_best_split(node.node_id)

                if score == -1:
                    continue

                found = True

                self.split_node(node.node_id, feature, threshold, p_lesser)

                if gain < self.min_gain:
                    self.nodes[node.lesser_child].set_counter(node.counter+1)
                    self.nodes[node.greater_child].set_counter(node.counter+1)

            if not found:
                stopping_condition = True
                
    def node_best_split(self, node_id):
        """
        This method finds the best split for a node.
        _______
        params:
            * node_id = int : identity of node.
        _______
        returns:
            * best_score = float : Gini score of best split
            * best_feature = int : Feature of best split
            * best_threshold = float : Threshold of best split
            * best_p_lesser = float : Number of samples in lesser region (samples of greater region can be inferred).
        """

        samples_idx = self.get_decision_path()[:, node_id]  # find the idx of samples belonging to this node
        samples = self.x[samples_idx, :]  # retrieve samples

        best_score = -np.inf
        best_gain = None
        best_feature = None 
        best_threshold = None
        best_p_lesser = None
        found = False

        for feature in range(self.n_features):
            score, gain, threshold, p_lesser = self.feature_best_split(node_id, feature, samples)

            if score == -1:
                continue  # No split was found on feature that satisfied minimum samples requirements

            score = score * self.feature_importance[feature]

            if score > best_score:
                found = True
                best_score = score
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                best_p_lesser = p_lesser
        
        if found:
            return best_score, best_gain, best_feature, best_threshold, best_p_lesser
        else:
            return -1, -1, -1, -1, -1 # There is no possible split probably due to requirement of min_samples

    def feature_best_split(self, node_id, feature, samples):

        node = self.nodes[node_id]
        bounds = node.bounds[feature, :]  # bounds = [min, max] for feature
        full_bounds = self.nodes[0].bounds[feature, :]
        side_length_fraction = (bounds[1] - bounds[0])/(full_bounds[1] - full_bounds[0])

        unique_points = np.sort(np.unique(samples[:, feature]))  # find the unique split points

        best_score = -np.inf
        best_gain = None
        best_threshold = None
        best_p_lesser = None

        if unique_points.shape[0] == 0:
            return -1, -1, -1, -1

        split_points = np.empty(2 * unique_points.shape[0])
        split_points[1::2] = unique_points
        split_points[0::2] = unique_points - self.offset_value
        split_points[0] = max(split_points[0], bounds[0])

        for i in range(split_points.shape[0]):

            s = split_points[i]

            lesser_side = s - bounds[0]
            try:
                lsf = (lesser_side) / (bounds[1] - bounds[0])  # Lesser Side Fraction
            except RuntimeWarning:
                lsf = 1e-5
                pass

            p_lesser = np.sum(samples[:, feature] <= s)
            e_lesser = node.n_samples * lsf
            p_greater = node.n_samples - p_lesser
            e_greater = node.n_samples - e_lesser

            if p_lesser < self.min_samples_per_node or p_greater < self.min_samples_per_node:
                continue

            lesser_score = gini_index(p_lesser, e_lesser)
            greater_score = gini_index(p_greater, e_greater)
            gini_indx = lsf * lesser_score + (1 - lsf) * greater_score
            gain = (0.5 - gini_indx)

            score = side_length_fraction * gain

            if score > best_score:
                best_score = score
                best_gain = gain
                best_threshold = s
                best_p_lesser = p_lesser

        if best_threshold is None:
            return -1, -1, -1, -1

        else:
            return best_score, best_gain, best_threshold, best_p_lesser

    def get_scores(self, normalise=True, e=None):
        """
        Retrieves the fspt scores for each node.
        _______
        params:
            * only_terminal = bool : option to retrieve only the scores of the terminal nodes.
                ** only really useful for plotting as retrieving the scores of samples requires the indices to match.
                ** TODO: Consider reworking this.
        """
        full_bounds = self.nodes[0].bounds
        full_side_lengths = full_bounds[:, 1] - full_bounds[:, 0]
        out = np.empty(self.node_count, dtype=float)

        if e is None:
            e = self.n_samples/self.n_features

        for i in range(self.node_count):
            node = self.nodes[i]
            bounds = node.bounds
            side_lengths = bounds[:, 1] - bounds[:, 0]
            samples = node.n_samples

            if (samples==0) and (node.get_volume()==0):
                out[i] = 0.0
                continue

            out[i] = np.sum([self.feature_importance[i] * samples /
                             (samples + e*side_lengths[i]/full_side_lengths[i]) for i in range(self.n_features)])

        if normalise:
            out = (out - np.min(out))/(np.max(out) - np.min(out))

        return out

    def set_min_gain(self, min_gain):
        self.min_gain = min_gain

    # def slice_tree(self, depth):
    #     new_tree = self.copy_tree(nodes=False)
    #     nodes_idx = [i for i in range(self.node_count) if self.nodes[i].depth <= depth]
    #     new_tree.set_nodes(self.nodes[nodes_idx])
    #     return new_tree

    # def copy_tree(self, nodes=True):
    #     new_tree = FspTree(self.x, self.node_count, self.min_gain, self.counter_threshold, self.max_depth,
    #                    self.offset_value, self.min_samples_per_node)
    #     if nodes:
    #         new_tree.set_nodes(self.get_nodes())
    #
    #     return new_tree


class ConformalFspt(FspTree):
    def __init__(self, x, init_size=4, min_gain=0.15, counter_threshold=3, max_depth=None,
                 offset_value=0.01, min_samples_per_node=1, feature_importance=None):
        super(ConformalFspt, self).__init__(x, init_size, min_gain, counter_threshold, max_depth, offset_value,
                                            min_samples_per_node, feature_importance)
        self.calibration_conf = None

    def find_rejects(self, x, sig):
        p_vals = self.p_value(x)

        return p_vals < sig

    def p_value(self, x):
        if self.calibration_conf is None:
            raise ValueError("Object is missing a calibration set.")

        n_preds = x.shape[0]
        scores = self.get_scores()[self.predict(x)]
        out = np.empty(n_preds, dtype=float)

        for i in range(n_preds):
            count_less_conf = np.sum(self.calibration_conf < scores[i]) + 1
            p = count_less_conf / (self.calibration_conf.shape[0] + 1)
            out[i] = p

        return out

    def calibrate(self, x):
        self.calibration_conf = self.conf_measure(x)

    def conf_measure(self, x):
        return self.predict_scores(x)


class BaggedFspt:

    def __init__(self, x, init_size=4, min_gain=5e-3, counter_threshold=4, max_depth=None,
                 n_trees=5, offset_value=0.01, verbose=False, min_samples_per_node=1, feature_importance=None):

        self.x = x
        self.n_samples = x.shape[0]
        self.n_classes = x.shape[1]
        self.n_trees = n_trees
        self.verbose = verbose
        self.trees = self.build_trees(init_size, min_gain, counter_threshold, max_depth,
                                      offset_value, min_samples_per_node, feature_importance)
        self.calibration_conf = None

    def find_rejects(self, x, level, voting=False):
        """
        Method for finding the rejects of data at the given level according to the scores of the trees.
        Two options for finding rejects are provided, voting and aggregating.
        _______
        params:
            * x = np.ndarray
            * level = float
                ** Level of rejection.
            * aggregate: bool
                ** If True, rejects are calculated based on the mean score of the trees.
                ** If False, rejects are calculated by voting.
        _______
        returns:
            out = np.ndarray, shape=[#samples], dtype = bool
        """
        if not voting:
            certainty_scores = self.predict_scores(x, aggregate=True)
            return certainty_scores < level

        else:
            certainty_scores = self.predict_scores(x, aggregate=False)
            majority = math.floor(self.n_trees/2)
            return np.sum(certainty_scores < level, axis=1) >= majority

    def find_rejects_conformal(self, x, sig):
        p_vals = self.p_value(x)
        return p_vals < sig

    def p_value(self, x):
        if self.calibration_conf is None:
            raise ValueError("Object is missing a calibration set.")

        n_preds = x.shape[0]
        scores = self.predict_scores(x)
        out = np.empty(n_preds, dtype=float)

        for i in range(n_preds):
            count_less_conf = np.sum(self.calibration_conf < scores[i]) + 1
            p = count_less_conf / (self.calibration_conf.shape[0] + 1)
            out[i] = p

        return out

    def calibrate(self, x):
        self.calibration_conf = self.predict_scores(x)

    def predict_scores(self, x, aggregate=True):
        """
        Finds the average certainty score of the trees for each sample in x.
        _______
        params:
            * x = numpy.ndarray, shape = [#samples, #features]
        _______
        returns:
            * out = numpy.ndarray, dtype = float, shape = [#samples]
        """
        nodes = self.predict(x)
        scores = np.empty((x.shape[0], self.n_trees), dtype=float)
        for i in range(self.n_trees):
            scores[:, i] = self.trees[i].predict_scores(x, normalise=True)
            scores[nodes[:, i] == -1] = 0

        if aggregate:
            out = np.mean(scores, axis=1)
            return out
            # return (out - np.min(out))/(np.max(out) - np.min(out))
        else:
            return scores

    def predict(self, x):
        """
        Finds the node each tree predicts for all samples in x.
        _______
        params:
            * x = numpy.ndarray, [#samples, #features]
        _______
        returns:
            * out = numpy.ndarray, dtype = int, shape = [#samples, self.n_trees]
        """
        out = np.empty((x.shape[0], self.n_trees), dtype=int)

        for i in range(self.n_trees):
            out[:, i] = self.trees[i].predict(x)

        return out

    def build_trees(self, init_size, min_gain, counter_threshold, max_depth, offset_value,
                    min_samples_per_node, feature_importance):
        trees = np.empty(self.n_trees, dtype=object)
        for i in range(self.n_trees):
            sample_idx = np.random.choice(range(self.n_samples), size=self.n_samples, replace=True)
            trees[i] = FspTree(self.x[sample_idx, :], init_size=init_size, min_gain=min_gain,
                               counter_threshold=counter_threshold, max_depth=max_depth, offset_value=offset_value,
                               min_samples_per_node=min_samples_per_node, feature_importance=feature_importance)

            trees[i].build_tree()

            if self.verbose:
                print("built tree {}".format(i+1))

        return trees

    def get_trees(self):
        return self.trees
