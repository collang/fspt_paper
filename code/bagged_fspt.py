import numpy as np
import math
from fspt2 import FspTree

class BaggedFspt:

    def __init__(self, x, init_size=4, min_gain=5e-3, counter_threshold=4, max_depth=None,
                 n_trees=5, offset_value=0.01, verbose=True):

        self.x = x
        self.n_samples = x.shape[0]
        self.n_classes = x.shape[1]
        self.n_trees = n_trees
        self.verbose = verbose

        self.trees = self.build_trees(init_size, min_gain, counter_threshold, max_depth, offset_value)

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
            certainty_scores = self.certainty_score(x, aggregate=True)
            return certainty_scores < level

        else:
            certainty_scores = self.certainty_score(x, aggregate=False)
            majority = math.floor(self.n_trees/2)
            return np.sum(certainty_scores < level, axis=1) >= majority

    def certainty_score(self, x, aggregate=True):
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
            scores[:, i] = self.trees[i].get_scores(normalise = False)[nodes[:, i]]
            scores[nodes[:, i] == -1] = 0

        if aggregate:
            out = np.mean(scores, axis=1)
            return (out - np.min(out))/(np.max(out) - np.min(out))
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

    def build_trees(self, init_size, min_gain, counter_threshold, max_depth, offset_value):
        trees = np.empty(self.n_trees, dtype=object)
        for i in range(self.n_trees):
            sample_idx = np.random.choice(range(self.n_samples), size=self.n_samples, replace=True)
            trees[i] = FspTree(self.x[sample_idx, :], init_size=init_size, min_gain=min_gain,
                    counter_threshold=counter_threshold, max_depth=max_depth, offset_value=offset_value)
            trees[i].build_tree()

            if self.verbose:
                print("built tree {}".format(i+1))

        return trees

    def get_trees(self):
        return self.trees
