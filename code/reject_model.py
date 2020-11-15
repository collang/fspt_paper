import numpy as np


class FsptRejectModel:
    def __init__(self, tree, level):
        self.tree = tree
        self.level = level

    def find_rejects(self, x):
        """
        Predict with rejection.
        _______
        params:
            * x = np.ndarray : [#samples, #features]
        _______
        returns:
            * reject = np.ndarray : [#samples]
                ** boolean array: True if sample is rejected, False otherwise.
        """
        nodes = self.tree.predict(x)
        scores = self.tree.get_scores()[nodes]
        out_range = (nodes == -1)
        reject = np.zeros(x.shape[0], dtype=bool)

        for i in range(x.shape[0]):
            if (not out_range[i]) and (scores[i] >= self.level):  # if in range and score is above level:
                pass
            else:
                reject[i] = True

        return reject


# TODO: This aint finished
class BFsptRejectModel:
    def __init__(self, tree, level):
        self.tree = tree
        self.level = level

    def find_rejects(self, x):
        score = self.tree.certainty_score(x)
        reject = np.zeros(x.shape[0], dtype=bool)

        for i in range(x.shape[0]):
            pass


class CpRejectModel:
    def __init__(self):
        pass

    def find_rejects(self, prediction):

        reject = np.zeros(prediction.shape[0], dtype=bool)
        prediction_set_size = np.sum(prediction, axis=1)

        for i in range(prediction.shape[0]):

            if prediction_set_size[i] != 1:
                reject[i] = True

        return reject



class CombinedRejectModel:
    def __init__(self, tree, model, level):
        self.tree = tree
        self.model = model
        self.level = level

        self.fspt_reject_model = FsptRejectModel(tree, level)
        self.cp_reject_model = CpRejectModel()

    def find_rejects(self, x, predictions=None):
        """
        Method for finding the rejects using both FsptRejectModel and CpRejectModel.
        _______
        params:
            * x = numpy.ndarray : [#samples, #features]
        _______
        returns:
            * reject = numpy.ndarray : [#samples, [fspt_reject, cp_reject]]
        """
        if predictions is None:
            predictions = self.model.predict_intervals(x)

        reject = np.zeros([x.shape[0], 2], dtype=bool)

        reject[:, 0] = self.fspt_reject_model.find_rejects(x)
        reject[:, 1] = self.cp_reject_model.find_rejects(predictions)

        return reject