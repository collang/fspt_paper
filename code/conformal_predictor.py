import tensorflow.keras as keras
import numpy as np


def classification_nconf(prediction, y):
    """
    _______
    params:
        * prediction : numpy array of shape [n_samples, n_classes]
            ** Class probability estimates for each sample.
        * y : numpy array of shape [n_samples]
            **True output labels of each sample.
    """
    mask = np.zeros_like(prediction, dtype=bool)
    mask[np.arange(len(y)), y] = True
    masked_predictions = np.ma.masked_array(prediction, mask=mask)
    ncm = np.max(masked_predictions, axis=1) - prediction[np.arange(len(y)), y]
    return ncm.__array__()


class NN_Conformal_Predictor:

    def __init__(self, model = None, sig_level=0.05):
        self.NN = model
        self.sig = sig_level
        self.n_samples = None
        self.n_features = None
        self.n_classes = None
        self.calibration_nconf = None

    def fit(self, x, y, epochs=10, n_classes=None):
        """
        Trains the neural net model on the provided data.
        _______
        params:
            * x
            * y
            * epochs
            * n_classes
        """
        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]

        if n_classes is None:
            self.n_classes = np.unique(y).size
        else:
            self.n_classes = n_classes

        if self.NN is None:
            self.NN = self.build_default_NN(self.n_classes)
            self.NN.compile(optimizer="Adam", loss="categorical_crossentropy")

        self.NN.fit(x, y, epochs=epochs, verbose=0)

    def calibrate(self, x, y):
        predictions = self.NN.predict(x)
        self.calibration_nconf = self.nconf_measure(predictions, y)

    def predict_intervals(self, x):
        """
        Return the set prediction at the given significance level.
        _______
        params:
            * x = numpy.ndarray : [#samples, #features]
        _______
        returns:
            * out = numpy.ndarray : [#samples, #classes]
                ** Boolean array signifying which classes are in the prediction set.
        """
        if self.calibration_nconf is None:
            raise AttributeError("Missing Calibration Set")

        predictions = self.NN.predict(x)
        nconf_score = self.nconf_measure(predictions)
        p_vals = self.p_values(nconf_score)
        out = p_vals > self.sig
        return out

    def predict(self, x):
        """
        Return the base model prediction.
        _______
        params:
            * x = numpy.ndarray : [#samples, #features]
        _______
        returns:
            * out = numpy.ndarray : [#samples, #classes]
                ** Boolean array signifying which class the base model predicts.
        """

        nn_output = self.NN.predict(x)
        out = np.zeros_like(nn_output, dtype=bool)
        predictions = np.argmax(nn_output, axis=1)
        out[range(predictions.size), predictions] = True
        return out

    def predict_p_values(self, x):
        """
        Returns the p-values of predictions for each sample.
        _______
        params:
            * x = numpy.ndarray : [#_samples, #_features]
                ** data array
        _______
        returns:
            * p_values = numpy.ndarray : [#samples]
                ** p-values for each sample calculated using the calibration set.
        """
        nn_output = self.NN.predict(x)
        predictions = np.argmax(nn_output, axis=1)
        nconf_score = self.nconf_measure(predictions)
        p_vals = self.p_values(nconf_score)
        return p_vals

    def p_values(self, nconf_score):
        """
        Calculates p-value for each prediction.
        _______
        params:
            * nconf_score = numpy.ndarray : [#predictions, #classes]
        _______
        returns:
            * out = numpy.ndarray : [#predictions, #classes]
        """

        if self.calibration_nconf is None:
            raise ValueError("Call self.calibrate() on this object before using this method.")

        n_preds = nconf_score.shape[0]
        out = np.empty_like(nconf_score)

        for i in range(n_preds):

            for cls in range(self.n_classes):
                count_more_nconf = np.sum(self.calibration_nconf >= nconf_score[i, cls]) + 1
                p = count_more_nconf / (self.calibration_nconf.size+1)
                out[i, cls] = p

        return out

    def nconf_measure(self, predictions, y=None):
        """
        Return the nonconformity measure for all predictions and optionally all classes.
        _______
        params:
            * predictions = numpy.ndarray : [#predictions, #classes]
            * y = list or numpy.ndarray : [#predictions]
                ** If y == None this function calculates NCM for all possible labels.
                    This mode is for use on the Test set predictions.
                ** If y != None this function calculates the NCM for the true label class.
        _______
        returns:
            * out = numpy.ndarray : if y == None : [#predictions, #classes] else: [#predictions]
        """

        if y is None:  # For Test Set
            out = np.empty_like(predictions)

            for cls in range(self.n_classes):
                mask = np.zeros_like(predictions, dtype=bool)  # array of False
                mask[:, cls] = True
                masked_predictions = np.ma.masked_array(predictions, mask=mask)  # Mask class of interest
                ncm = np.max(masked_predictions, axis=1) - predictions[:, cls]
                out[:, cls] = ncm

        else:  # For Calibration Set
            n_preds = predictions.shape[0]

            mask = np.zeros_like(predictions, dtype=bool)
            mask[np.arange(n_preds), y] = True
            masked_predictions = np.ma.masked_array(predictions, mask=mask)
            out = np.max(masked_predictions, axis=1) - predictions[np.arange(n_preds), y]

        return out

    def set_NN(self, NN):
        self.NN = NN

    def reset_sig_level(self, sig_level):
        """
        Method for changing the significance level of the Conformal Predictor.
        """
        self.sig = sig_level

    def build_default_NN(self, n_classes):
        return Neural_Net(n_classes)


class FSPTConformalPredictor:
    def __init__(self, tree, calibration_data):
        self.tree = tree
        self.calibration_conf = self.conf_measure(calibration_data)

    def find_rejects(self, x, sig):
        p = self.p_values(x)
        return p < sig

    def p_values(self, x):
        conf_scores = self.conf_measure(x)
        n_preds = conf_scores.shape[0]
        out = np.empty_like(conf_scores, dtype=float)

        for i in range(n_preds):
            count_less_conf = np.sum(self.calibration_conf <= conf_scores[i]) + 1
            out[i] = count_less_conf / (self.calibration_conf.shape[0] + 1)

        return out

    def conf_measure(self, x):
        nodes = self.tree.predict(x)
        return self.tree.get_scores()[nodes]



class Neural_Net(keras.Model):
    def __init__(self, n_classes, n_fc1=32, n_fc2=64, n_fc3=32):
        super(Neural_Net, self).__init__()
        self.fc1 = keras.layers.Dense(n_fc1, activation="tanh")
        self.fc2 = keras.layers.Dense(n_fc2, activation="tanh")
        self.fc3 = keras.layers.Dense(n_fc3, activation="tanh")
        self.out = keras.layers.Dense(n_classes, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        return self.out(self.fc3(self.fc2(self.fc1(inputs))))
