import tensorflow as tf


class Estimator(object):
    """
    An state-action value function estimator. All state inputs must already
    be transformed by a 'featurizer' before being used with Estimator methods.
    """

    def __init__(self):
        self.model = None

    def update(self, s, a, r, s_prime):
        raise NotImplementedError("This method must be implemented by"
                                  "your Estimator class.")

    def predict(self, s):
        raise NotImplementedError("This method must be implemented by"
                                  "your Estimator class")

    def save(self):
        raise NotImplementedError("This method must be implemented by"
                                  "your Estimator class")

    def load(self, name):
        raise NotImplementedError("This method must be implemented by"
                                  "your Estimator class")

    def name(self):
        raise NotImplementedError("This method must be implemented by"
                                  "your Estimator class")



