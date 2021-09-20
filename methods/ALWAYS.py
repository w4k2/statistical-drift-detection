import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class ALWAYS(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.clf = None
        self.drift = []

    def feed(self, X, y, pred):
        self.drift.append(2)
        return self
