import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class NEVER(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.drift = []

    def feed(self, X, y, pred):
        self.drift.append(0)
        return self
