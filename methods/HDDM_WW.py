from sklearn.base import BaseEstimator, ClassifierMixin
from skmultiflow.drift_detection import HDDM_W

class HDDM_WW(BaseEstimator, ClassifierMixin):
    def __init__(self, drift_confidence=0.001, warning_confidence=0.005, lambda_option=0.05, two_side_option=True):
        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.lambda_option = lambda_option
        self.two_side_option = two_side_option

        self.meta = HDDM_W(drift_confidence,
                           warning_confidence,
                           lambda_option,
                           two_side_option)

        self.drift = []

    def feed(self, X, real, pred):
        n_drifts = 0
        n_warnings = 0

        for y, y_pred in zip(real,pred):
            self.meta.add_element(y==y_pred)
            if self.meta.detected_change():
                n_drifts += 1
            elif self.meta.detected_warning_zone():
                n_warnings += 1

        if n_drifts > 0:
            self.drift.append(2)
        elif n_warnings > 0:
            self.drift.append(1)
        else:
            self.drift.append(0)

        return self
