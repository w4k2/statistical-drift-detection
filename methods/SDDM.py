import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity


_SQRT2 = np.sqrt(2)

class SDDM(BaseEstimator, ClassifierMixin):
    def __init__(self, sigma=3, immobilizer=5):
        self.immobilizer = immobilizer
        self.sigma = sigma
        self.drift = []
        self.els_arr = [[],[],[]]
        self.base_kernels = None

    def feed(self, X, y, pred):
        samples = [X, X[y==0], X[y==1], [y]]
        sources = [X, X, X, [y]]

        self.kernels = [KernelDensity().fit(sample)
                    for sample in samples]

        if self.base_kernels is None:
            self.base_kernels = self.kernels

        self.cf_s = [[
            np.exp(kernel.score_samples(source))
            for kernel, source in zip(k, sources)]
                for k in (self.kernels, self.base_kernels)]

        # Dzieła zebrane
        el1 = self._tdm(*self.cf_s)
        el2 = self._cmcd(*self.cf_s)
        el3 = self._pd(*self.cf_s)
        el_arr = [el1,el2,el3]

        # Integracja
        if len(self.drift)>0:
            last_drf = np.argwhere(np.array(self.drift)==2)
            last_drf = last_drf[-1][0] if len(last_drf)>0 else 0

            if len(self.els_arr[0]) - last_drf > self.immobilizer:
                is_drift_arr = np.array([self._is_drift(el_i, els_i[last_drf:]) for el_i, els_i in zip(el_arr, self.els_arr)])
                dd = np.sum(is_drift_arr)

                if dd:
                    self.drift.append(2)
                    self.base_kernels = self.kernels
                else:
                    self.drift.append(0)
            else:
                self.drift.append(0)
        else:
            self.drift.append(0)

        self.els_arr[0].append(el1)
        self.els_arr[1].append(el2)
        self.els_arr[2].append(el3)
        return self

    def _is_drift(self, el, els):
        return np.abs(el - np.mean(els)) > np.std(els) * self.sigma

    def _hellinger(self, p, q):
        return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2

    def _tdm(self, f_s, c_s):
        return self._hellinger(f_s[0], c_s[0])

    def _cmcd(self, f_s, c_s):
        return np.sum([
            ((f_s[3] + c_s[3])/2) * .5 * np.sum(np.abs(f_s[1+i] - c_s[1+i]), axis=0) for i in range(2)
        ])

    def _pd(self, f_s, c_s):
        return np.sum((((c_s[0] + f_s[0])/2) * .5 * (np.sum([
            ((f_s[3] * f_s[1+i]) / f_s[0]) - ((c_s[3] * c_s[1+i]) / c_s[0]) for i in range(2)
        ]))), axis=0)
