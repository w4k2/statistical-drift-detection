import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity


_SQRT2 = np.sqrt(2)

class ESDDM(BaseEstimator, ClassifierMixin):
    def __init__(self, sigma=3, immobilizer=5, n_detectors = 3, subspace_size='auto', random_state=None):
        self.immobilizer = immobilizer
        self.sigma = sigma
        self.n_detectors = n_detectors
        self.subspace_size = subspace_size
        self.random_state = random_state
        self.random = np.random.RandomState(self.random_state)

        self.niewiem = []
        self.count = 0

    def feed(self, X, y, pred):
        print('i')
        self.count+=1
        # Liczba cech
        self.n_features = X.shape[1]

        # Inicjalizacja kontenerow na dryfy, miary oraz kernele
        if not hasattr(self, "drift"):
            self.drift = []
            self.els_arr = []
            self.base_kernels = []

            [self.els_arr.append([])
             for i in range(3*self.n_detectors)]
            [self.base_kernels.append([])
             for i in range(self.n_detectors)]

            # Oneliner to die
            self._subspace_size = self.subspace_size if isinstance(self.subspace_size, int) else np.ceil(np.sqrt(self.n_features)).astype(int)
            self.subspaces = np.array([self.random.choice(list(range(self.n_features)),
                                                        size=self._subspace_size,
                                                        replace=False)
                                       for _ in range(self.n_detectors)])

            print(self.subspaces)

        el_arr = []

        # Zbior kerneli do podmiany za bazowe
        self.temp_kernels = []
        for sub_id, subspace in enumerate(self.subspaces):
            samples = [
                X[:,subspace],
                X[:,subspace][y==0],
                X[:,subspace][y==1],
                [y]
            ]
            sources = [
                X[:,subspace],
                X[:,subspace],
                X[:,subspace],
                [y]
            ]

            self.kernels = [KernelDensity().fit(sample)
                        for sample in samples]
            self.temp_kernels.append(self.kernels)

            # Bazowe kernele dla każdego podzbioru cech tylko raz
            if len(self.base_kernels[sub_id]) == 0:
                self.base_kernels[sub_id] = self.kernels

            self.cf_s = [[
                np.exp(kernel.score_samples(source))
                for kernel, source in zip(k, sources)]
                    for k in (self.kernels, self.base_kernels[sub_id])]

            # Dzieła zebrane
            el1 = self._tdm(*self.cf_s)
            el2 = self._cmcd(*self.cf_s)
            el3 = self._pd(*self.cf_s)
            el3 = 0
            el_arr.append([el1,el2,el3])

        # Zebranie wszystkiego do ciaglej listy
        el_arr = np.squeeze(np.array(el_arr).reshape(1,-1)).tolist()




        # Integracja
        if len(self.drift)>0:


            last_drf = np.argwhere(np.array(self.drift)==2)
            last_drf = last_drf[-1][0] if len(last_drf)>0 else 0

            if len(self.els_arr[0]) - last_drf > self.immobilizer:
                # Zmienna do życia
                self.els_arr_plot = np.array(self.els_arr).reshape(self.n_detectors, 3, -1)
                #print('a', self.els_arr_plot, self.els_arr_plot.shape)
                # Po zmiennej



                #print('b', np.array(self.els_arr), np.array(self.els_arr).shape)
                #exit()

                #is_drift_arr = np.array([self._is_drift(el_i, els_i[last_drf:])
                #                         for el_i, els_i in zip(el_arr,
                #                                                self.els_arr)])
                is_drift_arr = np.array([self._is_drift(el_i, np.mean(self.els_arr_plot[:,el_idx%3,:], axis=0))
                                         for el_idx, (el_i, els_i) in enumerate(zip(el_arr,
                                                                self.els_arr))])

                print(is_drift_arr)

                #exit()
                dd = np.sum(is_drift_arr)

                # TU JEST WYKRYCIE
                if dd>np.sqrt(len(is_drift_arr)*2/3):
                    self.drift.append(2)
                    self.base_kernels = self.temp_kernels
                    print('JEST')
                    self.niewiem.append(self.count)
                else:
                    self.drift.append(0)
                    print('NIE 1')
            else:
                self.drift.append(0)
                print('NIE 2')
        else:
            self.drift.append(0)
            print('NIE 3')

        for id, el in enumerate(el_arr):
            self.els_arr[id].append(el)

        # Zmienna do plotowania poszczegolnych, usrednionych miar
        self.els_arr_plot = np.array(self.els_arr).reshape(self.n_detectors, 3, -1)

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
