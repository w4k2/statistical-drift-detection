import e2_config
import numpy as np
from methods.dderror import dderror
from tabulate import tabulate
from scipy.stats import ttest_rel

drf_types = ['cubic', 'nearest']
detector_names = e2_config.e2_clf_names()

drifts_n = [3,5,7]
features_n = 15

streams = [
    'australian',
    'banknote',
    'diabetes',
    'wisconsin',
]

replications=10


results_all = np.zeros((replications, len(streams), len(drf_types), len(drifts_n), len(detector_names), 3))
                # replications, streams, drf_types, drifts_num, detectors, (acc, d1, d2, cnt)
metric_names = ["d1", "d2", "cnt_ratio"]



for str_id, stream in enumerate(streams):
    #drifts, drf types
    for d_id, d in enumerate(drifts_n):
        for d_type_id, d_type in enumerate(drf_types):

            res_arr = np.load('results_ex2_real/drf_arr_str_%s.csv_%s_%idrfs.npy' %(stream, d_type, d))
            print(res_arr.shape) # reps, detectors, chunks-1

            dderror_arr = np.zeros((res_arr.shape[0], res_arr.shape[1], 3))

            for rep in range(res_arr.shape[0]):
                for det in range(res_arr.shape[1]):
                    real_drf = np.argwhere(res_arr[rep, det, 0]==2).flatten()
                    det_drf = np.argwhere(res_arr[rep, det, 1]==2).flatten()
                    err = dderror(real_drf, det_drf, res_arr.shape[3])
                    dderror_arr[rep, det] = err
            
            #d1
            results_all[:, str_id, d_type_id, d_id, :, 0] = dderror_arr[:,:,0]
            #d2
            results_all[:, str_id, d_type_id, d_id, :, 1] = dderror_arr[:,:,1]
            #cnt
            results_all[:, str_id, d_type_id, d_id, :, 2] = dderror_arr[:,:,2]

results_all = results_all[:,:,:,:,:5,:]
detector_names = detector_names[:5]
print(results_all.shape) # 10, 4, 2, 3, 5, 3

for str_id, stream in enumerate(streams):
    for drf_id, drf_type in enumerate(drf_types):

        #usrednienie po cechach
        all_str = results_all[:,str_id, drf_id]

        print(all_str.shape)  # reps x drfs x detectors x metrics

        row_names = ["%i drifts" % d for d in drifts_n]

        """
        t-test
        """

        alpha = 0.05

        #dla kazdej metryki
        for metric_id in range(3):
            t=[]
            t.append(["", "(1)", "(2)", "(3)", "(4)", "(5)"])
            t.append(['midrule'] + [''])

            # dla kazdeej liczby dryfow
            for row_id, row in enumerate(row_names):

                if row_id == 4:
                    t.append(['midrule'] + [''])

                res_temp = all_str[:,row_id,:,metric_id]
                e_res_temp = np.mean(res_temp, axis=0)
                std_res_temp = np.std(res_temp, axis=0)

                length = 5

                s = np.zeros((length, length))
                p = np.zeros((length, length))

                for i in range(length):
                    for j in range(length):
                        s[i, j], p[i, j] = ttest_rel(res_temp.T[i], res_temp.T[j])

                _ = np.where((p < alpha) * (s < 0))

                conclusions = [list(1 + _[1][_[0] == i])
                        for i in range(length)]

                t.append(["%s" % row] + ["%.3f" % v for v in e_res_temp])
                t.append([''] + ["%.3f" % v for v in std_res_temp])

                t.append([''] + [", ".join(["%i" % i for i in c])
                        if len(c) > 0 and len(c) < length-1 else ("all" if len(c) == length-1 else "---")
                        for c in conclusions])

            # print(tabulate(t, detector_names, floatfmt="%.3f", tablefmt="latex_booktabs")) 
            with open('tables/table_%s_%s_%s.txt' % (metric_names[metric_id], stream, drf_type), 'w') as f:
                f.write(tabulate(t, detector_names, floatfmt="%.3f", tablefmt="latex_booktabs"))
            # exit()

