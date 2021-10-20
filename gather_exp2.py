import e2_config
import numpy as np
from methods.dderror import dderror
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


drf_types = e2_config.e2_drift_types()
recurring = e2_config.e2_recurring()

detector_names = e2_config.e2_clf_names()
replications = e2_config.e2_replications()

drifts_n = e2_config.e2_n_drifts()
features_n = e2_config.e2_n_features()


results_all = np.zeros((replications, len(recurring), len(drf_types), len(features_n), len(drifts_n), len(detector_names), 4))
                # replications, recurring, drf_types, features, drifts_num, detectors, (acc, d1, d2, cnt)

# str_names=[]
metric_names = ["Accuracy", "d1", "d2", "cnt_ratio"]

for rec_id, rec in enumerate(recurring):
    for drf_id, drf_type in enumerate(drf_types):

        #features, drifts
        for f_id, f in enumerate(features_n):
            for d_id, d in enumerate(drifts_n):

                res_clf = np.load('results_ex2_d_f_45/clf_%ifeat_%idrifts_%s_%s.npy' %(f, d, drf_type, rec))
                # replications x detectors x chunks-1

                res_arr = np.load('results_ex2_d_f_45/drf_arr_%ifeat_%idrifts_%s_%s.npy' %(f, d, drf_type, rec))
                # replications x detectors x (real, detected) x chunks-1

                dderror_arr = np.zeros((res_arr.shape[0], res_arr.shape[1], 3))

                for rep in range(res_arr.shape[0]):
                    for det in range(res_arr.shape[1]):
                        real_drf = np.argwhere(res_arr[rep, det, 0]==2).flatten()
                        det_drf = np.argwhere(res_arr[rep, det, 1]==2).flatten()
                        err = dderror(real_drf, det_drf, res_arr.shape[3])
                        dderror_arr[rep, det] = err
                
                #accuracy
                results_all[:, rec_id, drf_id, f_id, d_id, :, 0] = np.mean(res_clf, axis=2)
                #d1
                results_all[:, rec_id, drf_id, f_id, d_id, :, 1] = dderror_arr[:,:,0]
                #d2
                results_all[:, rec_id, drf_id, f_id, d_id, :, 2] = dderror_arr[:,:,1]
                #cnt
                results_all[:, rec_id, drf_id, f_id, d_id, :, 3] = dderror_arr[:,:,2]

                # str_names.append("%s_%s_drfits_%i_features_%i" % (rec, drf_type, d, f))

# print(len(str_names))
print(results_all.shape) # 10, 2, 3, 4, 4, 6, 4

# mean_res_all = np.mean(results_all, axis=0) # 2, 3, 4, 4, 6, 4
# std_res_all = np.std(results_all, axis=0)

# print(mean_res_all.shape)
# exit()

# mean_acc = np.concatenate((np.array(str_names)[:, np.newaxis], mean_res_all[:,:,0]), axis=1)
# mean_d1 = np.concatenate((np.array(str_names)[:, np.newaxis], mean_res_all[:,:,1]), axis=1)
# mean_d2 = np.concatenate((np.array(str_names)[:, np.newaxis], mean_res_all[:,:,2]), axis=1)
# mean_cnt = np.concatenate((np.array(str_names)[:, np.newaxis], mean_res_all[:,:,3]), axis=1)

# print("\n Mean accuracy")
# print(tabulate(mean_acc, headers=detector_names, floatfmt=".3f"))

# print("\n Mean d1")
# print(tabulate((mean_d1), headers=detector_names, floatfmt=".3f"))

# print("\n Mean d2")
# print(tabulate(mean_d2, headers=detector_names, floatfmt=".3f"))

# print("\n Mean cnt_ratio")
# print(tabulate(mean_cnt, headers=detector_names, floatfmt=".3f"))

# 10, 2, 3, 4, 4, 6, 4 
# reps x rec x drf type x features x drf num x detectors x metrics

# str_names = []

for drf_id, drf_type in enumerate(drf_types): #3
    for rec_id, rec in enumerate(recurring): #2
        # str_names.append("%s, %s" % (drf_type, rec))

        #usrednienie po cechach
        all_features = results_all[:,rec_id, drf_id] # reps x features x drf num x detectors x metrics

        mean_features = np.mean(all_features, axis = 1)
        std_features = np.std(all_features, axis = 1)

        # print(mean_features.shape) # 10 x 4, 6, 4 -> reps, drf_num, detectors x metric

        #usrednienie po dryfach
        all_drifts = results_all[:,rec_id, drf_id]

        mean_drifts = np.mean(all_drifts, axis = 2)
        std_drifts = np.std(all_drifts, axis = 2)
    
        # print(mean_drifts.shape) # 10 x 4, 6, 4 -> reps, features, detectors x metric

        #razem 
        res = np.concatenate((mean_features, mean_drifts), axis=1)
        res_std = np.concatenate((std_features, std_drifts), axis=1)

        # print(res.shape)
        # exit()

        row_names = ["%i features" % f for f in features_n]
        row_names_d = ["%i drifts" % d for d in drifts_n]

        for d in row_names_d:
            row_names.append(d)

        # print(row_names)
        # exit()

        """
        t-test
        """

        alpha = 0.05

        #dla kazdej metryki
        for metric_id in range(4):
            t=[]
            t.append(["", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)"])
            t.append(['midrule'] + [''])

            # dla kazdeej liczby dryfow i cech
            for row_id, row in enumerate(row_names):

                if row_id == 4:
                    t.append(['midrule'] + [''])

                res_temp = res[:,row_id,:,metric_id]
                e_res_temp = np.mean(res_temp, axis=0)
                std_res_temp = np.std(res_temp, axis=0)

                # print(res_temp.shape)

                length = 6

                s = np.zeros((length, length))
                p = np.zeros((length, length))

                for i in range(length):
                    for j in range(length):
                        s[i, j], p[i, j] = ttest_rel(res_temp.T[i], res_temp.T[j])

                if metric_id == 0:
                    _ = np.where((p < alpha) * (s > 0))
                else:
                    _ = np.where((p < alpha) * (s < 0))

                conclusions = [list(1 + _[1][_[0] == i])
                            for i in range(length)]

                t.append(["%s" % row] + ["%.3f" % v for v in e_res_temp])
                t.append([''] + ["%.3f" % v for v in std_res_temp])

                t.append([''] + [", ".join(["%i" % i for i in c])
                        if len(c) > 0 and len(c) < length-1 else ("all" if len(c) == length-1 else "---")
                        for c in conclusions])

            # print(tabulate(t, detector_names, floatfmt="%.3f", tablefmt="latex_booktabs")) 
            with open('tables/table_%s_%s_%s.txt' % (metric_names[metric_id], drf_type, rec), 'w') as f:
                f.write(tabulate(t, detector_names, floatfmt="%.3f", tablefmt="latex_booktabs"))
            # exit()


