"""
Tables and statistical analysis for experiment 2
"""

def dderror(drifts, detections, n_chunks):

    if len(detections) == 0: # no detections
        detections = np.arange(n_chunks)

    n_detections = len(detections)
    n_drifts = len(drifts)

    ddm = np.abs(drifts[:, np.newaxis] - detections[np.newaxis,:])

    cdri = np.min(ddm, axis=0)
    cdec = np.min(ddm, axis=1)

    d1metric = np.mean(cdri)
    d2metric = np.mean(cdec)
    cmetric = np.abs((n_drifts/n_detections)-1)

    return d1metric, d2metric, cmetric
    # d1 - detection from nearest drift
    # d2 - drift from nearest detection

# import e2_config
# import e2_config_hddm
import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_rel


drf_types = ['sudden', 'gradual', 'incremental']
recurring = ['recurring', 'not-recurring']

detector_names = ['DDM', 'EDDM', 'ADWIN', 'SDDE',
                  ' HDDM_W', 'HDDM_A']
replications = 10

drifts_n = [3,5,7]
features_n = [10,15,20]


results_all = np.zeros((replications, len(recurring), len(
    drf_types), len(features_n), len(drifts_n), len(detector_names), 4))
# replications, recurring, drf_types, features, drifts_num, detectors, (acc, d1, d2, cnt)

# str_names=[]
metric_names = ["d1", "d2", "cnt_ratio"]

for rec_id, rec in enumerate(recurring):
    for drf_id, drf_type in enumerate(drf_types):

        #features, drifts
        for f_id, f in enumerate(features_n):
            for d_id, d in enumerate(drifts_n):

                res_clf = np.load(
                    'results_ex2_d_f_45/clf_%ifeat_%idrifts_%s_%s_all.npy' % (f, d, drf_type, rec))
                # replications x detectors x chunks-1
                res_clf = res_clf[:,:-2] # remove always and never

                res_arr = np.load(
                    'results_ex2_d_f_45/drf_arr_%ifeat_%idrifts_%s_%s_all.npy' % (f, d, drf_type, rec))
                # replications x detectors x (real, detected) x chunks-1
                res_arr = res_arr[:,:-2] # remove always and never

                print(res_clf.shape, res_arr.shape)

                dderror_arr = np.zeros((res_arr.shape[0], res_arr.shape[1], 3))

                for rep in range(res_arr.shape[0]):
                    for det in range(res_arr.shape[1]):
                        real_drf = np.argwhere(
                            res_arr[rep, det, 0] == 2).flatten()
                        det_drf = np.argwhere(
                            res_arr[rep, det, 1] == 2).flatten()
                        err = dderror(real_drf, det_drf, res_arr.shape[3])
                        dderror_arr[rep, det] = err

                #accuracy
                results_all[:, rec_id, drf_id, f_id,
                            d_id, :, 0] = np.mean(res_clf, axis=2)
                #d1
                results_all[:, rec_id, drf_id, f_id,
                            d_id, :, 1] = dderror_arr[:, :, 0]
                #d2
                results_all[:, rec_id, drf_id, f_id,
                            d_id, :, 2] = dderror_arr[:, :, 1]
                #cnt
                results_all[:, rec_id, drf_id, f_id,
                            d_id, :, 3] = dderror_arr[:, :, 2]

                # str_names.append("%s_%s_drfits_%i_features_%i" % (rec, drf_type, d, f))
np.save("results_2", results_all)
# print(len(str_names))
print(results_all.shape)  # 10, 2, 3, 3, 3, 8, 4
print(results_all.shape) # reps x rec x drf type x features x drf num x detectors x metrics

results_all = results_all[:,:,:,:,:,:,1:] # rm accuracy
print(results_all.shape) # reps x rec x drf type x features x drf num x detectors x metrics

results_all_mean = np.mean(results_all, axis=0)
results_all_std = np.std(results_all, axis=0)
print(results_all_mean.shape) #  rec x drf type x features x drf num x detectors x metrics

#for every metric
for metric_id in range(3):

    alpha = 0.05

    t = []
    t.append(["", "(1)", "(2)", "(3)", "(4)","(5)", "(6)"])


    for drf_id, drf_type in enumerate(drf_types):  # 3
        for rec_id, rec in enumerate(recurring):  # 2
            for n_f_id, n_f in enumerate(features_n):
                for n_d_id, n_d in enumerate(drifts_n):
                    
                    temp = results_all[:,rec_id,drf_id,n_f_id,n_d_id]
                    print(temp.shape)
                    str_name = '%s | %i dim | %s | %i' % (rec[0], n_f, drf_type[0], n_d)                
                    
                    """
                    t-test
                    """

                    metric_temp = temp[:, :, metric_id]
                    length = 6

                    s = np.zeros((length, length))
                    p = np.zeros((length, length))

                    for i in range(length):
                        for j in range(length):
                            s[i, j], p[i, j] = ttest_rel(metric_temp.T[i], metric_temp.T[j])

                    _ = np.where((p < alpha) * (s < 0))
                    conclusions = [list(1 + _[1][_[0] == i]) for i in range(length)]

                    t.append(["%s" % str_name] + ["%.3f" % v for v in results_all_mean[rec_id,drf_id,n_f_id,n_d_id,:,metric_id]])
                    t.append([''] + ["%.3f" % v for v in results_all_std[rec_id,drf_id,n_f_id,n_d_id,:,metric_id]])

                    t.append([''] + [", ".join(["%i" % i for i in c])
                                    if len(c) > 0 and len(c) < length-1 else ("all" if len(c) == length-1 else "---")
                                    for c in conclusions])


    # print(tabulate(t, detector_names, floatfmt="%.3f", tablefmt="latex_booktabs"))
    with open('tables/table_%s.txt' % (metric_names[metric_id]), 'w') as f:
        f.write(tabulate(t, detector_names,floatfmt="%.3f", tablefmt="latex_booktabs"))
        
    # exit()
