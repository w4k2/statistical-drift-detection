from numpy.core.fromnumeric import mean
import e2_config
import numpy as np
from methods.dderror import dderror
from tabulate import tabulate

drf_types = e2_config.e2_drift_types()
recurring = e2_config.e2_recurring()

detector_names = e2_config.e2_clf_names()
metrics_names = e2_config.metrics_names()
replications = e2_config.e2_replications()

results_all = np.zeros((replications, len(recurring) * len(drf_types), len(detector_names), 4))
                # replications, recurring + drf_types, detectors, (acc, d1, d2, cnt)

str_cnt = 0
str_names=[]

for rec_id, rec in enumerate(recurring):
    for drf_id, drf_type in enumerate(drf_types):
        res_clf = np.load('results_ex2/clf_15feat_5drifts_%s_%s.npy' %(drf_type, rec))
        # replications x detectors x chunks-1

        res_arr = np.load('results_ex2/drf_arr_15feat_5drifts_%s_%s.npy' %(drf_type, rec))
        # replications x detectors x (real, detected) x chunks-1

        dderror_arr = np.zeros((res_arr.shape[0], res_arr.shape[1], 3))

        for rep in range(res_arr.shape[0]):
            for det in range(res_arr.shape[1]):
                real_drf = np.argwhere(res_arr[rep, det, 0]==2).flatten()
                det_drf = np.argwhere(res_arr[rep, det, 1]==2).flatten()
                err = dderror(real_drf, det_drf, res_arr.shape[3])
                dderror_arr[rep, det] = err
        
        #accuracy
        results_all[:, str_cnt, :, 0] = np.mean(res_clf, axis=2)
        #d1
        results_all[:, str_cnt, :, 1] = dderror_arr[:,:,0]
        #d2
        results_all[:, str_cnt, :, 2] = dderror_arr[:,:,1]
        #cnt
        results_all[:, str_cnt, :, 3] = dderror_arr[:,:,2]

        str_cnt +=1
        str_names.append("%s_%s" % (rec, drf_type))

mean_res_all = np.mean(results_all, axis=0)

mean_acc = np.concatenate((np.array(str_names)[:, np.newaxis], mean_res_all[:,:,0]), axis=1)
mean_d1 = np.concatenate((np.array(str_names)[:, np.newaxis], mean_res_all[:,:,1]), axis=1)
mean_d2 = np.concatenate((np.array(str_names)[:, np.newaxis], mean_res_all[:,:,2]), axis=1)
mean_cnt = np.concatenate((np.array(str_names)[:, np.newaxis], mean_res_all[:,:,3]), axis=1)

print("\n Mean accuracy")
print(tabulate(mean_acc, headers=detector_names, floatfmt=".3f"))

print("\n Mean d1")
print(tabulate((mean_d1), headers=detector_names, floatfmt=".3f"))

print("\n Mean d2")
print(tabulate(mean_d2, headers=detector_names, floatfmt=".3f"))

print("\n Mean cnt")
print(tabulate(mean_cnt, headers=detector_names, floatfmt=".3f"))