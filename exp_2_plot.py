import e2_config
import numpy as np

drf_types = e2_config.e2_drift_types()
recurring = e2_config.e2_recurring()

base_detector_names = e2_config.e2_clf_names()
metrics_names = e2_config.metrics_names()

for rec in recurring:
    for drf_type in drf_types:
        res_clf = np.load('results_ex2/clf_15feat_5drifts_%s_%s.npy' %(drf_type, rec))
        res_clf_mean = np.mean(res_clf, axis=0)
        print(res_clf_mean)