import e2_config
import e2_config_hddm

import numpy as np

drf_types = e2_config.e2_drift_types()
recurring = e2_config.e2_recurring()

detector_names_original = e2_config.e2_clf_names()
detector_names_hddm = e2_config_hddm.e2_clf_names()

replications = e2_config.e2_replications()

drifts_n = e2_config_hddm.e2_n_drifts()
features_n = e2_config_hddm.e2_n_features()


for rec_id, rec in enumerate(recurring):
    for drf_id, drf_type in enumerate(drf_types):

        #features, drifts
        for f_id, f in enumerate(features_n):
            for d_id, d in enumerate(drifts_n):

                res_clf_original = np.load('results_ex2_d_f_45/clf_%ifeat_%idrifts_%s_%s.npy' %(f, d, drf_type, rec))
                # replications x detectors(6) x chunks-1

                res_arr_original = np.load('results_ex2_d_f_45/drf_arr_%ifeat_%idrifts_%s_%s.npy' %(f, d, drf_type, rec))
                # replications x detectors(6) x (real, detected) x chunks-1

                res_clf_hddm = np.load('results_ex2_d_f_45/clf_%ifeat_%idrifts_%s_%s_2.npy' %(f, d, drf_type, rec))
                # replications x detectors(2) x chunks-1

                res_arr_hddm = np.load('results_ex2_d_f_45/drf_arr_%ifeat_%idrifts_%s_%s_2.npy' %(f, d, drf_type, rec))
                # replications x detectors(2) x (real, detected) x chunks-1

                new_clf = np.concatenate((res_clf_original, res_clf_hddm), axis=1)
                new_arr = np.concatenate((res_arr_original, res_arr_hddm), axis=1)

                np.save('results_ex2_d_f_45/clf_%ifeat_%idrifts_%s_%s_all.npy' %(f, d, drf_type, rec), new_clf)
                np.save('results_ex2_d_f_45/drf_arr_%ifeat_%idrifts_%s_%s_all.npy' %(f, d, drf_type, rec), new_arr)


