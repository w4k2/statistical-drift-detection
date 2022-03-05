import e2_config
import e2_config_hddm

import numpy as np

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

detector_names_original = e2_config.e2_clf_names()
detector_names_hddm = e2_config_hddm.e2_clf_names()

for str_id, stream in enumerate(streams):
    #drifts, drf types
    for d_id, d in enumerate(drifts_n):
        for d_type_id, d_type in enumerate(drf_types):

            res_arr_original = np.load('results_ex2_real/drf_arr_str_%s.csv_%s_%idrfs.npy' %(stream, d_type, d))
            res_arr_hddm = np.load('results_ex2_real/drf_arr_str_%s.csv_%s_%idrfs.npy_hddm.npy' %(stream, d_type, d))

            res_clf_original = np.load('results_ex2_real/clf_str_%s.csv_%s_%idrfs.npy' %(stream, d_type, d))
            res_clf_hddm = np.load('results_ex2_real/clf_str_%s.csv_%s_%idrfs.npy_hddm.npy' %(stream, d_type, d))

            new_clf = np.concatenate((res_clf_original, res_clf_hddm), axis=1)
            new_arr = np.concatenate((res_arr_original, res_arr_hddm), axis=1)

            # print(new_clf.shape) 10, 8, 399

            #always, never na ostatnie 2 
            a_n_clf = np.copy(new_clf[:,[4,5]])
            a_n_arr = np.copy(new_arr[:,[4,5]])

            new_clf[:,[4,5]] = np.copy(new_clf[:,[6,7]])
            new_clf[:,[6,7]] = a_n_clf

            new_arr[:,[4,5]] = np.copy(new_arr[:,[6,7]])
            new_arr[:,[6,7]] = a_n_arr

            np.save('results_ex2_real/drf_arr_str_%s.csv_%s_%idrfs_all.npy' %(stream, d_type, d), new_arr)
            np.save('results_ex2_real/clf_str_%s.csv_%s_%idrfs_all.npy' %(stream, d_type, d), new_clf)


