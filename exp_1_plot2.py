import numpy as np
import matplotlib.pyplot as plt
from methods import dderror
import e1_config

subspace_sizes = e1_config.e1_subspace_sizes()
drf_types = e1_config.e1_drift_types()
th_arr = e1_config.e1_drf_threshold()
det_arr = e1_config.e1_n_detectors()

fig, ax = plt.subplots(3, 3, figsize=(18, 18), dpi=300)
fig.suptitle("Classification score", fontsize=25)

for ss_id, ss in enumerate(subspace_sizes):
    for drf_id, drf in enumerate(drf_types):

        res_clf = np.load('results_ex1/clf_15feat_5drifts_%s_%isubspace_size.npy' %  (drf, ss))
        print(res_clf.shape) #replications x threshold x detectors

        res_clf_mean = np.mean(res_clf, axis=0)

        ax[drf_id,ss_id].imshow(res_clf_mean, cmap='binary', origin='lower')

        ax[drf_id,ss_id].set_yticks(list(range(len(th_arr))))
        ax[drf_id,ss_id].set_yticklabels(['%.2f' % v for v in th_arr])
        ax[drf_id,ss_id].set_ylabel("Threshold")

        ax[drf_id,ss_id].set_xticks(list(range(len(det_arr))))
        ax[drf_id,ss_id].set_xticklabels(['%.0f' % v for v in det_arr])
        ax[drf_id,ss_id].set_xlabel("n detectors")

        ax[drf_id,ss_id].set_title("%s %i ss" % (drf, ss))

plt.tight_layout()
fig.subplots_adjust(top=0.93)
plt.savefig('figures_ex1/clf.png')
plt.close()

fig, ax = plt.subplots(3, 3, figsize=(18, 18), dpi=300)
fig.suptitle("Detection error", fontsize=25)

for ss_id, ss in enumerate(subspace_sizes):
    for drf_id, drf in enumerate(drf_types):

        res_arr = np.load('results_ex1/drf_arr_15feat_5drifts_%s_%isubspace_size.npy' %  (drf, ss))
        print(res_arr.shape) #replications x threshold x detectors x (real, detected) x chunks-1

        dderror_arr = np.zeros((res_arr.shape[0], res_arr.shape[1], res_arr.shape[2]))

        for rep in range(res_arr.shape[0]):
            for th in range(res_arr.shape[1]):
                for det in range(res_arr.shape[2]):
                    real_drf = np.argwhere(res_arr[rep, th, det, 0]==2).flatten()
                    det_drf = np.argwhere(res_arr[rep, th, det, 1]==2).flatten()
                    err = dderror(real_drf, det_drf, res_arr.shape[4])[1]
                    dderror_arr[rep,th,det] = err            

        res_arr_mean = np.mean(dderror_arr, axis=0)

        # for i in range(3):
        #     res_arr_mean[:,:,i] -= np.min(res_arr_mean[:,:,i])
        #     res_arr_mean[:,:,i] /= np.max(res_arr_mean[:,:,i])
        # res_arr_mean = 1 - res_arr_mean



        ax[drf_id,ss_id].imshow(res_arr_mean, cmap='binary', origin='upper')

        ax[drf_id,ss_id].set_yticks(list(range(len(th_arr))))
        ax[drf_id,ss_id].set_yticklabels(['%.2f' % v for v in th_arr])
        ax[drf_id,ss_id].set_ylabel("Threshold")

        ax[drf_id,ss_id].set_xticks(list(range(len(det_arr))))
        ax[drf_id,ss_id].set_xticklabels(det_arr)
        ax[drf_id,ss_id].set_xlabel("n detectors")
    
        ax[drf_id,ss_id].set_title("%s %i ss" % (drf, ss))

plt.tight_layout()
fig.subplots_adjust(top=0.93)
plt.savefig('figures_ex1/err_d2.png')
