import numpy as np
import matplotlib.pyplot as plt
from methods import dderror
import e1_config

subspace_sizes = e1_config.e1_subspace_sizes()
drf_types = e1_config.e1_drift_types()
th_arr = e1_config.e1_drf_threshold()
det_arr = e1_config.e1_n_detectors()

print(th_arr, det_arr)

for ss_id, ss in enumerate(subspace_sizes):
    for drf_id, drf in enumerate(drf_types):
        if drf == 'incremental':
            continue

        res_clf = np.load('results_ex1/clf_15feat_5drifts_%s_%isubspace_size.npy' %  (drf, ss))
        res_arr = np.load('results_ex1/drf_arr_15feat_5drifts_%s_%isubspace_size.npy' %  (drf, ss))

        print(res_clf.shape) #replications x threshold x detectors
        print(res_arr.shape) #replications x threshold x detectors x (real, detected) x chunks-1

        res_clf_mean = np.mean(res_clf, axis=0)
        dderror_arr = np.zeros((res_arr.shape[0],res_arr.shape[1], res_arr.shape[2]))

        for rep in range(res_arr.shape[0]):
            for th in range(res_arr.shape[1]):
                for det in range(res_arr.shape[2]):
                    real_drf = np.argwhere(res_arr[rep, th, det, 0]==2).flatten()
                    det_drf = np.argwhere(res_arr[rep, th, det, 1]==2).flatten()
                    err = dderror(real_drf, det_drf, res_arr.shape[4])
                    dderror_arr[rep,th,det] = err

        res_arr_mean = np.mean(dderror_arr, axis=0)

        """
        Plot clf
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)

        ax.imshow(res_clf_mean, cmap='binary', origin='lower')

        ax.set_yticks(list(range(len(th_arr))))
        ax.set_yticklabels(['%.2f' % v for v in th_arr])
        ax.set_ylabel("Threshold")

        ax.set_xticks(list(range(len(det_arr))))
        ax.set_xticklabels(['%.0f' % v for v in det_arr])
        ax.set_xlabel("n detectors")        
        ax.set_title("%s %i ss" % (drf, ss))

        for _a, __a in enumerate(th_arr):
            for _b, __b in enumerate(det_arr):
                v = res_clf_mean[_a, _b]
                ax.text(_b, _a, "%.3f" % (
                    v) , va='center', ha='center', c='white' if v > np.mean(res_clf_mean) else 'black', fontsize=5)


        plt.tight_layout()
        plt.savefig('figures_ex1/clf_%s_%i.png' % (drf, ss))
        plt.close()

        """
        Plot err
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)

        ax.imshow(res_arr_mean, cmap='binary', origin='upper')

        ax.set_yticks(list(range(len(th_arr))))
        ax.set_yticklabels(['%.2f' % v for v in th_arr])
        ax.set_ylabel("Threshold")

        ax.set_xticks(list(range(len(det_arr))))
        ax.set_xticklabels(det_arr)
        ax.set_xlabel("n detectors")
        ax.set_title("%s %i ss" % (drf, ss))

        for _a, __a in enumerate(th_arr):
            for _b, __b in enumerate(det_arr):
                ax.text(_b, _a, "%.3f" % (
                    res_arr_mean[_a, _b]) , va='center', ha='center', c='red', fontsize=5)
                    

        plt.tight_layout()
        plt.savefig('figures_ex1/err_%s_%i.png' % (drf, ss))

        plt.close()
