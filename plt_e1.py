"""
Plot heatmaps -- sensitivity & number of detectors parameters 
Each type of stream (with sudden, gradual and incremental drift) and subspace size (1,2,3) in separate output image
"""

import numpy as np
import matplotlib.pyplot as plt
from methods.dderror import dderror
import e1_config

subspace_sizes = e1_config.e1_subspace_sizes()
drf_types = e1_config.e1_drift_types()
th_arr = e1_config.e1_drf_threshold()
det_arr = e1_config.e1_n_detectors()

gd = 10

addr_a = np.linspace(0, len(th_arr)-1, gd)
addr_b = np.linspace(0, len(det_arr)-1, gd)

val_a = ['%.2f' % v for v in np.linspace(0, 1, gd)]
val_b = ['%.0f' % v for v in np.linspace(1, 100, gd)]

for ss_id, ss in enumerate(subspace_sizes):
    for drf_id, drf in enumerate(drf_types):
        fig, ax = plt.subplots(1, 1, figsize=(7,7),
                            sharey=True)

        res_clf = np.load('results_ex1/clf_15feat_5drifts_%s_%isubspace_size.npy' %  (drf, ss))
        res_arr = np.load('results_ex1/drf_arr_15feat_5drifts_%s_%isubspace_size.npy' %  (drf, ss))

        res_clf_mean = np.mean(res_clf, axis=0)
        dderror_arr = np.zeros((res_arr.shape[0],
                                res_arr.shape[1],
                                res_arr.shape[2], 3))

        for rep in range(res_arr.shape[0]):
            for th in range(res_arr.shape[1]):
                for det in range(res_arr.shape[2]):
                    real_drf = np.argwhere(res_arr[rep, th, det, 0]==2).flatten()
                    det_drf = np.argwhere(res_arr[rep, th, det, 1]==2).flatten()
                    err = dderror(real_drf, det_drf, res_arr.shape[4])
                    dderror_arr[rep,th,det] = err

        res_arr_mean = np.mean(dderror_arr, axis=0)

        # """
        # Plot clf
        # """
        # aa = ax[1,1]
        # im = aa.imshow(res_clf_mean, cmap='binary', origin='lower',
        #           vmin=.5, vmax=1)

        for i in range(3):
            # normalizacja
            res_arr_mean[:,:,i] -= np.min(res_arr_mean[:,:,i])
            res_arr_mean[:,:,i] /= np.max(res_arr_mean[:,:,i])


        ax.imshow(res_arr_mean, origin='lower')
        ax.set_ylabel("sensitivity")
        # if j==1:
        ax.set_xlabel("#detectors")

        ax.set_yticks(list(range(len(th_arr))))
        ax.set_yticklabels(['%.2f' % v for v in th_arr])
        ax.set_xticks(list(range(len(det_arr))))
        ax.set_xticklabels(['%.0f' % v for v in det_arr])
        ax.set_yticks(addr_a)
        ax.set_yticklabels(val_a, fontsize=8)
        ax.set_xticks(addr_b)
        ax.set_xticklabels(val_b, fontsize=8)

        ax.grid(ls=":")
        [ax.spines[spine].set_visible(False)
            for spine in ['top', 'bottom', 'left', 'right']]

        ax.set_title('Drift Detection Errors Heatmap')
  
        plt.tight_layout()
        plt.savefig('figures_ex1/err_%s_%i.png' % (drf, ss))
        plt.savefig('pub_figures/err_%s_%i.eps' % (drf, ss))
        plt.savefig('foo.png')

        plt.close()

        # exit()
