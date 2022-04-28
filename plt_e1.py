"""
Plot heatmaps -- sensitivity & number of detectors parameters 
Each type of stream (with sudden, gradual and incremental drift) and subspace size (1,2,3) in separate output image
"""

import numpy as np
import matplotlib.pyplot as plt
from methods import dderror
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
        fig, ax = plt.subplots(2, 3, figsize=(12,12/1.618),
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

        """
        Plot clf
        """
        aa = ax[1,1]
        im = aa.imshow(res_clf_mean, cmap='binary', origin='lower',
                  vmin=.5, vmax=1)

        #divider = make_axes_locatable(aa)
        #cax = divider.append_axes('right', size='5%', pad=0.05)
        #fig.colorbar(im, cax=cax, orientation='vertical')


        """
        Plot err
        """
        cmaps = ['binary_r', 'binary_r', 'binary_r']
        metrics = ['d1',  'd2', 'cnt']
        for i in range(3):
            aa = ax[0,i]
            aa.imshow(res_arr_mean[:,:,i]*10, cmap=cmaps[i], origin='lower')

            #divider = make_axes_locatable(aa)
            #cax = divider.append_axes('right', size='5%', pad=0.05)
            #fig.colorbar(im, cax=cax, orientation='vertical')

            vba = res_arr_mean[:,:,i]
            print(i, np.min(vba), np.max(vba))

            # normalizacja
            res_arr_mean[:,:,i] -= np.min(res_arr_mean[:,:,i])
            res_arr_mean[:,:,i] /= np.max(res_arr_mean[:,:,i])



        aa = ax[1,0]
        aa.imshow(res_arr_mean, origin='lower')


        img_mono = res_clf_mean
        img_poly = res_arr_mean
        img_mono -= .5
        img_mono *= 2
        img_mono = 1 - img_mono
        img_mix = img_poly * img_mono[:,:, np.newaxis]
        img_mix = np.mean(img_mix, axis=2)

        aa = ax[1,2]
        im = aa.imshow(img_mix, origin='lower', cmap='binary_r')

        #divider = make_axes_locatable(aa)
        #cax = divider.append_axes('right', size='5%', pad=0.05)
        #fig.colorbar(im, cax=cax, orientation='vertical')


        for j in range(2):
            for k in range(3):
                aa = ax[j,k]
                if k==0:
                    aa.set_ylabel("sensitivity")
                # if j==1:
                aa.set_xlabel("#detectors")

                aa.set_yticks(list(range(len(th_arr))))
                aa.set_yticklabels(['%.2f' % v for v in th_arr])
                aa.set_xticks(list(range(len(det_arr))))
                aa.set_xticklabels(['%.0f' % v for v in det_arr])
                aa.set_yticks(addr_a)
                aa.set_yticklabels(val_a, fontsize=8)
                aa.set_xticks(addr_b)
                aa.set_xticklabels(val_b, fontsize=8)

                aa.grid(ls=":")
                [aa.spines[spine].set_visible(False)
                 for spine in ['top', 'bottom', 'left', 'right']]


        ax[0,0].set_title('Mean closest detection-drift distance')
        ax[0,1].set_title('Mean closest drift-detection distance')
        ax[0,2].set_title('Drift-detection ratio')
        ax[1,0].set_title('Drift Detection Errors Heatmap')
        ax[1,1].set_title('Classification accuracy')
        ax[1,2].set_title('Overall optimization criterion')

        plt.tight_layout()
        plt.savefig('figures_ex1/err_%s_%i.png' % (drf, ss))
        plt.savefig('pub_figures/err_%s_%i.eps' % (drf, ss))
        plt.savefig('foo.png')

        plt.close()

        #exit()
