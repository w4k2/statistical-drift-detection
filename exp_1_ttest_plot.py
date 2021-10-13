import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import e1_config
from methods.dderror import dderror

subspace_sizes = e1_config.e1_subspace_sizes()
drf_types = e1_config.e1_drift_types()
th_arr = e1_config.e1_drf_threshold()
det_arr = e1_config.e1_n_detectors()

alpha = .25

for ss_id, ss in enumerate(subspace_sizes):
    for drf_id, drf in enumerate(drf_types):
        res_arr = np.load('results_ex1/drf_arr_15feat_5drifts_%s_%isubspace_size.npy' %  (drf, ss))

        dderror_arr = np.zeros((res_arr.shape[0],res_arr.shape[1], res_arr.shape[2], 3))

        for rep in range(res_arr.shape[0]):
            for th in range(res_arr.shape[1]):
                for det in range(res_arr.shape[2]):
                    real_drf = np.argwhere(res_arr[rep, th, det, 0]==2).flatten()
                    det_drf = np.argwhere(res_arr[rep, th, det, 1]==2).flatten()
                    err = dderror(real_drf, det_drf, res_arr.shape[4])
                    dderror_arr[rep,th,det] = err

        # a = 8 th[1]
        # b = 9 det[2]
        # n = 10 rep [0]

        for metric in range(3):

            dderror_metric = dderror_arr[:,:,:,metric]
            
            a = dderror_metric.shape[1]
            b = dderror_metric.shape[2]

            # # Gather data
            # data = np.random.normal(size=(a,b,n))

            # Calculate mean
            mean_data = np.mean(dderror_metric,axis=0) # th x det

            # best mean
            best_mean = np.min(mean_data)
            # print(mean_data==best_mean)
            # exit()

            # Calculate CMP
            cmp = np.mean(dderror_metric[:,mean_data==best_mean], axis=1)

            # Mask
            mask = mean_data == best_mean

            # CMP-img
            pimg = np.zeros((a,b))

            # Test
            for i in range(a):
                for j in range(b):
                    pimg[i,j] = ttest_rel(cmp, dderror_metric[:,i,j]).pvalue

            # DEP-img
            dimg = pimg > alpha

            candidates = dimg + mask

            thresholds, detectors = np.where(candidates)

            bd = np.min(detectors)
            bt = np.min(thresholds[detectors==bd])

            print(bd, bt)

            mat = np.copy(candidates).astype('int')
            mat[bt, bd]=2

            """
            Plot.
            """
            fig, ax = plt.subplots(2,3,figsize=(8,5))

            ax[0,0].imshow(mean_data, vmin=-1, vmax=1)
            ax[0,1].imshow(mask)
            ax[0,2].imshow(pimg)
            ax[1,0].imshow(dimg)
            ax[1,1].imshow(candidates)
            ax[1,2].imshow(mat)

            for i in range(2):
                for j in range(3):
                    ax[i,j].set_yticks(list(range(len(th_arr))))
                    ax[i,j].set_yticklabels(['%.2f' % v for v in th_arr], fontsize=6)
                    ax[i,j].set_ylabel("Threshold")

                    ax[i,j].set_xticks(list(range(len(det_arr))))
                    ax[i,j].set_xticklabels(det_arr, fontsize=6)
                    ax[i,j].set_xlabel("n detectors")


            plt.tight_layout()
            plt.savefig('figures_ex1/metric%i_%s_%i.png' % (metric, drf, ss))

            plt.clf()
            exit()