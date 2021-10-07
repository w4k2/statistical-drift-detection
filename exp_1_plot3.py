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
    plt.close()
    fig, ax = plt.subplots(3, 3, figsize=(18, 18), dpi=300)

    for drf_id, drf in enumerate(drf_types):
        # if drf == 'incremental':
        #     continue

        res_arr = np.load('results_ex1/drf_arr_15feat_5drifts_%s_%isubspace_size.npy' %  (drf, ss))
        print(res_arr.shape) #replications x threshold x detectors x (real, detected) x chunks-1

        dderror_arr = np.zeros((res_arr.shape[0],res_arr.shape[1], res_arr.shape[2]))

        for rep in range(res_arr.shape[0]):
            for th in range(res_arr.shape[1]):
                for det in range(res_arr.shape[2]):
                    real_drf = np.argwhere(res_arr[rep, th, det, 0]==2).flatten()
                    det_drf = np.argwhere(res_arr[rep, th, det, 1]==2).flatten()
                    err = dderror(real_drf, det_drf, res_arr.shape[4])
                    dderror_arr[rep,th,det] = err

        res_arr_mean = np.mean(dderror_arr, axis=0)
        print(res_arr_mean.shape) # th x det

        ax[drf_id,0].imshow(res_arr_mean, cmap='binary', origin='upper')

        ax[drf_id,0].set_yticks(list(range(len(th_arr))))
        ax[drf_id,0].set_yticklabels(['%.2f' % v for v in th_arr])
        ax[drf_id,0].set_ylabel("Threshold")

        ax[drf_id,0].set_xticks(list(range(len(det_arr))))
        ax[drf_id,0].set_xticklabels(det_arr)
        ax[drf_id,0].set_xlabel("n detectors")
        ax[drf_id,0].set_title("%s %i ss" % (drf, ss))

        opts = np.argmin(res_arr_mean, axis=0)
        ax[drf_id,1].plot(det_arr, th_arr[opts])

        ax[drf_id,1].set_xlabel("n detectors")

        ax[drf_id,1].set_yticks(th_arr)
        ax[drf_id,1].set_ylabel("Threshold")
        ax[drf_id,1].set_ylim(0,1)

        ax[drf_id,1].grid()
        ax[drf_id,1].set_title("opt %s %i ss" % (drf, ss))


        binary = res_arr_mean == np.min(res_arr_mean)
        ax[drf_id,2].imshow(binary, cmap='binary', origin='upper')
        ax[drf_id,2].set_yticks(list(range(len(th_arr))))
        ax[drf_id,2].set_yticklabels(['%.2f' % v for v in th_arr])
        ax[drf_id,2].set_ylabel("Threshold")

        ax[drf_id,2].set_xticks(list(range(len(det_arr))))
        ax[drf_id,2].set_xticklabels(det_arr)
        ax[drf_id,2].set_xlabel("n detectors")
        ax[drf_id,2].set_title("%s %i ss" % (drf, ss))

    plt.tight_layout()
    plt.savefig('figures_ex1/err_%i.png' % ss)
