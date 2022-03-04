import e2_config
import numpy as np
import matplotlib.pyplot as plt
from methods.dderror import dderror

drf_types = e2_config.e2_drift_types()
recurring = e2_config.e2_recurring()
detector_names = e2_config.e2_clf_names()

for rec_id, rec in enumerate(recurring):
    for drf_id, drf_type in enumerate(drf_types):
        res_clf = np.load('results_ex2/clf_15feat_5drifts_%s_%s.npy' %(drf_type, rec))
        # replications x detectors x chunks-1
        res_clf_mean = np.mean(res_clf, axis=0)

        res_arr = np.load('results_ex2/drf_arr_15feat_5drifts_%s_%s.npy' %(drf_type, rec))
        # replications x detectors x (real, detected) x chunks-1

        dderror_arr = np.zeros((res_arr.shape[0],res_arr.shape[1], 3))

        for rep in range(res_arr.shape[0]):
            for det in range(res_arr.shape[1]):
                real_drf = np.argwhere(res_arr[rep, det, 0]==2).flatten()
                det_drf = np.argwhere(res_arr[rep, det, 1]==2).flatten()
                err = dderror(real_drf, det_drf, res_arr.shape[3])
                dderror_arr[rep, det] = err

        res_arr_mean = np.mean(dderror_arr, axis=0)

        """
        Plot
        """

        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.set_title("%s, %s" % (rec, drf_type), fontsize=15)

        for det_id, det_name in enumerate(detector_names):
            drf_cnt=np.ones((res_arr.shape[3]))

            zzz = res_arr[:,:,1,:]
            zzz[zzz==1]=0 # warningi przeszkadzaja
            zzz = np.swapaxes(zzz, 0,1)
            zzz = np.reshape(zzz, (-1, 199))

            print(zzz, zzz.shape)

            ax.imshow(zzz, cmap= 'binary', origin='lower')

        ax.set_yticks([5,15,25,35,45,55])
        ax.set_yticklabels("%s - %.3f, %.3f, %.3f " %
            (d, res_arr_mean[i,0],res_arr_mean[i,1],res_arr_mean[i,2]) for i, d in enumerate(detector_names))

        plt.tight_layout()
        plt.savefig("figures_ex2/pix_%s_%s.png" % (rec, drf_type))
        plt.savefig('foo.png')
        #exit()
