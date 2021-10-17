import e2_config
import numpy as np
import matplotlib.pyplot as plt
from methods.dderror import dderror
import matplotlib.colors


drf_types = e2_config.e2_drift_types()
recurring = e2_config.e2_recurring()
detector_names = e2_config.e2_clf_names()

drifts_n = e2_config.e2_n_drifts()
features_n = e2_config.e2_n_features()

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "lemonchiffon", "black"])

for rec_id, rec in enumerate(recurring):
    for drf_id, drf_type in enumerate(drf_types):

        # if drf_type != 'sudden':
        #     continue

        plt.close()
        fig, ax = plt.subplots(len(drifts_n), len(features_n), figsize=(20, 9))
        fig.suptitle("%s, %s" % (rec, drf_type), fontsize=15)

        #features, drifts
        for f_id, f in enumerate(features_n):
            for d_id, d in enumerate(drifts_n):

                # res_clf = np.load('results_ex2_d_f_45/clf_%ifeat_%idrifts_%s_%s.npy' %(f, d, drf_type, rec))
                # replications x detectors x chunks-1
                # res_clf_mean = np.mean(res_clf, axis=0)

                res_arr = np.load('results_ex2_d_f_45/drf_arr_%ifeat_%idrifts_%s_%s.npy' %(f, d, drf_type, rec))
                # replications x detectors x (real, detected) x chunks-1

                # dderror_arr = np.zeros((res_arr.shape[0],res_arr.shape[1], 3))

                # for rep in range(res_arr.shape[0]):
                #     for det in range(res_arr.shape[1]):
                #         real_drf = np.argwhere(res_arr[rep, det, 0]==2).flatten()
                #         det_drf = np.argwhere(res_arr[rep, det, 1]==2).flatten()
                #         err = dderror(real_drf, det_drf, res_arr.shape[3])
                #         dderror_arr[rep, det] = err

                # res_arr_mean = np.mean(dderror_arr, axis=0)

                """
                Plot
                """
                ax[d_id,f_id].set_title("drifts: %i, features: %i" % (d, f))

                for det_id, det_name in enumerate(detector_names):
                    drf_cnt=np.ones((res_arr.shape[3]))

                    zzz = res_arr[:,:,1,:]
                    zzz[zzz==1]=0 # warningi przeszkadzaja

                    mask = zzz==0
                    mask[:,[0,1,2,4,5],:]=False
                    zzz[mask]=1

                    zzz = np.swapaxes(zzz, 0,1)
                    zzz = np.reshape(zzz, (-1, 199))

                    # print(zzz, zzz.shape)

                    ax[d_id,f_id].imshow(zzz, cmap= cmap, origin='lower', interpolation='none')

                ax[d_id,f_id].set_yticks([5,15,25,35,45,55])
                ax[d_id,f_id].set_yticklabels("%s" % d for i, d in enumerate(detector_names))

        plt.tight_layout()
        fig.subplots_adjust(top=0.93)
        plt.savefig("figures_ex2/d_f_pix_%s_%s.png" % (rec, drf_type))
        # plt.savefig('foo.png')
        # exit()
