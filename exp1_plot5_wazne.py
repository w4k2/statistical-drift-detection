import e2_config
import numpy as np
import matplotlib.pyplot as plt
from methods.dderror import dderror

drf_types = e2_config.e2_drift_types()
recurring = e2_config.e2_recurring()

detector_names = e2_config.e2_clf_names()
metrics_names = e2_config.metrics_names()

colors = ['green', 'brown', 'dodgerblue', 'purple', 'gold', 'gray']
lss = ['-', '-', '-', '-', '-', '-']

for rec_id, rec in enumerate(recurring):
    for drf_id, drf_type in enumerate(drf_types):
        res_clf = np.load('results_ex2/clf_15feat_5drifts_%s_%s.npy' %(drf_type, rec))
        # replications x detectors x chunks-1
        res_clf_mean = np.mean(res_clf, axis=0)

        res_arr = np.load('results_ex2/drf_arr_15feat_5drifts_%s_%s.npy' %(drf_type, rec))
        # replications x detectors x (real, detected) x chunks-1

        dderror_arr = np.zeros((res_arr.shape[0],res_arr.shape[1]))

        for rep in range(res_arr.shape[0]):
            for det in range(res_arr.shape[1]):
                real_drf = np.argwhere(res_arr[rep, det, 0]==2).flatten()
                det_drf = np.argwhere(res_arr[rep, det, 1]==2).flatten()
                err = dderror(real_drf, det_drf, res_arr.shape[3])
                dderror_arr[rep, det] = err

        res_arr_mean = np.mean(dderror_arr, axis=0)

        # print(res_arr_mean)
        # exit()

        """
        Plot
        """

        plt.close()
        fig, ax = plt.subplots(2, 2, figsize=(20, 12), dpi=300)
        fig.suptitle("%s, %s" % (rec, drf_type), fontsize=15)

        for det_id, det_name in enumerate(detector_names):
            ax[0,0].plot(res_clf_mean[det_id], label=det_name, c=colors[det_id], linestyle=lss[det_id], alpha=0.7)

        ax[0,0].legend()
        ax[0,0].set_title("Classification accuracy")

        real = np.argwhere(res_arr[0,0,0,:]==2)
        ax[0,1].set_xlim(0,len(res_clf_mean[1]))
        ax[0,1].vlines(real, 0, len(detector_names)-1)
        ax[1,1].vlines(real, 0, len(detector_names)-1, color='black', ls=":")

        for det_id, det_name in enumerate(detector_names):
            drf_cnt=np.ones((res_arr.shape[3]))

            zzz = res_arr[:,:,1,:]
            zzz = np.swapaxes(zzz, 0,1)
            zzz = np.reshape(zzz, (-1, 199))


            print(drf_cnt, drf_cnt.shape)
            print(zzz, zzz.shape)

            for rep in range(res_arr.shape[0]):
                detected = np.argwhere(res_arr[rep,det_id,1,:]==2)
                drf_cnt[detected] +=1

            det_sum = np.argwhere(drf_cnt>1)
            aaa = np.copy(drf_cnt)
            drf_cnt = drf_cnt[drf_cnt>1]

            det_y = [det_id for i in range(len(det_sum))]
            ax[0,1].scatter(det_sum, det_y, alpha=1, s=drf_cnt*15, c=colors[det_id], label=res_arr_mean[det_id])

            ax[1,1].plot((aaa-1)/11 + det_id, c=colors[det_id])
            #print(det_sum)

            ax[1,0].imshow(zzz, origin='lower', cmap='seismic')

        ax[1,0].set_yticks([5,15,25,35,45,55])
        ax[1,0].set_yticklabels("%s - %.3f" % (d, res_arr_mean[i]) for i, d in enumerate(detector_names))

        ax[0,1].set_yticks(list(range(len(detector_names))))
        ax[0,1].set_yticklabels("%s - %.3f" % (d, res_arr_mean[i]) for i, d in enumerate(detector_names))
        ax[0,1].set_title("Detections")
        ax[1,1].grid(ls=":")
        ax[1,1].set_yticks(list(range(len(detector_names))))
        ax[1,1].set_yticklabels("%s - %.3f" % (d, res_arr_mean[i]) for i, d in enumerate(detector_names))

        # ax[1].legend(loc=3, facecolor='white', framealpha=.95, edgecolor='white', ncol=3)


        fig.subplots_adjust(top=0.93)
        plt.tight_layout()
        plt.savefig("figures_ex2/%s_%s.png" % (rec, drf_type))
        plt.savefig('foo.png')

        # exit()