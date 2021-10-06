import e2_config
import numpy as np
import matplotlib.pyplot as plt

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

        plt.close()
        fig, ax = plt.subplots(1, 2, figsize=(20, 6), dpi=300)
        fig.suptitle("%s, %s" % (rec, drf_type), fontsize=15)

        for det_id, det_name in enumerate(detector_names):
            ax[0].plot(res_clf_mean[det_id], label=det_name, c=colors[det_id], linestyle=lss[det_id], alpha=0.7)

        ax[0].legend()
        ax[0].set_title("Classification accuracy")

        real = np.argwhere(res_arr[0,0,0,:]==2)
        ax[1].set_xlim(0,len(res_clf_mean[1]))
        ax[1].vlines(real, 0, len(detector_names)-1)

        for det_id, det_name in enumerate(detector_names):
            drf_cnt=np.ones((res_arr.shape[3]))

            for rep in range(res_arr.shape[0]):
                detected = np.argwhere(res_arr[rep,det_id,1,:]==2)
                drf_cnt[detected] +=1
            
            det_sum = np.argwhere(drf_cnt>1)
            drf_cnt = drf_cnt[drf_cnt>1]

            det_y = [det_id for i in range(len(det_sum))]
            ax[1].scatter(det_sum, det_y, alpha=0.4, s=drf_cnt*15, c=colors[det_id])

        ax[1].set_yticks(list(range(len(detector_names))))
        ax[1].set_yticklabels(detector_names)
        ax[1].set_title("Detections")
        
        fig.subplots_adjust(top=0.93)
        plt.tight_layout()
        plt.savefig("figures_ex2/%s_%s.png" % (rec, drf_type))

        # exit()
