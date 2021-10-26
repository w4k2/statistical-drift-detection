import e2_config
import numpy as np
import matplotlib.pyplot as plt
from methods.dderror import dderror

drf_types = e2_config.e2_drift_types()
recurring = e2_config.e2_recurring()

detector_names = [".45", ".5", ".55", ".6", ".7", ".8"]
y_values = np.array([.45, .5, .55, .6 ,.7, .8])
metrics_names = e2_config.metrics_names()

colors = ['green', 'brown', 'dodgerblue', 'purple', 'gold', 'gray']
colors = ['#333', '#333']
black_rainbow = ['#333', '#777']

base_width = 12
fig, axx = plt.subplots(2, 3, figsize=(base_width, base_width/1.618))

for rec_id, rec in enumerate(recurring):
    for drf_id, drf_type in enumerate(drf_types):
        ax = axx[rec_id, drf_id]
        res_clf = np.load('results_ex2_2/clf_15feat_5drifts_%s_%s.npy' %(drf_type, rec))
        # replications x detectors x chunks-1
        res_clf_mean = np.mean(res_clf, axis=0)
        print(res_clf_mean.shape)

        res_arr = np.load('results_ex2_2/drf_arr_15feat_5drifts_%s_%s.npy' %(drf_type, rec))
        # replications x detectors x (real, detected) x chunks-1
        dderror_arr = np.zeros((res_arr.shape[0], res_arr.shape[1], 3))

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

        ax.set_title("%s %s drift" % (drf_type, rec), fontsize=12)

        real = np.squeeze(np.argwhere(res_arr[0,0,0,:]==2))

        print(real)
        ax.set_xlim(0,len(res_clf_mean[1]))
        ax.vlines(real, 0, len(detector_names)-1, color='#000', lw=1, ls=":", zorder=1)
        ax.hlines(y_values, 0, 200, color='#333', lw=1, zorder=1)

        for det_id, det_name in enumerate(detector_names):
            drf_cnt=np.ones((res_arr.shape[3]))

            for rep in range(res_arr.shape[0]):
                detected = np.argwhere(res_arr[rep,det_id,1,:]==2)
                drf_cnt[detected] +=1

            det_sum = np.argwhere(drf_cnt>1)
            drf_cnt = drf_cnt[drf_cnt>1]

            det_y = [det_id for i in range(len(det_sum))]

            aaa = np.ones_like(det_sum) * y_values[det_id]

            print('DC', drf_cnt)

            ax.scatter(det_sum,
                       aaa,
                       alpha=1,
                       s=drf_cnt*10,
                       c='#333',
                       label=res_arr_mean[det_id])


        ax.set_xticks(real)
        ax.set_xticklabels(['%i' % zzz for zzz in (real+1)])
        #ax.grid(ls=':')
        ax.set_yticks(y_values)
        ax.set_yticklabels(["%.2f" % v for v in y_values])
        ax.set_ylim(.4,.85)
        ax.set_xlim(0,200)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if rec_id == 1:
            ax.set_xlabel('Number of chunks processed')
        if drf_id == 0:
            ax.set_ylabel('Detector treshold')

        fig.subplots_adjust(top=0.93)
        plt.tight_layout()
        plt.savefig("figures_ex2/sensitivity_%s_%s.png" % (rec, drf_type))
        plt.savefig("foo.png")
        #exit()
