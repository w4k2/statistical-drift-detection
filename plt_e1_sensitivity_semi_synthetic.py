"""
Plot -- how sensitivity parameter affects detections in generated real-concept streams
"""

import e2_config
import numpy as np
import matplotlib.pyplot as plt

n_chunks = 400

detector_names = [".3", ".35", ".4", ".45", ".5", ".55", ".6"]
y_values = np.array([.3, .35, .4, .45, .5, .55, .6])
metrics_names = e2_config.metrics_names()

colors = ['#333', '#333']
black_rainbow = ['#333', '#777']

base_width = 12
fig, axx = plt.subplots(2, 3, figsize=(base_width, base_width/1.618))

drf_types = ['cubic', 'nearest']
detector_names = e2_config.e2_clf_names()[:-2]

drifts_n = 5

streams = [
    'australian',
    'banknote',
    'diabetes',
]


for str_id, stream in enumerate(streams):

    #drifts, drf types
    for d_type_id, d_type in enumerate(drf_types):
        drf_arr = np.load('results_ex2_2/gen_drf_arr_str_%s.csv_%s_5drfs.npy' % (stream, d_type))
        real = np.squeeze(np.argwhere(drf_arr[0,0,0,:]==2))
        print(real)

        # detections = np.zeros((10,len(y_values),2,399))
        
        # for s_id, sen in enumerate(y_values):

        #     conf_cpy = np.copy(conf)

        #     minimum_to_detect = int(30*sen)
        #     print(minimum_to_detect)

        #     conf_cpy[conf_cpy<minimum_to_detect]=0
        #     conf_cpy[conf_cpy>=minimum_to_detect]=2

        #     detections[:,s_id,1,:] = conf_cpy[:,1,:]

        #     print(detections[0,s_id,1,:])
        
        # exit()
        # """
        # Plot
        # """

        ax = axx[d_type_id, str_id]
        ax.set_title("%s %s drift" % (stream, d_type), fontsize=12)

        ax.set_xlim(0,n_chunks-1)

        ax.vlines(real, 0, len(detector_names)-1, color='#000', lw=1, ls=":", zorder=1)
        ax.hlines(y_values, 0, 400, color='#333', lw=1, zorder=1)

        for det_id, det_name in enumerate(detector_names):
            drf_cnt=np.ones((drf_arr.shape[3]))

            for rep in range(drf_arr.shape[0]):
                detected = np.argwhere(drf_arr[rep,det_id,1,:]==2)
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
                        label=drf_arr[det_id])


        ax.set_xticks(real)
        ax.set_xticklabels(['%i' % zzz for zzz in (real+1)])
        #ax.grid(ls=':')
        ax.set_yticks(y_values)
        ax.set_yticklabels(["%.2f" % v for v in y_values])
        ax.set_ylim(.25,.65)
        ax.set_xlim(0,400)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if d_type_id == 1:
            ax.set_xlabel('Number of chunks processed')
        if str_id == 0:
            ax.set_ylabel('Detector sensitivity')

        fig.subplots_adjust(top=0.93)
        plt.tight_layout()
        plt.savefig("pub_figures/e3_sensitivity.eps")
        plt.savefig("figures_ex2/e3_sensitivity.eps")
        plt.savefig("foo.png")
        #exit()
