import e2_config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import os

drf_types = ['cubic', 'nearest']
detector_names = e2_config.e2_clf_names()[:-2]

drifts_n = [3,5,7]
features_n = 15

streams = [
    'australian',
    'banknote',
    'diabetes',
    'wisconsin',
    'yeast-2_vs_8'
]

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "lemonchiffon", "black"])

for str_id, stream in enumerate(streams):

    plt.close()
    fig, ax = plt.subplots(len(drifts_n), len(drf_types), figsize=(14, 5), sharey=True)
    fig.suptitle("%s" % (stream), fontsize=14)

    #drifts, drf types
    for d_id, d in enumerate(drifts_n):
        for d_type_id, d_type in enumerate(drf_types):

            res_arr = np.load('results_ex2_real/drf_arr_str_%s.csv_%s_%idrfs.npy' %(stream, d_type, d))
            print(res_arr.shape) # reps, detectors, chunks-1

            """
            Plot
            """
            czytotu = np.where(res_arr[0,0,0,:] == 2)[0]

            zzz = res_arr[:,:,1,:]
            zzz[zzz==1]=0 # warningi przeszkadzaja

            mask = zzz==0
            mask[:,[0,1,2,4,5],:]=False
            zzz[mask]=1

            zzz = np.swapaxes(zzz, 0, 1)
            zzz = np.reshape(zzz, (-1, 399))
            print(zzz.shape)
            # exit()


            zzz = zzz[:-20, :]

            # print(zzz, zzz.shape)

            ax[d_id, d_type_id].spines['top'].set_visible(False)
            ax[d_id, d_type_id].spines['bottom'].set_visible(False)
            ax[d_id, d_type_id].spines['left'].set_visible(False)
            ax[d_id, d_type_id].spines['right'].set_visible(False)

            aa = ax[d_id, d_type_id]
            aa.set_xticks(czytotu)
            aa.set_xticklabels(czytotu+1, fontsize=10)
            aa.grid(ls=":", axis='x', lw=1, color='black')

            aa.hlines([0,10,20,30,40], 0, 400, color='black', lw=.5)

            ax[d_id,d_type_id].imshow(zzz,
                                    cmap=cmap,
                                    origin='lower',
                                    interpolation='none',
                                    aspect=2)

            aa.set_ylim(0, 40)

            if d_id == 0:
                aa.set_title('%s drift' % d_type, fontsize=12)

            if d_type_id == 0:
                aa.set_ylabel('%i drifts' % d, fontsize=12)

            if d_id == 3:
                aa.set_xlabel('number of chunks processed', fontsize=10)


            ax[d_id,d_type_id].set_yticks([5,15,25,35])
            ax[d_id,d_type_id].set_yticklabels("%s" % d
                                            for i, d in enumerate(detector_names))

    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    plt.savefig("figures_ex2_real/real_pix_%s.png" % (stream))
    plt.savefig("figures_ex2_real/real_pix_%s.eps" % (stream))
    plt.savefig('foo.png')
    #exit()
