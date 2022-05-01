"""
Plot - How the metod performs for highly multidimentional streams
"""

import numpy as np
import e2_config
from tqdm import tqdm
import matplotlib.pyplot as plt

res = np.load('results_ex5/drf_arr_res.npy')
n_features = np.round(np.linspace(2,100,10)).astype('int')

fig, ax = plt.subplots(len(n_features), 6, figsize=(13,15), sharex=True)

for sens_id, sensitivity in enumerate(['.05', '0.1', '0.15', '.2', '0.25', '.3']):
    for n_f_id, n_f in enumerate(n_features):
        drf_arr = res[n_f_id,:,sens_id].reshape(10,1,2,49)
        print(drf_arr.shape)

        drf_cnt=np.ones((drf_arr.shape[3]))
        czytotu = np.where(drf_arr[0,0,0,:] == 2)[0]
        print(czytotu)

        zzz = drf_arr[:,:,1,:]
        zzz[zzz==1]=0 # removing warnings

        zzz = np.swapaxes(zzz, 0,1)
        zzz = np.reshape(zzz, (-1, 49))

        print('ZZZ', zzz.shape)

        aa = ax[n_f_id, sens_id]

        aa.spines['left'].set_visible(False)
        aa.spines['right'].set_visible(False)

        aa.set_xticks(czytotu)
        aa.set_xticklabels(czytotu+1, fontsize=8)
        aa.grid(ls=":", axis='x', lw=1, color='black')

        aa.imshow(zzz,
                    vmin=0, vmax=2,
                    cmap='binary',
                    origin='lower',
                    interpolation='none',
                    aspect=2)

        if n_f_id==len(n_features)-1:
            ax[-1,sens_id].set_xlabel('chunk id', fontsize=8)
        
        if n_f_id==0:
            ax[0,sens_id].set_title(sensitivity)
        
        aa.set_yticks([])
        aa.set_ylabel(n_f)

    plt.tight_layout()
    plt.savefig('foo.png')


