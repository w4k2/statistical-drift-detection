"""
Plot - How the metod performs for highly multidimentional streams
"""

import numpy as np
import e2_config
from tqdm import tqdm
import matplotlib.pyplot as plt

res = np.load('results_ex5/drf_arr_res.npy')
n_features = np.round(np.linspace(10,100,10)).astype('int')

fig, ax = plt.subplots(len(n_features), 5, figsize=(10,10), sharex=True)

for sens_id, sensitivity in enumerate(['0.1', '0.15', '0.2', '0.25', '0.3']):
    for n_f_id, n_f in enumerate(n_features):
        drf_arr = res[n_f_id,:,sens_id].reshape(10,1,2,49)

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
            ax[-1,sens_id].set_xlabel('chunk id')
        
        if n_f_id==0:
            ax[0,sens_id].set_title('sen: %s' % sensitivity)
        if sens_id==0:
            aa.set_ylabel('dim: %i' % n_f)

        
        aa.set_yticks([])

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig("pub_figures/exp5.eps")
plt.savefig("pub_figures/exp5.png")


