"""
Plot - how does chunk size affect method's performance
"""

import strlearn as sl
import numpy as np
import matplotlib.pyplot as plt

res = np.load('results_ex8/drf_arr_res.npy')

print(res.shape) # chunksizes x reps x sensitivity x 2 x chunks-1

chunk_sizes = np.linspace(100,1000,10).astype('int')

fig, ax = plt.subplots(len(chunk_sizes), 3, figsize=(10,10), sharex=True)

for sens_id, sensitivity in enumerate(['0.15', '0.25', '0.35']):
    for ch_s_id, ch_s in enumerate(chunk_sizes):
        drf_arr = res[ch_s_id,:,sens_id].reshape(10,1,2,49)

        drf_cnt=np.ones((drf_arr.shape[3]))
        czytotu = np.where(drf_arr[0,0,0,:] == 2)[0]
        print(czytotu)

        zzz = drf_arr[:,:,1,:]
        zzz[zzz==1]=0 # removing warnings

        zzz = np.swapaxes(zzz, 0,1)
        zzz = np.reshape(zzz, (-1, 49))

        print('ZZZ', zzz.shape)

        aa = ax[ch_s_id, sens_id]

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

        if ch_s_id==len(chunk_sizes)-1:
            ax[-1,sens_id].set_xlabel('chunk id')
        
        if ch_s_id==0:
            ax[0,sens_id].set_title('sen: %s' % sensitivity)
        if sens_id==0:
            aa.set_ylabel('ch. s.: %i' % ch_s)

        
        aa.set_yticks([])

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig("pub_figures/exp8.eps")
plt.savefig("pub_figures/exp8.png")




