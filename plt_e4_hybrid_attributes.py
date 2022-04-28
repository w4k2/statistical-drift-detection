"""
Plot - How the metod performs for hybrid attributes
"""
import strlearn as sl
import numpy as np
from sklearn.base import clone
import e2_config
from tqdm import tqdm
from methods import Meta, SDDE
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import matplotlib.colors

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "lemonchiffon", "black"])
title = ['numeric', 'numeric/binary', 'numeric/categorical']

fig, ax = plt.subplots(3, 1, figsize=(7,3))

for i in range(3):
    if i==0:
        drf_arr = np.load('results_ex4/drf_arr_num.npy')
    if i==1:
        drf_arr = np.load('results_ex4/drf_arr_bin.npy')
    if i==2:
        drf_arr = np.load('results_ex4/drf_arr_cat.npy')

    drf_cnt=np.ones((drf_arr.shape[3]))

    czytotu = np.where(drf_arr[0,0,0,:] == 2)[0]
    print(czytotu)

    zzz = drf_arr[:,:,1,:]
    zzz[zzz==1]=0 # removing warnings

    zzz = np.swapaxes(zzz, 0,1)
    zzz = np.reshape(zzz, (-1, 199))

    print('ZZZ', zzz.shape)

    # ax[i].spines['top'].set_visible(False)
    # ax[i].spines['bottom'].set_visible(False)
    ax[i].spines['left'].set_visible(False)
    ax[i].spines['right'].set_visible(False)

    aa = ax[i]
    aa.set_xticks(czytotu)
    aa.set_xticklabels(czytotu+1, fontsize=8)
    aa.grid(ls=":", axis='x', lw=1, color='black')

    ax[i].imshow(zzz,
                vmin=0, vmax=2,
                cmap=cmap,
                origin='lower',
                interpolation='none',
                aspect=2)

    aa.set_title(title[i], fontsize=10)
    if i==2:
        aa.set_xlabel('number of chunks processed', fontsize=8)
    
    aa.set_ylabel('replication', fontsize=8)

plt.tight_layout()
plt.savefig('foo.png')


