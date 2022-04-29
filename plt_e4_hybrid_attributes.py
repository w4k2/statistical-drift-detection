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

replications = e2_config.e2_replications()
random_states = np.random.randint(0, 10000, replications)

static_params = e2_config.e2_static2()

n_features = {15: { 'n_features': 15, 'n_informative': 15}}
n_drifts = {5: { 'n_drifts': 5}}

drf_types = e2_config.e2_drift_types()
recurring = {'not-recurring': {}}

fig, ax = plt.subplots(len(drf_types), 3, figsize=(10, 3),
                               sharey=True)

for drf_id, drf_type in enumerate(drf_types):

    for c_id, category in enumerate(['numeric', 'binary', 'categorical']):

        if c_id==0:
            res_arr = np.load('results_ex4/drf_arr_%ifeat_%idrifts_%s_%s.npy' %(15, 5, drf_type, 'not-recurring'))

        if c_id==1:
            res_arr = np.load('results_ex4/drf_arr_%ifeat_%idrifts_%s_%s_bin.npy' %(15, 5, drf_type, 'not-recurring'))
        
        if c_id==2:
            res_arr = np.load('results_ex4/drf_arr_%ifeat_%idrifts_%s_%s_cat.npy' %(15, 5, drf_type, 'not-recurring'))

        """
        Plot
        """
        drf_cnt=np.ones((res_arr.shape[3]))

        czytotu = np.where(res_arr[0,0,0,:] == 2)[0]
        print(czytotu)

        zzz = res_arr[:,:,1,:]
        zzz[zzz==1]=0 # removing warnings

        ax[drf_id, c_id].spines['left'].set_visible(False)
        ax[drf_id, c_id].spines['right'].set_visible(False)

        aa = ax[drf_id, c_id]
        aa.set_xticks(czytotu)
        aa.set_xticklabels(czytotu+1, fontsize=8)
        aa.grid(ls=":", axis='x', lw=1, color='black')

        if drf_id==0:
            aa.set_title(category)
        if c_id==0:
            aa.set_ylabel(drf_type, fontsize=10)
            aa.set_yticks([])

        if drf_id==2:
            aa.set_xlabel('chunk id')

        zzz = zzz.reshape(10,199)

        ax[drf_id, c_id].imshow(zzz,
                        cmap='binary',
                        vmin=0, vmax=2,
                        origin='lower',
                        interpolation='none',
                        aspect=2)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig("pub_figures/exp4.eps")
plt.savefig("pub_figures/exp4.png")