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
n_drifts = {3: { 'n_drifts': 3}}

drf_types = e2_config.e2_drift_types()
recurring = {'not-recurring': {}}

sensitivity = ['.2', '.3', '.4', '.5']
categories = ['num', 'bin', 'cat']

fig, ax = plt.subplots(len(drf_types), 3, figsize=(10, 7),
                               sharey=True)

for drf_id, drf_type in enumerate(drf_types):
    for c_id, category in enumerate(categories):
        ax[-1,c_id].set_xlabel('chunk id')

        res_arr = np.load('results_ex4/drf_arr_%s_%s.npy' % (drf_type, category))
        print(res_arr.shape)
        # print(res_arr.shape)
        
        """
        Plot
        """
        aa = ax[drf_id, c_id]

        gt = np.where(res_arr[0,0,0,:] == 2)[0]
        detections = res_arr[:,:,1,:]
        # print(gt.shape)
        # print(detections.shape)
        # print(drf_type, category)
        # print(detections[0])
        # exit()

        aa.spines['left'].set_visible(False)
        aa.spines['right'].set_visible(False)

        aa.set_xticks(gt)
        aa.set_xticklabels(gt+1, fontsize=8)
        aa.grid(ls=":", axis='x', lw=0.5, color='black')

        if drf_id==0:
            aa.set_title(category)

        if c_id==0:
            aa.set_ylabel(drf_type, fontsize=10)
            aa.set_yticks([5,15,25,35])
            aa.set_yticklabels(sensitivity)
        
        aa.hlines([10,20,30,40], 0, 199, color='black', lw=0.5)

        print(detections.shape)
        det_reshaped=np.zeros((40,199))
        for i in range(4):
            s=i*10
            e=(i+1)*10
            det_reshaped[s:e] = detections[:,i]

        aa.imshow(det_reshaped,
                        cmap='binary',
                        vmin=0, vmax=2,
                        origin='lower',
                        interpolation='none',
                        aspect=2)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig("pub_figures/exp4.eps")
plt.savefig("pub_figures/exp4.png")