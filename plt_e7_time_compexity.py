"""
Experiment 7 - measure time complexity of method
"""
import numpy as np
import matplotlib.pyplot as plt

chunk_sizes = [50,100,150,200,250]
features=[10,20,30,40,50]
subspace_sizes = [1,2]

res = np.load('results_ex7/time.npy')
#replications, chunk_sizes, features, subspace_sizes

fig, ax = plt.subplots(1,2,figsize=(10,6))

ls=[':', "-"]
colors=['tomato', 'cornflowerblue', 'forestgreen', 'grey', 'gold']

for ch_s_id, ch_s in enumerate(chunk_sizes):
    for sub_size_id, sub_size in enumerate(subspace_sizes):
        
        r = res[:,ch_s_id, :, sub_size_id]
        ax[0].plot(features, np.mean(r, axis=0), ls=ls[sub_size_id], color=colors[ch_s_id], label='chunk size %i, subspace_size %i' % (ch_s, sub_size))
        ax[0].scatter(features, np.mean(r, axis=0), color=colors[ch_s_id])
ax[0].set_xlabel('features')
ax[0].set_ylabel('time [s]')
ax[0].legend()

for feat_id, feat in enumerate(features):
    for sub_size_id, sub_size in enumerate(subspace_sizes):
        
        r = res[:,:, feat_id, sub_size_id]
        ax[1].plot(chunk_sizes, np.mean(r, axis=0), ls=ls[sub_size_id], color=colors[feat_id], label='features %i, subspace_size %i' % (feat, sub_size))
        ax[1].scatter(chunk_sizes, np.mean(r, axis=0), color=colors[feat_id])
ax[1].set_xlabel('chunk size')
ax[1].legend()
               

plt.tight_layout()
plt.savefig('foo.png')