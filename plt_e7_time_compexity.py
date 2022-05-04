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
res = res[:,:,:,0]

fig, ax = plt.subplots(1,2,figsize=(10,6), sharey=True)

colors=['#aaa', '#888', '#666', '#444', '#222']

for ch_s_id, ch_s in enumerate(chunk_sizes):
    r = res[:,ch_s_id]
    ax[0].plot(features, np.mean(r, axis=0), color=colors[ch_s_id], label='chunk size %i' % (ch_s))
    ax[0].scatter(features, np.mean(r, axis=0), color=colors[ch_s_id],marker='+', s=70)


for feat_id, feat in enumerate(features):    
    r = res[:,:, feat_id]
    ax[1].plot(chunk_sizes, np.mean(r, axis=0), color=colors[feat_id], label='features %i' % (feat))
    ax[1].scatter(chunk_sizes, np.mean(r, axis=0), color=colors[feat_id], marker='+', s=70)

for i in range(2):
    ax[i].set_xlabel(['features', 'chunk size'][i])
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].grid(ls=':')
    ax[i].set_xticks([features,chunk_sizes][i])
    ax[i].legend(frameon=False, loc=2)
    ax[i].set_ylim(0,30)
ax[0].set_ylabel('time [s]')

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig("pub_figures/exp7.eps")
plt.savefig("pub_figures/exp7.png")
