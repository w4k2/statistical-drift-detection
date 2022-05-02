"""
Experiment 6 - evaluation on noisy streams
"""

import numpy as np
import matplotlib.pyplot as plt
from methods import dderror

y_flip = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
y_flip_labels = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']
n_redundant = [0,3,6,9,12]
n_redundant_percent = np.array(n_redundant)/15
n_redundant_percent_labels  = ['%.2f' % i for i in n_redundant_percent]

res = np.load('results_ex6/drf_arr_res.npy') 
res_clf = np.load('results_ex6/clf_res.npy') 
print(res.shape) #y_flip, informative, reps, 2, 49
print(res_clf.shape) #y_flip, informative, reps, 49

fig, ax = plt.subplots(6,5,figsize=(13,11))

for y_f_id, y_f in enumerate(y_flip):
    for n_red_id, n_red in enumerate(n_redundant):

        dets = res[y_f_id, n_red_id, :, 1]

        aa = ax[y_f_id, n_red_id]
        aa.set_title('red: %.2f, flip: %.2f' % (n_red/15, y_f))

        aa.imshow(dets,
                vmin=0, vmax=2,
                cmap='binary',
                origin='lower',
                interpolation='none',
                aspect=2)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig("pub_figures/exp6_drf.eps")
plt.savefig("pub_figures/exp6_drf.png")

plt.clf()
fig, ax = plt.subplots(6,5,figsize=(13,11))

for y_f_id, y_f in enumerate(y_flip):
    for n_red_id, n_red in enumerate(n_redundant):

        clf = res_clf[y_f_id, n_red_id]
        clf_mean = np.mean(clf, axis=0)

        aa = ax[y_f_id, n_red_id]
        aa.set_title('red: %.2f, flip: %.2f' % (n_red/15, y_f))

        aa.plot(clf_mean)

plt.tight_layout()
plt.savefig('foo2.png')
plt.savefig("pub_figures/exp6_clf.eps")
plt.savefig("pub_figures/exp6_clf.png")

"""
Heatmaps
"""

metric_names = ['D1', 'D2', 'R', 'accuracy']

plt.clf()
fig, ax = plt.subplots(2,2,figsize=(9,10))
ax = ax.ravel()
heatmaps = np.zeros((4, 10, len(y_flip), len(n_redundant)))

for y_f_id, y_f in enumerate(y_flip):
    for n_red_id, n_red in enumerate(n_redundant):
        for rep in range(10):
            real_drf = np.argwhere(res[y_f_id, n_red_id, rep, 0]==2).flatten()
            det_drf = np.argwhere(res[y_f_id, n_red_id, rep, 1]==2).flatten()
            err = dderror(real_drf, det_drf, res.shape[4])
            heatmaps[:3, rep, y_f_id, n_red_id] = err

            clf_err = np.mean(res_clf[y_f_id, n_red_id, rep])
            heatmaps[3, rep, y_f_id, n_red_id] = clf_err

heatmaps_mean = np.mean(heatmaps, axis=1)

for i in range(4):
    ax[i].set_title(metric_names[i])
    ax[i].imshow(heatmaps_mean[i],
                cmap='binary',
                interpolation='none',)

    ax[i].set_yticks(range(len(y_flip_labels)))
    ax[i].set_yticklabels(y_flip_labels)
    ax[i].set_ylabel('y_flip')

    ax[i].set_xticks(range(len(n_redundant_percent_labels)))
    ax[i].set_xticklabels(n_redundant_percent_labels)
    ax[i].set_xlabel('redundant %')


plt.tight_layout()
plt.savefig('foo3.png')
plt.savefig("pub_figures/exp6_heat.eps")
plt.savefig("pub_figures/exp6_heat.png")