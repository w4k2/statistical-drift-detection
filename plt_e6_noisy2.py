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
        aa2 = aa.twinx()
        aa.set_title('red: %.2f, flip: %.2f' % (n_red/15, y_f))

        for dets_rep in dets:
            
            d = np.argwhere(dets_rep==2)
            aa.vlines(d, 0.5, 1, color='tomato', alpha=0.3)
        
        clf = res_clf[y_f_id, n_red_id]
        clf_mean = np.mean(clf, axis=0)
        aa2.plot(clf_mean)
        aa2.set_ylim(0.5,1)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig("pub_figures/exp6_drf_clf.eps")
plt.savefig("pub_figures/exp6_drf_clf.png")
