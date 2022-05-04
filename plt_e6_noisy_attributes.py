"""
Experiment 6 - evaluation on noisy streams (+ attribute noise)
"""

import numpy as np
import matplotlib.pyplot as plt

y_flip = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
attr_noise = np.linspace(0,1,5)

res = np.load('results_ex6/drf_arr_res2.npy') 
res_clf = np.load('results_ex6/clf_res2.npy') 
print(res.shape) #y_flip, informative, reps, 2, 49
print(res_clf.shape) #y_flip, informative, reps, 49

fig, ax = plt.subplots(6,5,figsize=(13,11), sharex=True, sharey=True)

for y_f_id, y_f in enumerate(y_flip):
    for attr_n_id, attr_n in enumerate(attr_noise):
        if attr_n_id==0:
            ax[y_f_id,0].set_ylabel('flip: %.2f' % y_f) 
        if y_f_id==0:
            ax[0,attr_n_id].set_title('magnitude: %.2f' % attr_n) 
        if y_f_id==5:
            ax[y_f_id,attr_n_id].set_xlabel('chunk id') 



        dets = res[y_f_id, attr_n_id, :, 1]

        aa = ax[y_f_id, attr_n_id]
        aa.spines['top'].set_visible(False)
        aa.spines['right'].set_visible(False)
        # aa.set_title('noise: %.2f, flip: %.2f' % (attr_n, y_f))

        for dets_rep in dets:
            
            d = np.argwhere(dets_rep==2)
            aa.vlines(d, 0.5, 1, color='tomato', alpha=0.3)
        
        clf = res_clf[y_f_id, attr_n_id]
        clf_mean = np.mean(clf, axis=0)
        aa.plot(clf_mean)
        aa.set_ylim(0.5,1)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig("pub_figures/exp6_drf_clf_attr.eps")
plt.savefig("pub_figures/exp6_drf_clf_attr.png")
