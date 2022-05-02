"""
Experiment 6 - evaluation on noisy streams
"""

import strlearn as sl
import numpy as np
from sklearn.base import clone
import e2_config
from tqdm import tqdm
from methods import Meta, SDDE
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

y_flip = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15]
n_informative = [15,13,11,9,7,5]

res = np.load('results_ex6/drf_arr_res.npy') 
print(res.shape) #y_flip, informative, reps, 2, 49

fig, ax = plt.subplots(6,6,figsize=(13,11))

for y_f_id, y_f in enumerate(y_flip):
    for n_in_id, n_in in enumerate(n_informative):

        dets = res[y_f_id, n_in_id, :, 1]

        aa = ax[y_f_id, n_in_id]
        aa.set_title('inf: %.2f, flip: %.2f' % (n_in/15, y_f))

        aa.imshow(dets,
                vmin=0, vmax=2,
                cmap='binary',
                origin='lower',
                interpolation='none',
                aspect=2)

plt.tight_layout()
plt.savefig('foo.png')