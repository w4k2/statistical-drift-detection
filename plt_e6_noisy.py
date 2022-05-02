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

y_flip = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
n_redundant = [0,3,6,9,12]

res = np.load('results_ex6/drf_arr_res.npy') 
print(res.shape) #y_flip, informative, reps, 2, 49

fig, ax = plt.subplots(6,5,figsize=(13,11))

for y_f_id, y_f in enumerate(y_flip):
    for n_in_id, n_in in enumerate(n_redundant):

        dets = res[y_f_id, n_in_id, :, 1]

        aa = ax[y_f_id, n_in_id]
        aa.set_title('red: %.2f, flip: %.2f' % (n_in/15, y_f))

        aa.imshow(dets,
                vmin=0, vmax=2,
                cmap='binary',
                origin='lower',
                interpolation='none',
                aspect=2)

plt.tight_layout()
plt.savefig('foo.png')