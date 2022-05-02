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

def find_real_drift(chunks, drifts):
    interval = round(chunks/drifts)
    arr = np.zeros((chunks))
    idx = [interval*(i+.5) for i in range(drifts)]
    for i in idx:
        arr[int(i)]=2
    return arr[1:]

np.random.seed(13654)

replications = e2_config.e2_replications()
random_states = np.random.randint(0, 10000, replications)

static_params = e2_config.e2_static2()
static_params['n_chunks']=50
static_params['chunk_size']=250

metrics = e2_config.metrics()

y_flip = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15]
n_informative = [15,13,11,9,7,5]
features=15

t = replications*len(y_flip)*len(n_informative)
pbar = tqdm(total=t)

results_clf = np.zeros((len(y_flip), len(n_informative), replications, static_params['n_chunks']-1))
results_drf_arrs = np.zeros((len(y_flip), len(n_informative), replications, 2, static_params['n_chunks']-1))

real_drf = find_real_drift(static_params['n_chunks'], 2)

for flip_id, flip in enumerate(y_flip):
    for inf_id, n_inf in enumerate(n_informative):
        for replication in range(replications):

            detectors = [
                Meta(detector = SDDE(n_detectors= features, sensitivity=.35), base_clf = GaussianNB()),
            ]
            
            config = {
                **static_params,
                'n_features': features, 
                'n_informative': n_inf,
                'n_drifts': 2,
                'random_state': random_states[replication]
                }

            stream = sl.streams.StreamGenerator(**config)

            print("replication: %i, informative: %i, yflip: %f" % (replication, n_inf, flip))

            eval = sl.evaluators.TestThenTrain(metrics=metrics)
            eval.process(stream, detectors)

            scores = eval.scores
            results_clf[flip_id, inf_id, replication] = scores[:,:,0]

            results_drf_arrs[flip_id, inf_id, replication, 0] = real_drf
            results_drf_arrs[flip_id, inf_id, replication, 1] = np.array(detectors[0].detector.drift)

            pbar.update(1)

np.save('results_ex6/clf_res', results_clf)
np.save('results_ex6/drf_arr_res', results_drf_arrs)

pbar.close()