"""
Experiment 6 - evaluation on noisy streams (noise in labels and in attributes)
"""

import strlearn as sl
import numpy as np
from sklearn.base import clone
import e2_config
from tqdm import tqdm
from methods import Meta, SDDE
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

def noise_attributes(stream, scale):
    
    X_np, y_np= stream._make_classification()
    # print(X_np.shape)

    features_std = np.std(X_np, axis=0)
    # print(features_std.shape)

    noise = np.random.normal(0,scale*features_std,X_np.shape)
    # print(noise)

    X_np = X_np+noise
    
    file = np.concatenate([X_np, y_np[:,np.newaxis]], axis=1)
    np.save('stream_generated.npy', file)
    np.savetxt('stream_generated.txt', file)

    s = sl.streams.NPYParser("stream_generated.npy", chunk_size=static_params['chunk_size'], n_chunks=static_params['n_chunks'])
    return s

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

y_flip = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
attr_noise = np.linspace(0,1,5)
features=15

t = replications*len(y_flip)*len(attr_noise)
pbar = tqdm(total=t)

results_clf = np.zeros((len(y_flip), len(attr_noise), replications, static_params['n_chunks']-1))
results_drf_arrs = np.zeros((len(y_flip), len(attr_noise), replications, 2, static_params['n_chunks']-1))

real_drf = find_real_drift(static_params['n_chunks'], 2)

for flip_id, flip in enumerate(y_flip):
    for attr_n_id, attr_n in enumerate(attr_noise):
        for replication in range(replications):

            detectors = [
                Meta(detector = SDDE(n_detectors= features, sensitivity=.35), base_clf = GaussianNB()),
            ]
            
            config = {
                **static_params,
                'n_features': features, 
                'n_informative': features,
                'n_drifts': 2,
                'random_state': random_states[replication],
                'y_flip':flip
                }

            stream = sl.streams.StreamGenerator(**config)

            stream = noise_attributes(stream, attr_n)

            print("replication: %i, attr_noise: %f, yflip: %f" % (replication, attr_n, flip))

            eval = sl.evaluators.TestThenTrain(metrics=metrics)
            eval.process(stream, detectors)

            scores = eval.scores
            results_clf[flip_id, attr_n_id, replication] = scores[:,:,0]

            results_drf_arrs[flip_id, attr_n_id, replication, 0] = real_drf
            results_drf_arrs[flip_id, attr_n_id, replication, 1] = np.array(detectors[0].detector.drift)

            pbar.update(1)

np.save('results_ex6/clf_res2', results_clf)
np.save('results_ex6/drf_arr_res2', results_drf_arrs)

pbar.close()