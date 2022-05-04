"""
Experiment 8 - how does chunk size affect method's performance
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

n_features = 15

chunk_sizes = [405]
n_drifts = {2: { 'n_drifts': 2}}

drf_types = {'sudden': {}}
recurring = {'not-recurring': {}}

metrics = e2_config.metrics()
base_detectors_num = 3

print(len(n_drifts), len(chunk_sizes), len(drf_types) ,len(recurring), replications)
t = len(n_drifts)*len(chunk_sizes)*len(drf_types)*len(recurring)*replications
pbar = tqdm(total=t)

results_clf = np.zeros((len(chunk_sizes), replications, base_detectors_num, static_params['n_chunks']-1))
results_drf_arrs = np.zeros((len(chunk_sizes), replications, base_detectors_num, 2, static_params['n_chunks']-1))

for ch_s_id, ch_s in enumerate(chunk_sizes):

    static_params['chunk_size']=ch_s

    base_detectors = [
        Meta(detector = SDDE(n_detectors= n_features, sensitivity=.15), base_clf = GaussianNB()),
        Meta(detector = SDDE(n_detectors= n_features, sensitivity=.25), base_clf = GaussianNB()),
        Meta(detector = SDDE(n_detectors= n_features, sensitivity=.35), base_clf = GaussianNB()),
        ]
    
    for n_d in n_drifts:
        real_drf = find_real_drift(static_params['n_chunks'], n_d)
        for rec in recurring:
            for drf_type in drf_types:
                for replication in range(replications):

                    detectors=[]
                    for det_id in range(len(base_detectors)):
                        detectors.append(clone(base_detectors[det_id]))
                    
                    config = {
                        **static_params,
                        **drf_types[drf_type],
                        **recurring[rec],
                        'n_features': n_features, 
                        'n_informative': n_features,
                        **n_drifts[n_d],
                        'random_state': random_states[replication]
                                }
                    stream = sl.streams.StreamGenerator(**config)

                    print("replication: %i, chunk size: %s" % (replication, ch_s))

                    eval = sl.evaluators.TestThenTrain(metrics=metrics)
                    eval.process(stream, detectors)

                    scores = eval.scores
                    results_clf[ch_s_id, replication] = scores[:,:,0]

                    for det_id in range(len(detectors)):
                        results_drf_arrs[ch_s_id, replication, det_id, 0] = real_drf
                        results_drf_arrs[ch_s_id, replication, det_id, 1] = np.array(detectors[det_id].detector.drift)

                    pbar.update(1)

                np.save('results_ex8/clf_res', results_clf)
                np.save('results_ex8/drf_arr_res', results_drf_arrs)

pbar.close()