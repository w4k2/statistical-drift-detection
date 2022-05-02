"""
Experiment 7 - measure time complexity of method
"""

import time
import strlearn as sl
import numpy as np
from sklearn.base import clone
import e2_config
from tqdm import tqdm
from methods import Meta, SDDE
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

np.random.seed(13654)

replications = e2_config.e2_replications()
random_states = np.random.randint(0, 10000, replications)

static_params = e2_config.e2_static2()

chunk_sizes = [50,100,150,200,250]
static_params['n_chunks']=50

metrics = e2_config.metrics()
features=[10,20,30,40,50]

subspace_sizes = [1,2]

t = 2*replications*len(features)*len(chunk_sizes)
pbar = tqdm(total=t)

res_time = np.zeros((replications, len(chunk_sizes), len(features), len(subspace_sizes)))

for feat_id, feat in enumerate(features):
    for ch_s_id, ch_s in enumerate(chunk_sizes):
        for sub_size_id, sub_size in enumerate(subspace_sizes):
            for rep in range(replications):
                print("features: %i, chunk size: %i, replication: %i" % (feat, ch_s, rep))

                detectors = [
                    Meta(detector = SDDE(n_detectors= feat, sensitivity=.35, subspace_size=sub_size), base_clf = GaussianNB()),
                ]

                static_params['chunk_size']=ch_s
                
                config = {
                    **static_params,
                    'n_features': feat, 
                    'n_informative': feat,
                    'n_drifts': 2,
                    'random_state': random_states[rep],
                    }

                stream = sl.streams.StreamGenerator(**config)

                eval = sl.evaluators.TestThenTrain(metrics=metrics)

                start = time.time()
                eval.process(stream, detectors)
                elapsed = time.time() - start

                res_time[rep, ch_s_id, feat_id, sub_size_id] = elapsed
                print(elapsed)

                pbar.update(1)

np.save('results_ex7/time', res_time)

pbar.close()