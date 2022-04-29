"""
Experiment 4 - evaluation on hybrid attributes
stream #1 -> numeric only (15)
stream #2 -> 5 binary, 10 numeric
stream #3 -> 5 categorical(0-3 values), 10 numeric
"""

import strlearn as sl
import numpy as np
from sklearn.base import clone
import e2_config
from tqdm import tqdm
from methods import Meta, SDDE
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

def binarize(stream, num_binary):
    
    X_np, y_np= stream._make_classification()

    for idx in range(num_binary):
        feature_values = np.copy(X_np[:,idx])
        th = np.mean(feature_values)

        X_np[:,idx][feature_values>th]=1
        X_np[:,idx][feature_values<=th]=0

    
    file = np.concatenate([X_np, y_np[:,np.newaxis]], axis=1)
    np.save('stream_generated.npy', file)
    np.savetxt('stream_generated.txt', file)

    s = sl.streams.NPYParser("stream_generated.npy", chunk_size=static_params['chunk_size'], n_chunks=static_params['n_chunks'])
    return s

def categorize(stream, num_categorical):
    
    X_np, y_np= stream._make_classification()

    for idx in range(num_categorical):
        feature_values = np.copy(X_np[:,idx])
        th_mid = np.mean(feature_values)
        th_1 = np.mean(feature_values[feature_values<th_mid])
        th_3 = np.mean(feature_values[feature_values>th_mid])

        X_np[:,idx][feature_values<th_1]=0
        X_np[:,idx][feature_values>=th_1]=1
        X_np[:,idx][feature_values>=th_mid]=2
        X_np[:,idx][feature_values>=th_3]=3

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

n_features = {15: { 'n_features': 15, 'n_informative': 15}}
n_drifts = {5: { 'n_drifts': 5}}

drf_types = e2_config.e2_drift_types()
recurring = {'not-recurring': {}}

metrics = e2_config.metrics()

print(len(n_drifts), len(n_features), len(drf_types) ,len(recurring), replications)
t = len(n_drifts)*len(n_features)*len(drf_types)*len(recurring)*replications
pbar = tqdm(total=t)

for n_f in n_features:
    # base_detectors = [Meta(detector = SDDE(n_detectors= n_f, sensitivity=.35), base_clf = GaussianNB())] #numeric
    base_detectors = [Meta(detector = SDDE(n_detectors= n_f, sensitivity=.45), base_clf = GaussianNB())] # binary
    # base_detectors = [Meta(detector = SDDE(n_detectors= n_f, sensitivity=.5), base_clf = GaussianNB())] # categoric

    for n_d in n_drifts:
        real_drf = find_real_drift(static_params['n_chunks'], n_d)

        for rec in recurring:
            for drf_type in drf_types:

                results_clf = np.zeros((replications, len(base_detectors), static_params['n_chunks']-1))
                results_drf_arrs = np.zeros((replications, len(base_detectors), 2, static_params['n_chunks']-1))
                # replications x detectors x (real_drf, detected_drf) x chunks

                for replication in range(replications):

                    detectors=[]
                    for det_id in range(len(base_detectors)):
                        detectors.append(clone(base_detectors[det_id]))

                    str_name = "%ifeat_%idrifts_%s_%s" % (n_f,n_d,drf_type,rec)
                    
                    config = {
                        **static_params,
                        **drf_types[drf_type],
                        **recurring[rec],
                        **n_features[n_f],
                        **n_drifts[n_d],
                        'random_state': random_states[replication]
                                }
                    #original numeric
                    stream = sl.streams.StreamGenerator(**config)

                    # binary
                    # stream = binarize(stream, 5)

                    # categoric
                    stream = categorize(stream, 5)

                    print("replication: %i, stream: %s" % (replication, str_name))

                    eval = sl.evaluators.TestThenTrain(metrics=metrics)
                    eval.process(stream, detectors)

                    scores = eval.scores
                    results_clf[replication] = scores[:,:,0]

                    for det_id in range(len(detectors)):
                        results_drf_arrs[replication, det_id, 0] = real_drf
                        results_drf_arrs[replication, det_id, 1] = np.array(detectors[det_id].detector.drift)

                    pbar.update(1)

                # np.save('results_ex4/clf_%s' % str_name, results_clf)
                # np.save('results_ex4/drf_arr_%s' % str_name, results_drf_arrs)
                
                # np.save('results_ex4/clf_%s_bin' % str_name, results_clf)
                # np.save('results_ex4/drf_arr_%s_bin' % str_name, results_drf_arrs)

                np.save('results_ex4/clf_%s_cat' % str_name, results_clf)
                np.save('results_ex4/drf_arr_%s_cat' % str_name, results_drf_arrs)

pbar.close()