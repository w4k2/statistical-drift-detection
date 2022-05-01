"""
Experiment 4 - evaluation on hybrid attributes
stream #1 -> numeric only (15)
stream #2 -> 8 binary, 7 numeric
stream #3 -> 8 categorical(0-3 values), 7 numeric
"""

import strlearn as sl
import numpy as np
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
# static_params['n_chunks']=50
# static_params['chunk_size']=50

drf_types = e2_config.e2_drift_types()

metrics = e2_config.metrics()
sensitivity = [.2, .3, .4, .5]
categories = ['num', 'bin', 'cat']

print(len(drf_types), replications)
t = 3*len(sensitivity)*len(drf_types)*replications
pbar = tqdm(total=t)

drifs=5
features=15
real_drf = find_real_drift(static_params['n_chunks'], drifs)


for category in categories:
    for drf_type in drf_types:

        results_clf = np.zeros((replications, len(sensitivity), static_params['n_chunks']-1))
        results_drf_arrs = np.zeros((replications, len(sensitivity), 2, static_params['n_chunks']-1))

        for sen_id, sen in enumerate(sensitivity):
            for replication in range(replications):

                detector= [
                    Meta(detector = SDDE(n_detectors= features, sensitivity=sen), base_clf = GaussianNB()),
                ]

                str_name = "%s_%s" % (drf_type,category)
                
                config = {
                    **static_params,
                    **drf_types[drf_type],
                    'n_features': features, 
                    'n_informative': features,
                    'n_drifts': drifs,
                    'random_state': random_states[replication]
                            }
                
                #original numeric
                stream = sl.streams.StreamGenerator(**config)
                
                if category=='bin':
                    # binary
                    stream = binarize(stream, 8)

                if category=='cat':
                    # categorical
                    stream = categorize(stream, 8)

                print("replication: %i, stream: %s, sensitivity: %s" % (replication, str_name, sen))

                eval = sl.evaluators.TestThenTrain(metrics=metrics)
                eval.process(stream, detector)

                scores = eval.scores
                results_clf[replication] = scores[:,:,0]

                results_drf_arrs[replication, sen_id, 0] = real_drf
                results_drf_arrs[replication, sen_id, 1] = np.array(detector[0].detector.drift)

                pbar.update(1)

        np.save('results_ex4/clf_%s' % str_name, results_clf)
        np.save('results_ex4/drf_arr_%s' % str_name, results_drf_arrs)
pbar.close()