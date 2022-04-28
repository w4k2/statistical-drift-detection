"""
Experiment 4 - How the metod performs for hybrid attributes
"""
import strlearn as sl
import numpy as np
from sklearn.base import clone
import e2_config
from tqdm import tqdm
from methods import Meta, SDDE
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

np.random.seed(765)

static_params = e2_config.e2_static2()
static_params['chunk_size'] = 250
static_params['n_chunks'] = 200
drifs = 5

def generate_categoric_streams(random_state):
    strs =[]

    values_per_feature = 3
    binary = 9#43
    numeric = 9
    categoric = int(binary/values_per_feature)

    config = {
        **static_params,
        'n_features': binary+numeric,
        'n_drifts': drifs,
        'random_state': random_state
            }
    stream = sl.streams.StreamGenerator(**config)
    strs.append(stream)

    X_np, y_np= stream._make_classification()

    feature_idx = np.random.choice(X_np.shape[1], binary, replace=False)

    for idx in feature_idx:
        values = X_np[:,idx]

        for chunk in range(static_params['n_chunks']):
            start = static_params['chunk_size']*chunk
            end = static_params['chunk_size']*(chunk+1)

            values_chunk = np.copy(X_np[start:end,idx])
            th = np.mean(values_chunk)
        
            values[start:end][X_np[start:end,idx]>th] = 1
            values[start:end][X_np[start:end,idx]<=th] = 0

        X_np[:,idx] = values
    
    file = np.concatenate([X_np, y_np[:,np.newaxis]], axis=1)
    np.save('stream_generated.npy', file)
    np.savetxt('stream_generated.txt', file)

    s = sl.streams.NPYParser("stream_generated.npy", chunk_size=static_params['chunk_size'], n_chunks=static_params['n_chunks'])
    strs.append(s)

    X_np, y_np= stream._make_classification()

    new_X=[]
    idx=0
    while idx < X_np.shape[1]:
        if idx <= categoric*3:
            aa=[]
            for idx2 in range(3):
                values = X_np[:,idx+idx2]

                for chunk in range(static_params['n_chunks']):
                    start = static_params['chunk_size']*chunk
                    end = static_params['chunk_size']*(chunk+1)
                    
                    values_chunk = np.copy(X_np[start:end,idx+idx2])
                    th = np.mean(values_chunk)
                
                    values[start:end][X_np[start:end,idx+idx2]>th] = 1
                    values[start:end][X_np[start:end,idx+idx2]<=th] = 0

                aa.append(values)

            aa = np.array(aa).T
            aa = aa*np.array([4,2,1])
            aa = np.sum(aa, axis=1)
            new_X.append(aa)
            idx+=3

        else:
            new_X.append(X_np[:,idx])
            idx+=1

    new_X = np.array(new_X).T

    file = np.concatenate([new_X, y_np[:,np.newaxis]], axis=1)
    np.save('stream_generated2.npy', file)
    np.savetxt('stream_generated2.txt', file)
    strs.append(sl.streams.NPYParser('stream_generated2.npy', chunk_size=static_params['chunk_size'], n_chunks=static_params['n_chunks']))

    return strs


def find_real_drift(chunks, drifts):
    interval = round(chunks/drifts)
    arr = np.zeros((chunks))
    idx = [interval*(i+.5) for i in range(drifts)]
    for i in idx:
        arr[int(i)]=2
    return arr[1:]

metrics = e2_config.metrics()

replications = 10
random_states = np.random.randint(0, 10000, replications)

results_clf = [np.zeros((replications, 1, static_params['n_chunks']-1)), 
                np.zeros((replications, 1, static_params['n_chunks']-1)),
                np.zeros((replications, 1, static_params['n_chunks']-1))]
results_drf_arrs = [np.zeros((replications, 1, 2, static_params['n_chunks']-1)),
                np.zeros((replications, 1, 2, static_params['n_chunks']-1)),
                np.zeros((replications, 1, 2, static_params['n_chunks']-1))]

# n_features=[52, 52, 22]
n_features=[18, 18, 12]

for i in range(replications):
    streams = generate_categoric_streams(random_states[i])
    real_drf = find_real_drift(static_params['n_chunks'], drifs)

    for s_id, stream in enumerate(streams):
        detectors = [Meta(detector = SDDE(n_detectors= n_features[s_id], sensitivity=.5), base_clf = GaussianNB())]

        eval = sl.evaluators.TestThenTrain(metrics=metrics)
        eval.process(stream, detectors)

        scores = eval.scores
        results_clf[s_id][i] = scores[:,:,0]

        for det_id in range(1):
            results_drf_arrs[s_id][i, det_id, 0] = real_drf
            results_drf_arrs[s_id][i, det_id, 1] = np.array(detectors[0].detector.drift)
            print(results_drf_arrs[s_id][i, det_id, 1])

np.save('results_ex4/clf_num', results_clf[0])
np.save('results_ex4/drf_arr_num', results_drf_arrs[0])

np.save('results_ex4/clf_bin', results_clf[1])
np.save('results_ex4/drf_arr_bin', results_drf_arrs[1])

np.save('results_ex4/clf_cat', results_clf[2])
np.save('results_ex4/drf_arr_cat', results_drf_arrs[2])




