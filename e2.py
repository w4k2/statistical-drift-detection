"""
Experiment 2 -- comparison of DD methods for synthetic streams
(with 3,5,7 drits and 10,15,20 features)
"""
import strlearn as sl
import numpy as np
from sklearn.base import clone
import e2_config
from tqdm import tqdm

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

n_features = e2_config.e2_n_features()
n_drifts = e2_config.e2_n_drifts()

drf_types = e2_config.e2_drift_types()
recurring = e2_config.e2_recurring()


metrics = e2_config.metrics()

print(len(n_drifts), len(n_features), len(drf_types) ,len(recurring), replications)
t = len(n_drifts)*len(n_features)*len(drf_types)*len(recurring)*replications
pbar = tqdm(total=t)

for n_f in n_features:
    base_detectors = e2_config.e2_clfs(sdde_n_det = n_f, sdde_sensitivity=0.35)

    for n_d in n_drifts:
        real_drf = find_real_drift(static_params['n_chunks'], n_d)

        for rec in recurring:
            for drf_type in drf_types:

                # if drf_type != 'sudden':
                #     continue

                # if n_d != 5:
                #     continue

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
                    stream = sl.streams.StreamGenerator(**config)

                    print("replication: %i, stream: %s" % (replication, str_name))

                    eval = sl.evaluators.TestThenTrain(metrics=metrics)
                    eval.process(stream, detectors)

                    scores = eval.scores
                    results_clf[replication] = scores[:,:,0]

                    for det_id in range(len(detectors)):
                        results_drf_arrs[replication, det_id, 0] = real_drf
                        results_drf_arrs[replication, det_id, 1] = np.array(detectors[det_id].detector.drift)

                    pbar.update(1)

                np.save('results_ex2_d_f_45/clf_%s_2' % str_name, results_clf)
                np.save('results_ex2_d_f_45/drf_arr_%s_2' % str_name, results_drf_arrs)

pbar.close()