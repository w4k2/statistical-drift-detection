import strlearn as sl
import numpy as np
from sklearn.base import clone
import e2_config
from tqdm import tqdm
from methods import KDDDE, Meta
from sklearn.naive_bayes import GaussianNB

def find_real_drift(chunks, drifts):
    interval = round(chunks/drifts)
    arr = np.zeros((chunks))
    idx = [interval*(i+.5) for i in range(drifts)]
    for i in idx:
        arr[int(i)]=2
    return arr[1:]

np.random.seed(654)

replications = 10
random_states = np.random.randint(0, 10000, replications)

static_params = e2_config.e2_static()
drf_types = e2_config.e2_drift_types()
recurring = e2_config.e2_recurring()

base_detectors = [
    # Meta(detector = KDDDE(sensitivity = 0.3), base_clf = GaussianNB()),
    # Meta(detector = KDDDE(sensitivity = 0.35), base_clf = GaussianNB()),
    # Meta(detector = KDDDE(sensitivity = 0.4), base_clf = GaussianNB()),
    Meta(detector = KDDDE(sensitivity = 0.45), base_clf = GaussianNB()),
    Meta(detector = KDDDE(sensitivity = 0.5), base_clf = GaussianNB()),
    Meta(detector = KDDDE(sensitivity = 0.55), base_clf = GaussianNB()),
    Meta(detector = KDDDE(sensitivity = 0.6), base_clf = GaussianNB()),
    Meta(detector = KDDDE(sensitivity = 0.7), base_clf = GaussianNB()),
    Meta(detector = KDDDE(sensitivity = 0.8), base_clf = GaussianNB()),
]

metrics = e2_config.metrics()

print(len(drf_types),len(recurring),replications)
t = len(drf_types)*len(recurring)*replications
pbar = tqdm(total=t)
real_drf = find_real_drift(static_params['n_chunks'], static_params['n_drifts'])

for rec in recurring:
    for drf_type in drf_types:
        results_clf = np.zeros((replications, len(base_detectors), static_params['n_chunks']-1))
        results_drf_arrs = np.zeros((replications, len(base_detectors), 2, static_params['n_chunks']-1))
        # replications x detectors x (real_drf, detected_drf) x chunks

        for replication in range(replications):

            detectors=[]
            for det_id in range(len(base_detectors)):
                detectors.append(clone(base_detectors[det_id]))

            str_name = "%ifeat_%idrifts_%s_%s" % (static_params['n_features'],static_params['n_drifts'],drf_type,rec)
            
            config = {
                **static_params,
                **drf_types[drf_type],
                **recurring[rec],
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

        np.save('results_ex2_2/clf_%s' % str_name, results_clf)
        np.save('results_ex2_2/drf_arr_%s' % str_name, results_drf_arrs)

pbar.close()