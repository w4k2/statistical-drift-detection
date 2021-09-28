import strlearn as sl
import numpy as np
from methods import KDDDE, Meta
from sklearn.naive_bayes import GaussianNB
import e1_config
from tqdm import tqdm

def find_real_drift(chunks, drifts):
    interval = round(chunks/drifts)
    arr = np.zeros((chunks))
    idx = [interval*(i+.5) for i in range(drifts)]
    for i in idx:
        arr[int(i)]=2
    return arr[1:]

np.random.seed(654)

subspace_sizes = e1_config.e1_subspace_sizes()
n_detectors = e1_config.e1_n_detectors().astype('int')
drf_threshold = e1_config.e1_drf_threshold()

replications = e1_config.e1_replications()
random_states = np.random.randint(0, 10000, replications)

static_params = e1_config.e1_static()
drf_types = e1_config.e1_drift_types()

print(len(drf_types),replications,len(subspace_sizes),len(n_detectors),len(drf_threshold))
t = len(drf_types)*replications*len(subspace_sizes)*len(n_detectors)*len(drf_threshold)
pbar = tqdm(total=t)

real_drf = find_real_drift(static_params['n_chunks'], static_params['n_drifts'])

for ss_id, ss in enumerate(subspace_sizes):
    for drf_type in drf_types:

        results_clf = np.zeros((replications, len(drf_threshold), len(n_detectors)))
        results_drf_arrs = np.zeros((replications, len(drf_threshold), len(n_detectors), 2, static_params['n_chunks']-1))
        # replications x th x detectors x (real_drf, detected_drf) x chunks

        for replication in range(replications):
            str_name = "%ifeat_%idrifts_%s_%isubspace_size" % (static_params['n_features'],static_params['n_drifts'],drf_type,ss)
            print(str_name)

            for det_id, det in enumerate(n_detectors):
                for th_id, th in enumerate(drf_threshold):
                    config = {
                        **static_params,
                        **drf_types[drf_type],
                        'random_state': random_states[replication]
                                }
                    stream = sl.streams.StreamGenerator(**config)

                    # import pprint
                    # pp = pprint.PrettyPrinter(indent=4)

                    # filename = '%08i_%s' % (random_states[replication], drf_type)
                    # print(filename, stream)
                    # pp.pprint(config)
                
                    print("replication: %i, stream: %s" % (replication, str_name))
                    print("ss: %i, det: %i, th: %f" % (ss, det, th))

                    clf = Meta(GaussianNB(), KDDDE(n_detectors=det, subspace_size=ss, random_state=random_states[replication], sensitivity=th))
                    eval = sl.evaluators.TestThenTrain(metrics=(sl.metrics.balanced_accuracy_score))
                    eval.process(stream, clf)

                    # detected_drifts = np.argwhere(np.array(clf.detector.drift) == 2)

                    score = np.mean(eval.scores)
                    results_clf[replication, th_id, det_id] = score

                    results_drf_arrs[replication, th_id, det_id, 0] = real_drf
                    results_drf_arrs[replication, th_id, det_id, 1] = np.array(clf.detector.drift)

                    pbar.update(1)

        np.save('results_ex1/clf_%s' % str_name, results_clf)
        np.save('results_ex1/drf_arr_%s' % str_name, results_drf_arrs)
pbar.close()