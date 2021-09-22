import strlearn as sl
import numpy as np
from methods import ESDDM, Meta, dderror
from sklearn.naive_bayes import GaussianNB
import config
from tqdm import tqdm

np.random.seed(654)
n_chunks = 100
chunk_size = 100

det_threshold = [1,3]# config.e2_det_threshold()
n_detectors = [1,3]# config.e2_n_detectors()

replications = 3# config.replications()
random_states = np.random.randint(0, 10000, replications)

n_features = 10
n_drifts = 6
recurring = config.recurring()
concept_sigmoid_spacing = config.concept_sigmoid_spacing()

t = len(recurring)*len(concept_sigmoid_spacing)*replications*len(det_threshold)*len(n_detectors)
pbar = tqdm(total=t)

for r_id, rec in enumerate(recurring):
    for css_id, css in enumerate(concept_sigmoid_spacing):
        
        results_clf = np.zeros((replications, len(det_threshold), len(n_detectors)))
        results_err = np.zeros((replications, len(det_threshold), len(n_detectors)))

        for replication in range(replications):

            str_name = "%irec_%icss" % (rec,css)

            for t_id, th in enumerate(det_threshold):
                for det_id, det in enumerate(n_detectors):
                    
                    if th>det:
                        results_clf[replication, t_id, det_id] = 0.5
                        results_err[replication, t_id, det_id] = np.inf
                        pbar.update(1)
                        continue

                    stream = sl.streams.StreamGenerator(
                                recurring = rec,
                                concept_sigmoid_spacing = css,
                                n_drifts=n_drifts,
                                random_state=random_states[replication],
                                weights=[0.5, 0.5],
                                y_flip=0.01,
                                n_features=n_features,
                                n_informative=n_features,
                                n_redundant=0,
                                n_repeated=0,
                                n_clusters_per_class=1,
                                n_chunks=n_chunks,
                                chunk_size=chunk_size,
                                n_classes = 2
                            )


                    print("replication: %i, stream: %s" % (replication, str_name))
                    print("th: %i, det: %i" % (th, det))

                    clf = Meta(GaussianNB(), ESDDM(n_detectors=det, drf_threshold=th, random_state=random_states[replication]))
                    eval = sl.evaluators.TestThenTrain(metrics=(sl.metrics.balanced_accuracy_score))
                    eval.process(stream, clf)

                    detected_drifts = np.argwhere(np.array(clf.detector.drift) == 2)

                    score = np.mean(eval.scores)
                    results_clf[replication, t_id, det_id] = score

                    error = dderror(n_chunks, n_drifts, detected_drifts)
                    results_err[replication, t_id, det_id] = error

                    pbar.update(1)

        np.save('results_ex2/clf_%s' % str_name, results_clf)
        np.save('results_ex2/err_%s' % str_name, results_err)

pbar.close()