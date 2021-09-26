import strlearn as sl
import numpy as np
from methods import ESDDM, Meta, dderror
from sklearn.naive_bayes import GaussianNB
import config
from tqdm import tqdm

def find_real_drift(chunks, drifts):
    interval = round(chunks/drifts)
    arr = np.zeros((chunks))
    idx = [interval*(i+.5) for i in range(drifts)]
    for i in idx:
        arr[int(i)]=2
    return arr[1:]

np.random.seed(654)
n_chunks = int(config.n_chunks()/2)
chunk_size = int(config.chunk_size()/2)

subspace_sizes = config.e1_subspace_sizes()
n_detectors = config.e1_n_detectors().astype('int')

replications = config.replications()
random_states = np.random.randint(0, 10000, replications)

n_features = config.n_featues().astype('int')
n_drifts = config.n_drifts()
recurring =  config.recurring()
concept_sigmoid_spacing = config.concept_sigmoid_spacing()
incremental =  config.incremental()

print(n_features)

t = len(n_features)*len(n_drifts)*len(recurring)*len(concept_sigmoid_spacing)*replications*len(subspace_sizes)*len(n_detectors)
pbar = tqdm(total=t)

for f_id, f in enumerate(n_features):
    for d_id, drifts in enumerate(n_drifts):
        real_drf = find_real_drift(n_chunks, drifts)

        for r_id, rec in enumerate(recurring):
            for i_id, incr in enumerate(incremental):
                for css_id, css in enumerate(concept_sigmoid_spacing):
                    if incr and css != 5.:
                        continue
                    
                    results_clf = np.zeros((replications, len(subspace_sizes), len(n_detectors)))
                    results_drf_arrs = np.zeros((replications, len(subspace_sizes), len(n_detectors), 2, n_chunks-1))
                    # replications x ss x detectors x (real_drf, detected_drf) x chunks

                    for replication in range(replications):

                        str_name = "%ifeat_%idrifts_%irec_%iincr_%icss" % (f,drifts,rec,incr,css)

                        for ss_id, ss in enumerate(subspace_sizes):
                            for det_id, det in enumerate(n_detectors):

                                stream = sl.streams.StreamGenerator(
                                            recurring = rec,
                                            concept_sigmoid_spacing = css,
                                            n_drifts=drifts,
                                            random_state=random_states[replication],
                                            weights=[0.5, 0.5],
                                            y_flip=0.01,
                                            n_features=f,
                                            n_informative=f,
                                            n_redundant=0,
                                            n_repeated=0,
                                            n_clusters_per_class=1,
                                            n_chunks=n_chunks,
                                            chunk_size=chunk_size,
                                            n_classes = 2,
                                            incremental = incr
                                        )


                                print("replication: %i, stream: %s" % (replication, str_name))
                                print("ss: %i, det: %i" % (ss, det))

                                clf = Meta(GaussianNB(), ESDDM(n_detectors=det, subspace_size=ss, random_state=random_states[replication]))
                                eval = sl.evaluators.TestThenTrain(metrics=(sl.metrics.balanced_accuracy_score))
                                eval.process(stream, clf)

                                # detected_drifts = np.argwhere(np.array(clf.detector.drift) == 2)

                                score = np.mean(eval.scores)
                                results_clf[replication, ss_id, det_id] = score

                                results_drf_arrs[replication, ss_id, det_id, 0] = real_drf
                                results_drf_arrs[replication, ss_id, det_id, 1] = np.array(clf.detector.drift)

                                pbar.update(1)

                    np.save('results_ex1/clf_%s' % str_name, results_clf)
                    np.save('results_ex1/drf_arr_%s' % str_name, results_drf_arrs)
pbar.close()