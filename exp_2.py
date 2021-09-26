import strlearn as sl
import numpy as np
from methods import ESDDM, Meta
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
n_chunks = config.n_chunks()
chunk_size = config.chunk_size()

# det_th = np.array(config.e2_det_th()) #180 x 2
n_detectors = config.e2_n_detectors()
det_threshold = config.e2_det_threshold()

replications = config.replications()
random_states = np.random.randint(0, 10000, replications)

n_features = config.n_featues()
n_drifts = config.n_drifts()

recurring = config.recurring()
incremental = config.incremental()
concept_sigmoid_spacing = config.concept_sigmoid_spacing()

t = len(n_detectors)*len(det_threshold)*len(recurring)*len(concept_sigmoid_spacing)*replications*len(incremental)*len(n_features)*len(n_drifts)
pbar = tqdm(total=t)

for f_id, f in enumerate(n_features):
    for d_id, drifts in enumerate(n_drifts):
        real_drf = find_real_drift(n_chunks, drifts)

        for r_id, rec in enumerate(recurring):
            for i_id, incr in enumerate(incremental):
                for css_id, css in enumerate(concept_sigmoid_spacing):
                    if incr and css != 5.:
                        continue
                    
                    results_clf = np.zeros((replications, len(det_threshold), len(n_detectors)))
                    results_drf_arrs = np.zeros((replications, len(det_threshold), len(n_detectors), 2, n_chunks-1))
                    # replications x ss x detectors x (real_drf, detected_drf) x chunks

                    for replication in range(replications):

                        str_name = "%ifeat_%idrifts_%irec_%iincr_%icss" % (f,drifts,rec,incr,css)

                        for th_id, th in enumerate(det_threshold):
                            for det_id, det in enumerate(n_detectors):

                                if th>det:
                                    results_clf[replication, th_id, det_id] = 0.5

                                    results_drf_arrs[replication, th_id, det_id, 0] = real_drf
                                    results_drf_arrs[replication, th_id, det_id, 1] = range(n_chunks)
                                    
                                    pbar.update(1)
                                    continue

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
                                print("th: %i, det: %i" % (th, det))

                                clf = Meta(GaussianNB(), ESDDM(n_detectors=det, subspace_size=1, random_state=random_states[replication], drf_threshold=th))
                                eval = sl.evaluators.TestThenTrain(metrics=(sl.metrics.balanced_accuracy_score))
                                eval.process(stream, clf)

                                score = np.mean(eval.scores)
                                results_clf[replication, th_id, det_id] = score

                                results_drf_arrs[replication, th_id, det_id, 0] = real_drf
                                results_drf_arrs[replication, th_id, det_id, 1] = np.array(clf.detector.drift)

                                pbar.update(1)

                    np.save('results_ex2/clf_%s' % str_name, results_clf)
                    np.save('results_ex2/drf_arr_%s' % str_name, results_drf_arrs)
pbar.close()