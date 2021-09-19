import strlearn as sl
import matplotlib.pyplot as plt
import numpy as np
from methods import ESDDM, Meta
from sklearn.naive_bayes import GaussianNB
from scipy.stats import hmean

def find_real_drift(chunks, drifts):
    interval = round(chunks/drifts)
    return [interval*(i+.5) for i in range(drifts)]

def find_nearest(value, array):
    dist = (np.abs(array-value)).min()
    return dist

def dderror(chunks, drifts, detected_drift):
    if(len(detected_drift))==0:
        return np.inf
    real_drift = find_real_drift(chunks, drifts)

    # distances from real
    d_real = []
    for d in detected_drift:
        d_real.append(find_nearest(d, real_drift))

    # distances from detected
    d_detected = []
    for r in real_drift:
        d_detected.append(find_nearest(r, detected_drift))

    # print(real_drift)
    # print(d_detected, d_real)
    return (np.sum(d_detected) + np.sum(d_real))/len(real_drift)


n_chunks = 100
chunk_size = 150
n_drifts=5

tries = 5

subspace_sizes = [1,2,3,4,5,6]
n_detectors = [1,2,3,5,7,10,15,30]

n_features = [15,20,25,30,35,40,45]

np.random.seed(654)
random_states = np.random.randint(0,10000, tries)
print(random_states)

for f_id, f in enumerate(n_features):
    print(f_id, f)
    
    results = np.zeros((tries, len(subspace_sizes), len(n_detectors)))
    results_err = np.zeros((tries, len(subspace_sizes), len(n_detectors)))

    for rs_id, rs in enumerate(random_states):
        for ss_id, ss in enumerate(subspace_sizes):
            for det_id, det in enumerate(n_detectors):
                
                print(f, rs_id, ss, det)

                stream = sl.streams.StreamGenerator(n_drifts=n_drifts,
                                                    n_chunks=n_chunks,
                                                    chunk_size=chunk_size,
                                                    n_features=f,
                                                    n_informative=f,
                                                    n_redundant=0,
                                                    recurring=False,
                                                    random_state=rs)


                clf = Meta(GaussianNB(), ESDDM(n_detectors=det, subspace_size=ss, random_state=rs, drf_level=det/2))
                eval = sl.evaluators.TestThenTrain(metrics=(sl.metrics.balanced_accuracy_score))
                eval.process(stream, clf)

                drifts = np.argwhere(np.array(clf.detector.drift) == 2)
                elements = clf.detector.combined_elements
                elements_all = np.array(clf.detector.all_elements)

                # try:
                #     # Plot
                #     fig, ax = plt.subplots(5, 1, figsize=(5,10))
                #     for i in range(det):
                #         for j in range(2):
                #             ax[j].plot(elements[i,j,:])

                #     ax[0].set_title("TDM")
                #     ax[1].set_title("CMCD")

                #     ax[2].plot(hmean(elements_all, axis=0))
                #     ax[2].set_title("Harmonic mean all")

                #     ax[3].plot(clf.detector.confidence)
                #     ax[3].hlines(clf.detector.drf_level, 0, n_chunks, ls=":", color='r')
                #     ax[3].set_title("Decision confidence")

                #     ax[4].plot(np.linspace(0, stream.n_chunks-1, len(stream.concept_probabilities)),
                #             stream.concept_probabilities)
                #     ax[4].set_title("Concept")

                #     for i in range(5):
                #         ax[i].set_xticks(drifts)
                #         ax[i].grid(ls=":")

                #     plt.tight_layout()
                #     plt.savefig("foo.png")
                #     plt.clf()
                # except:
                #     pass

                # Error / Score

                score = np.mean(eval.scores)
                results[rs_id, ss_id, det_id] = score
                print(score) 

    print(results)
    np.save('results/hyperparams_clf_%i_features' % f, results)