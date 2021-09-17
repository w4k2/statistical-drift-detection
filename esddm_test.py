import strlearn as sl
import matplotlib.pyplot as plt
import numpy as np
from methods import ESDDM, Meta
from sklearn.naive_bayes import GaussianNB

n_chunks = 200
chunk_size = 250
n_detectors = 10
stream = sl.streams.StreamGenerator(n_drifts=11,
                                    n_chunks=n_chunks,
                                    chunk_size=chunk_size,
                                    n_features=21,
                                    n_informative=21,
                                    n_redundant=0,
                                    recurring=False,
                                    random_state=3578989345)

clf = Meta(GaussianNB(), ESDDM(n_detectors=n_detectors, subspace_size=2, random_state=121222))
eval = sl.evaluators.TestThenTrain(metrics=(sl.metrics.balanced_accuracy_score))
eval.process(stream, clf)

drifts = np.argwhere(np.array(clf.detector.drift) == 2)
elements = clf.detector.combined_elements
elements_all = np.array(clf.detector.all_elements)

fig, ax = plt.subplots(5, 1, figsize=(5,10))
for i in range(n_detectors):
    for j in range(2):
        ax[j].plot(elements[i,j,:])

ax[0].set_title("TDM")
ax[1].set_title("CMCD")

ax[2].plot(np.mean(elements_all, axis=0))
ax[2].set_title("Mean all")

ax[3].plot(clf.detector.confidence)
ax[3].hlines(clf.detector.drf_level, 0, n_chunks, ls=":", color='r')
ax[3].set_title("Decision confidence")

ax[4].plot(np.linspace(0, stream.n_chunks-1, len(stream.concept_probabilities)),
           stream.concept_probabilities)
ax[4].set_title("Concept")

for i in range(5):
    ax[i].set_xticks(drifts)
    ax[i].grid(ls=":")

plt.tight_layout()
plt.savefig("foo.png")
