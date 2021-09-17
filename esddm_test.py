import strlearn as sl
import matplotlib.pyplot as plt
import numpy as np
from methods import ESDDM, Meta
from sklearn.naive_bayes import GaussianNB

n_chunks = 200
chunk_size = 250
n_detectors = 10
stream = sl.streams.StreamGenerator(n_drifts=5,
                                    n_chunks=n_chunks,
                                    chunk_size=chunk_size,
                                    n_features=20,
                                    n_informative=20,
                                    n_redundant=0,
                                    recurring=False,
                                    random_state=3578989345)

clf = Meta(GaussianNB(), ESDDM(n_detectors=n_detectors, subspace_size=2))
# clf = Meta(GaussianNB(), SDDM())
eval = sl.evaluators.TestThenTrain(metrics=(sl.metrics.balanced_accuracy_score))
eval.process(stream, clf)

#els1 = np.array(clf.detector.els_arr_plot[0])
#els2 = np.array(clf.detector.els_arr_plot[1])
#els3 = np.array(clf.detector.els_arr_plot[2])
drifts = np.argwhere(np.array(clf.detector.drift) == 2)

#rgb = np.array([els1, np.squeeze(els2), els3])
#rgb -= np.min(rgb, axis=1)[:,np.newaxis]
#rgb /= np.max(rgb, axis=1)[:,np.newaxis]

fig, ax = plt.subplots(5, 1, figsize=(5,10))

#ax[0].plot(els1[5:], c="r")
#ax[0].plot(clf.detector.els_arr[0])

print(clf.detector.els_arr_plot, clf.detector.els_arr_plot.shape)

eee = clf.detector.els_arr_plot
for i in range(n_detectors):
    cymbal = []
    for j in range(3):
        ax[j].plot(eee[i,j,:])
        cymbal.append(eee[i,j,:])

    for z in range(3):
        cymbal[z] -= np.min(cymbal[z])
        cymbal[z] /= np.max(cymbal[z])

    ax[3].plot(np.mean(np.array(cymbal), axis=0))


#ax[1].plot(els2[5:], c="g")
#ax[2].plot(els3[5:], c="b")
#ax[3].imshow(np.array([rgb.T]), aspect=len(els1)/2)
ax[4].plot(np.linspace(0, stream.n_chunks-1, len(stream.concept_probabilities)),
           stream.concept_probabilities)
for i in range(5):
    ax[i].set_xticks(clf.detector.niewiem)
    #ax[i].set_xticks(drifts[1:])
    ax[i].grid(ls=":")

plt.tight_layout()
plt.savefig("foo.png")
