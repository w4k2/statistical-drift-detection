import config
import strlearn as sl
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

RANDOM_STATE = 51523

clf_names = config.e3_clf_names()
metrics = config.metrics()
metrics_names = config.metrics_names()
streams = config.e3_streams(RANDOM_STATE)

# detectors = ['DDM', 'EDDM', 'ADWIN', "SDDM", "ESDDM", "ALWAYS", "NEVER"]

colors = ['green', 'brown', 'dodgerblue', 'purple', 'gold', 'black', 'gray']
lss = ['-', '-', '-', '-', '-', '-', '-']
ss = [15,15,15,15,15,1,1]

for hash, stream in streams.items():
    print(hash, stream)

    clfs = config.e3_clfs()

    for clf in clfs:
        print("-", clf)

    start = time.time()
    evaluator = sl.evaluators.TestThenTrain(metrics)
    evaluator.process(stream, clfs)
    end = time.time()

    print("Time:", end - start)

    print(np.mean(evaluator.scores, axis=1))

    fig,ax = plt.subplots(2,2,figsize=(12,8))

    plin = np.linspace(0, stream.n_chunks, stream.n_chunks * stream.chunk_size)
    olin = np.linspace(0, stream.n_chunks, stream.n_chunks - 1)


    for clf_idx, clf in enumerate(evaluator.clfs_):
        ddf = np.array(clf.detector.drift)
        ddf[ddf == 0] = -1
        ax[0,0].scatter(olin, ddf-.1*clf_idx, label=clf_names[clf_idx], c=colors[clf_idx], s=ss[clf_idx])

        for m_idx in range(2):
            ax[1,m_idx].plot(olin,
                         medfilt(evaluator.scores[clf_idx, :, m_idx], 21),
                         label=clf_names[clf_idx],
                         c=colors[clf_idx])

        if clf_idx < 5:
            ax[0,1].plot(np.cumsum(np.array(clf.detector.drift)==2),
                         c = colors[clf_idx])
            ax[0,1].plot(np.cumsum(np.array(clf.detector.drift)==1),
                         c = colors[clf_idx], ls=':')

    ax[0,0].plot(plin, 3*stream.concept_probabilities, c='red', label='drifts', lw=1)

    for i in range(2):
        for j in range(2):
            ax[i,j].legend(loc=3 if i == 0 else 3, facecolor='white', framealpha=.95, edgecolor='white', ncol=3)
            ax[i,j].grid(ls=":")
            ax[i,j].spines['right'].set_visible(False)
            ax[i,j].spines['top'].set_visible(False)

    ax[1,0].set_title(hash)
    ax[0,0].set_title(stream)
    ax[0,0].set_ylim(-.1,3.1)
    ax[1,0].set_ylim(.5,1)

    ax[0,0].set_yticks([1, 2])
    ax[0,0].set_yticklabels(['warning', 'drift'])
    plt.tight_layout()

    plt.savefig('figures/18f_n_%s.png' % hash)
    plt.savefig('foo.png')

    # exit()
