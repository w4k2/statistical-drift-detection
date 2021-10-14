import e2_config
import numpy as np
import matplotlib.pyplot as plt
from methods.dderror import dderror

streams = [
    "covtypeNorm-1-2vsAll-pruned.arff",
    "INSECTS-abrupt_imbalanced_norm_5prc.arff",
    "INSECTS-abrupt_imbalanced_norm.arff",
    "INSECTS-gradual_imbalanced_norm_5prc.arff",
    "INSECTS-gradual_imbalanced_norm.arff",
    "INSECTS-incremental_imbalanced_norm_5prc.arff",
    "INSECTS-incremental_imbalanced_norm.arff",
    "poker-lsn-1-2vsAll-pruned.arff"
]

n_features = [54,33,33,33,33,33,33,10]
n_chunks = [265, 300, 300, 100, 100, 380, 380, 359]
chunk_size = 1000

colors = ['red', 'gold', 'purple', 'green', 'cornflowerblue', 'gray']

detector_names = e2_config.e2_clf_names()

for s in streams:
    results_clf = np.load("results_ex2_real/clf_%s.npy" % s)[0]
    results_drf_arr = np.load("results_ex2_real/drf_arr_%s.npy" % s)[0,:,1,:]

    print(results_clf.shape) # detectors x chunks - 1
    print(results_drf_arr.shape) #detectors x chunks - 1

    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(20, 6), dpi=300)
    fig.suptitle("%s" % s, fontsize=15)

    for det_id, det_name in enumerate(detector_names):
        ax[0].plot(results_clf[det_id], label=det_name, c=colors[det_id], alpha=0.7)

    ax[0].legend()
    ax[0].set_title("Classification accuracy")

    ax[1].set_xlim(0,len(results_clf[1]))

    for det_id, det_name in enumerate(detector_names):
        detected = np.argwhere(results_drf_arr[det_id]==2)
        det_y = [det_id for i in range(len(detected))]

        ax[1].scatter(detected, det_y, alpha=0.4, c=colors[det_id])

    ax[1].set_yticks(list(range(len(detector_names))))
    ax[1].set_yticklabels("%s" % (d) for i, d in enumerate(detector_names))
    ax[1].set_title("Detections")

    fig.subplots_adjust(top=0.93)

    plt.tight_layout()
    plt.savefig('figures_ex2/%s.png' % s)
    # exit()



