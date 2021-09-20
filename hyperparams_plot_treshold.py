import numpy as np
import matplotlib.pyplot as plt

treshold = [1,3,5,7,9,11,13,15,17,19]
n_detectors = [1,2,3,5,7,10,15,30]

n_features = [10,15,20,25]

for f in n_features:
    print(f)

    # res = np.load('results/th_hyperparams_clf_%i_features.npy' % f)
    res = np.load('results/th_hyperparams_%i_features.npy' % f)

    res[res == np.inf] = np.nanmax(res[res != np.inf])+1 #xd

    print(res.shape)

    res_mean = np.mean(res, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # ax.set_title('th_clf_%if_100ch_150chs_5d seed:654' % f)
    ax.set_title('th_%if_100ch_150chs_5d seed:654' % f)

    ax.imshow(res_mean, cmap='cividis')

    ax.set_yticks(list(range(len(treshold))))
    ax.set_yticklabels(treshold)
    ax.set_ylabel("Detection treshold")

    ax.set_xticks(list(range(len(n_detectors))))
    ax.set_xticklabels(n_detectors)
    ax.set_xlabel("n detectors")

    for _a, __a in enumerate(treshold):
        for _b, __b in enumerate(n_detectors):
            ax.text(_b, _a, "%.3f" % (
                res_mean[_a, _b]) , va='center', ha='center', c='white', fontsize=11)
                

    plt.tight_layout()
    # plt.savefig("figures/th_clf_%if_100ch_150chs_5d seed_654.png" % f)
    plt.savefig("figures/th_%if_100ch_150chs_5d seed_654.png" % f)

    plt.close()