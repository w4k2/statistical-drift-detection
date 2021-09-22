import numpy as np
import matplotlib.pyplot as plt

# threshold = [1,3,5,7,9,11,13,15,17,19]
# n_detectors = [1,2,3,5,7,10,15,30]

# n_features = [10,15,20,25]

threshold = range(1,30)
n_detectors = range(1,30)

n_features = [15]

for f in n_features:
    print(f)

    res = np.load('results/1-30-th_hyperparams_clf_%i_features.npy' % f)
    # res = np.load('results/1-30-th_hyperparams_%i_features.npy' % f)

    res[res == np.inf] = np.nanmax(res[res != np.inf])+1 #xd

    print(res.shape)

    res_mean = np.mean(res, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
    ax.set_title('BAC th_clf_%if_100ch_150chs_5d seed:654' % f)
    # ax.set_title('Drift difference error th_%if_100ch_150chs_5d seed:654' % f)

    ax.imshow(res_mean, cmap='cividis')

    ax.set_yticks(list(range(len(threshold))))
    ax.set_yticklabels(threshold)
    ax.set_ylabel("Detection threshold")

    ax.set_xticks(list(range(len(n_detectors))))
    ax.set_xticklabels(n_detectors)
    ax.set_xlabel("n detectors")

    for _a, __a in enumerate(threshold):
        for _b, __b in enumerate(n_detectors):
            ax.text(_b, _a, "%.3f" % (
                res_mean[_a, _b]) , va='center', ha='center', c='white', fontsize=4)
                

    plt.tight_layout()
    plt.savefig("figures/th_clf_%if_100ch_150chs_5d seed_654.png" % f)
    # plt.savefig("figures/th_%if_100ch_150chs_5d seed_654.png" % f)

    plt.close()


    # inny plot
    # opts = np.min(res_mean, axis=0)
    opts = np.max(res_mean, axis=0)
    argmins = []
    for m_i, m in enumerate(opts):
        argmins.append(np.argwhere(res_mean[:,m_i]==m))
        argmins[-1] = np.mean(argmins[-1])+1
    argmins = np.array(argmins)
    print(argmins, argmins.shape)

    plt.plot(n_detectors, argmins, label='Optimal threshold')
    plt.plot(n_detectors, np.sqrt(n_detectors), label = 'sqrt n_detectors')
    plt.plot(n_detectors, np.array(n_detectors)*0.75, label = '3/4 n_detectors')

    plt.xlabel('n_detectors')
    plt.ylabel('threshold')

    plt.legend()
    # plt.savefig('figures/opt.png')
    plt.savefig('figures/opt2.png')
