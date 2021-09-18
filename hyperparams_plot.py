import numpy as np
import matplotlib.pyplot as plt


subspace_sizes = [1,2,3,4,5,6]
n_detectors = [1,2,3,5,7,10,15]

res = np.load('hyperparams_clf.npy')
# res = np.load('hyperparams.npy')

res[res == np.inf] = np.nanmax(res[res != np.inf])+1 #xd

print(res.shape)

res_mean = np.mean(res, axis=0)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_title('clf_10f_100ch_150chs_5d seed:654')
# ax.set_title('10f_100ch_150chs_5d seed:654')


ax.imshow(res_mean, cmap='cividis')

ax.set_yticks(list(range(len(subspace_sizes))))
ax.set_yticklabels(subspace_sizes)
ax.set_ylabel("Subspace size")

ax.set_xticks(list(range(len(n_detectors))))
ax.set_xticklabels(n_detectors)
ax.set_xlabel("n detectors")

for _a, __a in enumerate(subspace_sizes):
    for _b, __b in enumerate(n_detectors):
        ax.text(_b, _a, "%.3f" % (
            res_mean[_a, _b]) , va='center', ha='center', c='white', fontsize=11)
            

plt.tight_layout()
plt.savefig("clf_10f_100ch_150chs_5d seed:654.png")
# plt.savefig("10f_100ch_150chs_5d seed:654.png")