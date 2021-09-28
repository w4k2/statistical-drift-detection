import numpy as np
import matplotlib.pyplot as plt
from methods import dderror



res_clf = np.load('results_ex1/clf_5feat_5drifts_sudden_1subspace_size.npy')
res_arr = np.load('results_ex1/drf_arr_5feat_5drifts_sudden_1subspace_size.npy')

print(res_clf.shape) #replications x threshold x detectors 
print(res_arr.shape) #replications x threshold x detectors x (real, detected) x chunks-1


res_clf_mean = np.mean(res_clf, axis=0)
print(res_clf_mean)
print(res_clf_mean.shape)

# plt.imshow(res_clf_mean)
# plt.savefig('foo.png')

dderror_arr = np.zeros((res_arr.shape[0],res_arr.shape[1], res_arr.shape[2]))

for rep in range(res_arr.shape[0]):
    for th in range(res_arr.shape[1]):
        for det in range(res_arr.shape[2]):
            err = dderror(res_arr[rep, th, det, 0], res_arr[rep, th, det, 1], res_arr.shape[4])
            dderror_arr[rep,th,det] = err

res_arr_mean = np.mean(dderror_arr, axis=0)

plt.imshow(res_arr_mean)
plt.savefig('foo.png')

# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# ax.set_title('clf_%if_100ch_150chs_5d seed:654' % f)
# # ax.set_title('%if_100ch_150chs_5d seed:654' % f)

# ax.imshow(res_clf_mean, cmap='cividis')

# ax.set_yticks(list(range(len(subspace_sizes))))
# ax.set_yticklabels(subspace_sizes)
# ax.set_ylabel("Subspace size")

# ax.set_xticks(list(range(len(n_detectors))))
# ax.set_xticklabels(n_detectors)
# ax.set_xlabel("n detectors")

# for _a, __a in enumerate(subspace_sizes):
#     for _b, __b in enumerate(n_detectors):
#         ax.text(_b, _a, "%.3f" % (
#             res_mean[_a, _b]) , va='center', ha='center', c='white', fontsize=11)
            

# plt.tight_layout()
# plt.savefig("figures/clf_%if_100ch_150chs_5d seed_654.png" % f)
# # plt.savefig("figures/%if_100ch_150chs_5d seed_654.png" % f)

# plt.close()