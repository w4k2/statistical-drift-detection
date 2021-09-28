import numpy as np
import matplotlib.pyplot as plt
from methods import dderror
import e1_config

res_clf = np.load('results_ex1/clf_15feat_5drifts_sudden_1subspace_size.npy')
res_arr = np.load('results_ex1/drf_arr_15feat_5drifts_sudden_1subspace_size.npy')

print(res_clf.shape) #replications x threshold x detectors 
print(res_arr.shape) #replications x threshold x detectors x (real, detected) x chunks-1

th_arr = e1_config.e1_drf_threshold()
det_arr = e1_config.e1_n_detectors()

print(th_arr, det_arr)

res_clf_mean = np.mean(res_clf, axis=0)
print(res_clf_mean)
print(res_clf_mean.shape)

# plt.imshow(res_clf_mean)
# plt.savefig('foo.png')

dderror_arr = np.zeros((res_arr.shape[0],res_arr.shape[1], res_arr.shape[2]))

for rep in range(res_arr.shape[0]):
    for th in range(res_arr.shape[1]):
        for det in range(res_arr.shape[2]):
            real_drf = np.argwhere(res_arr[rep, th, det, 0]==2)
            det_drf = np.argwhere(res_arr[rep, th, det, 1]==2)
            err = dderror(real_drf, det_drf, res_arr.shape[4])
            dderror_arr[rep,th,det] = err

res_arr_mean = np.mean(dderror_arr, axis=0)

# plt.imshow(res_arr_mean)
# plt.savefig('foo.png')




fig, ax = plt.subplots(1, 1, figsize=(8, 8))

ax.imshow(res_clf_mean, cmap='cividis')

ax.set_yticks(list(range(len(th_arr))))
ax.set_yticklabels(th_arr)
ax.set_ylabel("Threshold")

ax.set_xticks(list(range(len(det_arr))))
ax.set_xticklabels(det_arr)
ax.set_xlabel("n detectors")

for _a, __a in enumerate(th_arr):
    for _b, __b in enumerate(det_arr):
        ax.text(_b, _a, "%.3f" % (
            res_clf_mean[_a, _b]) , va='center', ha='center', c='white', fontsize=11)
            

plt.tight_layout()
plt.savefig('foo.png')
# plt.savefig("figures/%if_100ch_150chs_5d seed_654.png" % f)

plt.close()


fig, ax = plt.subplots(1, 1, figsize=(8, 8))

ax.imshow(res_arr_mean, cmap='cividis')

ax.set_yticks(list(range(len(th_arr))))
ax.set_yticklabels(th_arr)
ax.set_ylabel("Threshold")

ax.set_xticks(list(range(len(det_arr))))
ax.set_xticklabels(det_arr)
ax.set_xlabel("n detectors")

for _a, __a in enumerate(th_arr):
    for _b, __b in enumerate(det_arr):
        ax.text(_b, _a, "%.3f" % (
            res_arr_mean[_a, _b]) , va='center', ha='center', c='white', fontsize=11)
            

plt.tight_layout()
plt.savefig('foo2.png')
# plt.savefig("figures/%if_100ch_150chs_5d seed_654.png" % f)

plt.close()