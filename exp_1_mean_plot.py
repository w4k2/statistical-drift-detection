import numpy as np
import matplotlib.pyplot as plt
from methods import dderror
import e1_config

subspace_sizes = e1_config.e1_subspace_sizes()
drf_types = e1_config.e1_drift_types()
th_arr = e1_config.e1_drf_threshold()
det_arr = e1_config.e1_n_detectors()

print(th_arr, det_arr)

fig, ax = plt.subplots(3, 3, figsize=(8*1.618,8),
                       sharey=True, sharex=True)

for ss_id, ss in enumerate(subspace_sizes):
    for drf_id, drf in enumerate(drf_types):
        # if drf == 'incremental':
        #     continue

        res_clf = np.load('results_ex1/clf_15feat_5drifts_%s_%isubspace_size.npy' %  (drf, ss))
        res_arr = np.load('results_ex1/drf_arr_15feat_5drifts_%s_%isubspace_size.npy' %  (drf, ss))

        print(res_clf.shape) #replications x threshold x detectors
        print(res_arr.shape) #replications x threshold x detectors x (real, detected) x chunks-1

        res_clf_mean = np.mean(res_clf, axis=0)
        res_clf_mean_detectors = np.mean(res_clf_mean, axis=1)

        dderror_arr = np.zeros((res_arr.shape[0], res_arr.shape[1], res_arr.shape[2], 3)) # rep x th x det x 3

        for rep in range(res_arr.shape[0]):
            for th in range(res_arr.shape[1]):
                for det in range(res_arr.shape[2]):
                    real_drf = np.argwhere(res_arr[rep, th, det, 0]==2).flatten()
                    det_drf = np.argwhere(res_arr[rep, th, det, 1]==2).flatten()
                    err = dderror(real_drf, det_drf, res_arr.shape[4])
                    dderror_arr[rep,th,det] = err

        res_arr_mean = np.mean(dderror_arr, axis=0) # th x det x 3
        res_arr_mean_detectors = np.mean(res_arr_mean, axis = 1) # th x 3

        """
        Plot
        """
        aa = ax[drf_id, ss_id]
        aa.set_title('subspace size: %i, drift: %s' % (ss, drf))

        vec_0 = res_arr_mean_detectors[:,2]
        vec_1 = res_arr_mean_detectors[:,1]
        vec_2 = res_arr_mean_detectors[:,0]

        colors = [
            '#111111',
            '#555555',
            '#999999'
        ]
        alpha = .75

        aa.fill_between(th_arr, vec_0 * 0, vec_0,
                        color = colors[0], label = 'R',
                        alpha=alpha)
        aa.fill_between(th_arr, vec_0, vec_0 + vec_1,
                        color = colors[1], label = 'D2',
                        alpha=alpha)
        aa.fill_between(th_arr, vec_0 + vec_1, vec_0 + vec_1 + vec_2,
                        color = colors[2], label = 'D1',
                        alpha=alpha)


        aa.plot(th_arr, vec_0,
                color = colors[0])
        aa.plot(th_arr, vec_0 + vec_1,
                color = colors[1])
        aa.plot(th_arr, vec_0 + vec_1 + vec_2,
                color = colors[2])


        """
        aa.plot(th_arr, res_arr_mean_detectors[:,0],
                color = 'red', label = 'closest drift')
        aa.plot(th_arr, res_arr_mean_detectors[:,1],
                color = 'green', label='closest detection')
        aa.plot(th_arr, res_arr_mean_detectors[:,2],
                color = 'blue', label = 'dd-ratio')
        """

        #ax[drf_id, ss_id].plot(1-res_clf_mean_detectors, color = 'black', label = 'clf error')

        print(th_arr)

        #ax[drf_id, ss_id].set_xticks(list(range(len(th_arr))))
        #ax[drf_id, ss_id].set_xticklabels(['%.2f' % v for v in th_arr])
        if drf_id == 2:
            aa.set_xlabel("Sensitivity")

        # ax[drf_id, ss_id].set_ylim(0,35)
        if ss_id == 0:
            aa.set_ylabel("Accumulated error value")

        aa.set_yscale('log')
        aa.set_ylim(.1, 100)
        aa.set_xlim(th_arr[0], th_arr[-1])
        aa.grid(ls=":")
        aa.spines['top'].set_visible(False)
        aa.spines['right'].set_visible(False)

        aa.set_yticks([.1, 1, 10, 100])
        aa.set_yticklabels(['0.1', '1', '10', '100'])



plt.legend(ncol=3, loc=9, frameon=False)
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures_ex1/mean_all.png')
plt.savefig('pub_figures/mean_all.eps')

# plt.close()
# exit()
