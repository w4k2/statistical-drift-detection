import numpy as np
import matplotlib.pyplot as plt
from methods import dderror
import e1_config

subspace_sizes = e1_config.e1_subspace_sizes()
drf_types = e1_config.e1_drift_types()
th_arr = e1_config.e1_drf_threshold()
det_arr = e1_config.e1_n_detectors()

"""
Classification
"""
fig, ax = plt.subplots(3, 3, figsize=(5, 5),
                       sharex=True,
                       sharey=True)

# fig.suptitle("Classification score")

addr_a = np.linspace(0, len(th_arr)-1, 5)
addr_b = np.linspace(0, len(det_arr)-1, 5)

val_a = ['%.2f' % v for v in np.linspace(0, 1, 5)]
val_b = ['%.0f' % v for v in np.linspace(1, 100, 5)]

for ss_id, ss in enumerate(subspace_sizes):
    for drf_id, drf in enumerate(drf_types):
        res_clf = np.load('results_ex1/clf_15feat_5drifts_%s_%isubspace_size.npy' %  (drf, ss))

        res_clf_mean = np.mean(res_clf, axis=0)

        aa = ax[drf_id, ss_id]

        aa.imshow(res_clf_mean, cmap='binary', origin='lower',
                  vmin=.5, vmax=1)

        aa.set_yticks(addr_a)
        aa.set_yticklabels(val_a, fontsize=8)

        aa.set_xticks(addr_b)
        aa.set_xticklabels(val_b, fontsize=8)

        aa.grid(ls=":")
        [aa.spines[spine].set_visible(False)
         for spine in ['top', 'bottom', 'left', 'right']]

        if drf_id == 0:
            ax[drf_id, ss_id].set_title("subspace %i" % (ss), fontsize=10)
        if ss_id == 0:
            ax[drf_id, ss_id].set_ylabel("%s drift\nsensitivity" % drf, fontsize=10)
        if drf_id == len(drf_types)-1:
            ax[drf_id, ss_id].set_xlabel("#detectors", fontsize=10)


plt.tight_layout()
# fig.subplots_adjust(top=0.93)
plt.savefig('figures_ex1/clf.png')
plt.savefig('pub_figures/clf.eps')
plt.close()
#exit()

"""
Error
"""
all_errors = np.zeros((len(subspace_sizes), len(drf_types), len(th_arr), len(det_arr), 3))
vmins=[[],[],[]]
vmaxs=[[],[],[]]
for ss_id, ss in enumerate(subspace_sizes):
    for drf_id, drf in enumerate(drf_types):

        res_arr = np.load('results_ex1/drf_arr_15feat_5drifts_%s_%isubspace_size.npy' %  (drf, ss))
        print(res_arr.shape) #replications x threshold x detectors x (real, detected) x chunks-1

        dderror_arr = np.zeros((res_arr.shape[0], res_arr.shape[1], res_arr.shape[2], 3))

        for rep in range(res_arr.shape[0]):
            for th in range(res_arr.shape[1]):
                for det in range(res_arr.shape[2]):
                    real_drf = np.argwhere(res_arr[rep, th, det, 0]==2).flatten()
                    det_drf = np.argwhere(res_arr[rep, th, det, 1]==2).flatten()
                    err = dderror(real_drf, det_drf, res_arr.shape[4])
                    dderror_arr[rep,th,det] = err

        res_arr_mean = np.mean(dderror_arr, axis=0)
        all_errors[ss_id, drf_id] = res_arr_mean

        for i in range(3):
            vmins[i].append(np.min(res_arr_mean[:,:,i]))
            vmaxs[i].append(np.max(res_arr_mean[:,:,i]))

for ss_id, ss in enumerate(subspace_sizes):
    for drf_id, drf in enumerate(drf_types):

        for i in range(3):
            all_errors[ss_id, drf_id, :,:,i] -= np.min(vmins[i])
            all_errors[ss_id, drf_id, :,:,i] /= (np.max(vmaxs[i]) - np.min(vmins[i]))
        # all_errors[ss_id, drf_id] = 1 - all_errors[ss_id, drf_id]

vmax_global = np.max(all_errors)
vmin_global = np.min(all_errors)

print(vmax_global, vmin_global)

"""
Second plot
"""

fig, ax = plt.subplots(3, 3, figsize=(5, 5),
                       sharex=True,
                       sharey=True)

# fig.suptitle("Detection error")

for ss_id, ss in enumerate(subspace_sizes):
    for drf_id, drf in enumerate(drf_types):


        aa = ax[drf_id, ss_id]

        im = aa.imshow(all_errors[ss_id, drf_id], cmap='binary', origin='lower')

        #ax[drf_id,ss_id].set_yticks(list(range(len(th_arr))))
        #ax[drf_id,ss_id].set_yticklabels(['%.2f' % v for v in th_arr])
        #ax[drf_id,ss_id].set_ylabel("Threshold")

        #ax[drf_id,ss_id].set_xticks(list(range(len(det_arr))))
        #ax[drf_id,ss_id].set_xticklabels(det_arr)
        #ax[drf_id,ss_id].set_xlabel("n detectors")

        #ax[drf_id,ss_id].set_title("%s %i ss" % (drf, ss))

        aa.set_yticks(addr_a)
        aa.set_yticklabels(val_a, fontsize=8)

        aa.set_xticks(addr_b)
        aa.set_xticklabels(val_b, fontsize=8)

        aa.grid(ls=":")
        [aa.spines[spine].set_visible(False)
         for spine in ['top', 'bottom', 'left', 'right']]

        if drf_id == 0:
            ax[drf_id, ss_id].set_title("subspace size: %i" % (ss), fontsize=10)
        if ss_id == 0:
            ax[drf_id, ss_id].set_ylabel("%s drift\nsensitivity" % drf, fontsize=10)
        if drf_id == len(drf_types)-1:
            ax[drf_id, ss_id].set_xlabel("#detectors", fontsize=10)


# fig.subplots_adjust(right=1.6)

# cbar_ax1 = fig.add_axes([0.9, 0.15, 0.02, 0.7])
# cbar_ax2 = fig.add_axes([0.925, 0.15, 0.02, 0.7])
# cbar_ax3 = fig.add_axes([0.95, 0.15, 0.02, 0.7])

# fig.colorbar(im, cax=cbar_ax1)
# fig.colorbar(im, cax=cbar_ax2)
# fig.colorbar(im, cax=cbar_ax3)


plt.tight_layout()
#fig.subplots_adjust(top=0.93)
plt.savefig('figures_ex1/err_rgb.png')
plt.savefig('pub_figures/err_rgb.eps')


"""
Third plot
"""

fig, ax = plt.subplots(3, 3, figsize=(8, 8),
                       sharex=True,
                       sharey=True)

# fig.suptitle("Detection error")

for ss_id, ss in enumerate(subspace_sizes):
    for drf_id, drf in enumerate(drf_types):

        res_clf = np.load('results_ex1/clf_15feat_5drifts_%s_%isubspace_size.npy' %  (drf, ss))

        img_mono = np.mean(res_clf, axis=0)
        img_mono -= .5
        img_mono *= 2
        img_mono = 1 - img_mono
        img_poly = all_errors[ss_id, drf_id]

        print(img_mono.shape, img_poly.shape)

        img_mix = img_poly * img_mono[:,:, np.newaxis]

        img_mix = np.mean(img_mix, axis=2)

        print(np.min(img_mono), np.max(img_mono))

        aa = ax[drf_id, ss_id]
        im = aa.imshow(img_mix, origin='lower', cmap='binary_r')

        #ax[drf_id,ss_id].set_yticks(list(range(len(th_arr))))
        #ax[drf_id,ss_id].set_yticklabels(['%.2f' % v for v in th_arr])
        #ax[drf_id,ss_id].set_ylabel("Threshold")

        #ax[drf_id,ss_id].set_xticks(list(range(len(det_arr))))
        #ax[drf_id,ss_id].set_xticklabels(det_arr)
        #ax[drf_id,ss_id].set_xlabel("n detectors")

        #ax[drf_id,ss_id].set_title("%s %i ss" % (drf, ss))

        aa.set_yticks(addr_a)
        aa.set_yticklabels(val_a, fontsize=8)

        aa.set_xticks(addr_b)
        aa.set_xticklabels(val_b, fontsize=8)

        aa.grid(ls=":")
        [aa.spines[spine].set_visible(False)
         for spine in ['top', 'bottom', 'left', 'right']]

        if drf_id == 0:
            ax[drf_id, ss_id].set_title("subspace size: %i" % (ss), fontsize=10)
        if ss_id == 0:
            ax[drf_id, ss_id].set_ylabel("%s drift\nsensitivity" % drf, fontsize=10)
        if drf_id == len(drf_types)-1:
            ax[drf_id, ss_id].set_xlabel("#detectors", fontsize=10)


# fig.subplots_adjust(right=1.6)

# cbar_ax1 = fig.add_axes([0.9, 0.15, 0.02, 0.7])
# cbar_ax2 = fig.add_axes([0.925, 0.15, 0.02, 0.7])
# cbar_ax3 = fig.add_axes([0.95, 0.15, 0.02, 0.7])

# fig.colorbar(im, cax=cbar_ax1)
# fig.colorbar(im, cax=cbar_ax2)
# fig.colorbar(im, cax=cbar_ax3)


plt.tight_layout()
#fig.subplots_adjust(top=0.93)
plt.savefig('foo.png')
plt.savefig('figures_ex1/clf_err_rgb.png')
plt.savefig('pub_figures/clf_err_rgb.eps')
