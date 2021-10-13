import numpy as np
import matplotlib.pyplot as plt
from methods.dderror import dderror

drange = 25

lss = np.linspace(0,drange,100)

means = np.zeros((drange-2, drange-2, 3))
stds = np.zeros((drange-2, drange-2, 3))
mins = np.zeros((drange-2, drange-2, 3))
maxs = np.zeros((drange-2, drange-2, 3))

en1 = np.rint(np.linspace(2,drange, drange-2)).astype(int)
en2 = np.rint(np.linspace(2,drange, drange-2)).astype(int)
for dd1, n_drifts in enumerate(en1):
    for dd2, n_detections in enumerate(en2):
        acc = []
        for i in lss:

            if len(acc)==0:
                drifts = (np.random.uniform(size=n_drifts)*drange).astype(int)
                detections = drifts
            # elif len(acc)==1:
            #     drifts = (np.random.uniform(size=n_drifts)*drange).astype(int)
            #     detections = []
            # elif len(acc)==2:
            #     drifts = (np.random.uniform(size=n_drifts)*drange).astype(int)
            #     detections = np.arange(drange)
            else:
                drifts = (np.random.uniform(size=n_drifts)*drange).astype(int)
                detections = (np.random.uniform(size=n_detections)*drange).astype(int)

            ametric = dderror(drifts, detections, drange)

            acc.append(ametric)

        means[dd1, dd2] = np.mean(acc, axis=0)
        stds[dd1, dd2] = np.std(acc, axis=0)
        mins[dd1, dd2] = np.min(acc, axis=0)
        maxs[dd1, dd2] = np.max(acc, axis=0)

        
        means[dd1, dd2] -= np.min(means[dd1, dd2])
        means[dd1, dd2] /= np.max(means[dd1, dd2])

        stds[dd1, dd2] -= np.min(stds[dd1, dd2])
        stds[dd1, dd2] /= np.max(stds[dd1, dd2])

        mins[dd1, dd2] -= np.min(mins[dd1, dd2])
        mins[dd1, dd2] /= np.max(mins[dd1, dd2])

        maxs[dd1, dd2] -= np.min(maxs[dd1, dd2])
        maxs[dd1, dd2] /= np.max(maxs[dd1, dd2])

        fig, ax = plt.subplots(2,2,figsize=(12, 12))

        ax[0,0].imshow(means)
        ax[1,0].imshow(stds)
        ax[0,1].imshow(mins)
        ax[1,1].imshow(maxs)

        ax[0,0].set_title('means')
        ax[1,0].set_title('stds')
        ax[0,1].set_title('mins')
        ax[1,1].set_title('maxs')

        # for i in range(len(en1)):
        #     for j in range(len(en2)):
        #         if i%5 == 1:
        #             if j%5==1:

                        # ax[0,0].text(j,i, '%.2f' % means[i,j,0], ha='center',
                        #         va='center')
                        # ax[1,0].text(j,i, '%.2f' % stds[i,j], ha='center',
                        #         va='center')
                        # ax[0,1].text(j,i, '%.2f' % mins[i,j], ha='center',
                        #         va='center')
                        # ax[1,1].text(j,i, '%.2f' % maxs[i,j], ha='center',
                        #         va='center')

        #aaa.set_xticks([])
        #aaa.set_yticks([np.max(np.array(acc))])
        #aaa.set_yticklabels(['%.1f' % np.max(np.array(acc))])

        #aaa.set_xlabel('dri %i' % n_drifts)
        #aaa.set_ylabel('det %i' % n_detections)
        #aaa.set_title('%.2f | %.2f' % (np.nanmean(acc), np.nanstd(acc)))

        plt.tight_layout()
        plt.savefig('bar1.png')
        plt.close()
