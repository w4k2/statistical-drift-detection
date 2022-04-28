import numpy as np
import matplotlib.pyplot as plt
from methods.dderror import dderror

drange = 25

lss = np.linspace(0,drange,1000)

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

        """
        Plot
        """
        fig, ax = plt.subplots(3,4,figsize=(24, 18))
        metrics= ['all', 'd1', 'd2', 'cnt']
        cmaps = ['binary', 'Reds', 'Greens', 'Blues']

        for i in range(4):
            if i==0:
                ax[0,i].imshow(means, cmap=cmaps[i])
                ax[1,i].imshow(stds, cmap=cmaps[i])
                ax[2,i].imshow(maxs, cmap=cmaps[i])

                ax[0,i].set_title('means %s' % metrics[i])
                ax[1,i].set_title('stds %s' % metrics[i])
                ax[2,i].set_title('maxs %s' % metrics[i])
            else:
                ax[0,i].imshow(means[:,:,i-1], cmap=cmaps[i])
                ax[1,i].imshow(stds[:,:,i-1], cmap=cmaps[i])
                ax[2,i].imshow(maxs[:,:,i-1], cmap=cmaps[i])

                ax[0,i].set_title('means %s' % metrics[i])
                ax[1,i].set_title('stds %s' % metrics[i])
                ax[2,i].set_title('maxs %s' % metrics[i])

                for k in range(len(en1)):
                    for j in range(len(en2)):
                        if k%5 == 1:
                            if j%5==1:
                                ax[0,i].text(j,k, '%.2f' % means[k,j,i-1], ha='center',
                                        va='center')
                                ax[1,i].text(j,k, '%.2f' % stds[k,j,i-1], ha='center',
                                        va='center')
                                ax[2,i].text(j,k, '%.2f' % maxs[k,j,i-1], ha='center',
                                        va='center')
                                # ax[1,1].text(j,k, '%.2f' % maxs[k,j,i], ha='center',
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
