import numpy as np
import matplotlib.pyplot as plt

drange = 150
n_drifts = 5

lss = np.linspace(0,drange,1000)

means = np.zeros((drange-2, drange-2))
stds = np.zeros((drange-2, drange-2))
mins = np.zeros((drange-2, drange-2))
maxs = np.zeros((drange-2, drange-2))

en2 = np.rint(np.linspace(1, drange, drange-1)).astype(int)
print(en2)

for dd2, n_detections in enumerate(en2):
    acc = []
    for i in lss:

        drifts = (np.random.uniform(size=n_drifts)*drange).astype(int)
        detections = (np.random.uniform(size=n_detections)*drange).astype(int)

        ddm = np.abs(drifts[:, np.newaxis] - detections[np.newaxis,:])

        cdri = np.min(ddm, axis=0)
        cdec = np.min(ddm, axis=1)

        ametric = (np.sum(cdec) + np.sum(cdri))/drange

        acc.append(ametric)

    means[dd2] = np.mean(acc)
    stds[dd2] = np.std(acc)
    mins[dd2] = np.min(acc)
    maxs[dd2] = np.max(acc)

    fig, ax = plt.subplots(2,2,figsize=(12, 12))

    ax[0,0].plot(means)
    ax[1,0].plot(stds)
    ax[0,1].plot(mins)
    ax[1,1].plot(maxs)

    ax[0,0].set_title('means')
    ax[1,0].set_title('stds')
    ax[0,1].set_title('mins')
    ax[1,1].set_title('maxs')

    plt.tight_layout()
    plt.savefig('bar.png')
    plt.close()
