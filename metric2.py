import numpy as np
import matplotlib.pyplot as plt
from methods.dderror import dderror

drange = 150
n_drifts = 75
n_guesses = 150

means = []
stds = []
mins = []
maxs = []

zakres_liczby_detekcji = np.rint(np.linspace(1, drange, n_guesses)).astype(int)
print(zakres_liczby_detekcji)

for dd2, n_detections in enumerate(zakres_liczby_detekcji):

    acc = []
    for i in range(n_guesses):

        drifts = (np.random.uniform(size=n_drifts)*drange).astype(int)
        detections = (np.random.uniform(size=n_detections)*drange).astype(int)

        ametric = dderror(drifts, detections, drange)

        # ddm = np.abs(drifts[:, np.newaxis] - detections[np.newaxis,:])

        # cdri = np.min(ddm, axis=0)
        # cdec = np.min(ddm, axis=1)

        # ametric = (np.sum(cdec) + np.sum(cdri))/drange

        acc.append(ametric)

    means.append(np.mean(acc))
    stds.append(np.std(acc))
    mins.append(np.min(acc))
    maxs.append(np.max(acc))

    fig, ax = plt.subplots(2,2,figsize=(6, 6))

    ax[0,0].plot(means)
    ax[0,0].set_xticks(np.arange(0,150,n_drifts))
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
