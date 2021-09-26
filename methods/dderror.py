import numpy as np

def dderror(drifts, detections, n_chunks):

    if len(detections) == 0: # brak wykrytych dryfów
        detections = np.arange(n_chunks)

    ddm = np.abs(drifts[:, np.newaxis] - detections[np.newaxis,:])

    cdri = np.min(ddm, axis=0)
    cdec = np.min(ddm, axis=1)

    ametric = (np.sum(cdec) + np.sum(cdri))/n_chunks

    return ametric