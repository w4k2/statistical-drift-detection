from matplotlib import cm
import numpy as np

def dderror(drifts, detections, n_chunks):

    if len(detections) == 0: # no detections
        detections = np.arange(n_chunks)

    n_detections = len(detections)
    n_drifts = len(drifts)

    ddm = np.abs(drifts[:, np.newaxis] - detections[np.newaxis,:])

    cdri = np.min(ddm, axis=0)
    cdec = np.min(ddm, axis=1)

    d1metric = np.mean(cdri)
    d2metric = np.mean(cdec)
    cmetric = np.abs((n_drifts/n_detections)-1)

    return d1metric, d2metric, cmetric
    # d1 - detection from nearest drift
    # d2 - drift from nearest detection