import numpy as np

def find_real_drift(chunks, drifts):
    interval = round(chunks/drifts)
    return [interval*(i+.5) for i in range(drifts)]

def find_nearest(value, array):
    dist = (np.abs(array-value)).min()
    return dist

def dderror(chunks, drifts, detected_drift):
    if(len(detected_drift))==0:
        return np.inf
    real_drift = find_real_drift(chunks, drifts)

    # distances from real
    d_real = []
    for d in detected_drift:
        d_real.append(find_nearest(d, real_drift))

    # distances from detected
    d_detected = []
    for r in real_drift:
        d_detected.append(find_nearest(r, detected_drift))

    return (np.sum(d_detected) + np.sum(d_real))/len(real_drift)
