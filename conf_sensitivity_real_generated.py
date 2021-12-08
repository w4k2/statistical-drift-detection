import strlearn as sl
import numpy as np
from sklearn.base import clone
import e2_config
from tqdm import tqdm
from methods import SDDE, Meta
from sklearn.naive_bayes import GaussianNB
import os

def find_real_drift(chunks, drifts):
    interval = round(chunks/drifts)
    arr = np.zeros((chunks))
    idx = [interval*(i+.5) for i in range(drifts)]
    for i in idx:
        arr[int(i)]=2
    return arr[1:]

np.random.seed(654)

replications = 10
random_states = np.random.randint(0, 10000, replications)
recurring = e2_config.e2_recurring()

base_detectors = [
    Meta(detector = SDDE(sensitivity = 0.3), base_clf = GaussianNB()),
    # Meta(detector = SDDE(sensitivity = 0.35), base_clf = GaussianNB()),
    # Meta(detector = SDDE(sensitivity = 0.4), base_clf = GaussianNB()),
    # Meta(detector = SDDE(sensitivity = 0.45), base_clf = GaussianNB()),
    # Meta(detector = SDDE(sensitivity = 0.5), base_clf = GaussianNB()),
    # Meta(detector = SDDE(sensitivity = 0.55), base_clf = GaussianNB()),
    # Meta(detector = SDDE(sensitivity = 0.6), base_clf = GaussianNB()),
]
metrics = e2_config.metrics()

directory = 'real-gen-streams'
for _,_,files in os.walk(directory):
    print(files)

chunk_size = 250
n_chunks = 400

pbar = tqdm(total=replications*len(files))


for i, filepath in enumerate(files):
    # print(filepath)
    drifts = int(filepath.split('_')[3][0])
    # print(drifts)
    drf_type= filepath.split('_')[2]
    # print(drf_type)

    real_drf = find_real_drift(n_chunks, drifts)


    results_clf = np.zeros((replications, len(base_detectors), n_chunks-1))
    results_drf_arrs = np.zeros((replications, len(base_detectors), 2, n_chunks-1))
    # replications x detectors x (real_drf, detected_drf) x chunks

    for replication in range(replications):

        detectors=[]
        for det_id in range(len(base_detectors)):
            detectors.append(clone(base_detectors[det_id]))

        stream = sl.streams.NPYParser('%s/%s' % (directory, filepath), chunk_size=chunk_size, n_chunks=n_chunks)

        print("replication: %i, stream: %s" % (replication, filepath))

        eval = sl.evaluators.TestThenTrain(metrics=metrics)
        eval.process(stream, detectors)

        scores = eval.scores
        results_clf[replication] = scores[:,:,0]

        for det_id in range(len(detectors)):
            results_drf_arrs[replication, det_id, 0] = real_drf
            results_drf_arrs[replication, det_id, 1] = np.array(detectors[det_id].detector.confidence)
            print(np.array(detectors[det_id].detector.confidence))

        pbar.update(1)

    np.save('results_ex2_2/gen_clf_%s' % filepath, results_clf)
    np.save('results_ex2_2/gen_conf_arr_%s' % filepath, results_drf_arrs)

pbar.close()