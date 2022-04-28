"""
Experiment 2 -- comparison of DD methods for semi-synthetic streams
(with 3,5,7 drits and 10,15,20 features)
"""

import os
import numpy as np
import strlearn as sl
import e2_config_hddm as e2_config
from tqdm import tqdm

def find_real_drift(chunks, drifts):
    interval = round(chunks/drifts)
    arr = np.zeros((chunks))
    idx = [interval*(i+.5) for i in range(drifts)]
    for i in idx:
        arr[int(i)]=2
    return arr[1:]

n_features = 15
chunk_size = 250
n_chunks = 400

print(chunk_size*n_chunks)

metrics = e2_config.metrics()
replications = e2_config.e2_replications()
base_detectors = e2_config.e2_clfs()

directory = 'real-gen-streams'

for _,_,files in os.walk(directory):
    print(files)

pbar = tqdm(total=replications*len(files))

for i, filepath in enumerate(files):
    print(filepath)
    # print(stream.get_chunk())
    # exit()
    drifts = int(filepath.split('_')[3][0])
    print(drifts)

    results_clf = np.zeros((replications, len(base_detectors), n_chunks-1))
    results_drf_arrs = np.zeros((replications, len(base_detectors), 2, n_chunks-1))
    # replications x detectors x (real_drf, detected_drf) x chunks

    for replication in range(replications):
        stream = sl.streams.NPYParser('%s/%s' % (directory, filepath), chunk_size=chunk_size, n_chunks=n_chunks)

        detectors = e2_config.e2_clfs(n_features, .35)

        str_name = filepath

        print("replication: %i, stream: %s" % (replication, str_name))

        eval = sl.evaluators.TestThenTrain(metrics=metrics)
        # print(stream.get_chunk())
        # exit()
        eval.process(stream, detectors)

        scores = eval.scores
        results_clf[replication] = scores[:,:,0]

        for det_id in range(len(detectors)):
            results_drf_arrs[replication, det_id, 0] = find_real_drift(n_chunks, drifts)
            results_drf_arrs[replication, det_id, 1] = np.array(detectors[det_id].detector.drift)

        pbar.update(1)

    np.save('results_ex2_real/clf_%s_hddm' % str_name, results_clf)
    np.save('results_ex2_real/drf_arr_%s_hddm' % str_name, results_drf_arrs)

pbar.close()
