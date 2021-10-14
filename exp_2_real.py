from scipy.io import arff
import os 
import numpy as np
import pandas as pd
import strlearn as sl
from strlearn.streams import ARFFParser
import e2_config
from tqdm import tqdm


filepaths = [
    "real_streams/covtypeNorm-1-2vsAll-pruned.arff",
    "real_streams/INSECTS-abrupt_imbalanced_norm_5prc.arff",
    "real_streams/INSECTS-abrupt_imbalanced_norm.arff",
    "real_streams/INSECTS-gradual_imbalanced_norm_5prc.arff",
    "real_streams/INSECTS-gradual_imbalanced_norm.arff",
    "real_streams/INSECTS-incremental_imbalanced_norm_5prc.arff",
    "real_streams/INSECTS-incremental_imbalanced_norm.arff",
    "real_streams/poker-lsn-1-2vsAll-pruned.arff"
]

n_features = [54,33,33,33,33,33,33,10]
n_chunks = [265, 300, 300, 100, 100, 380, 380, 359]
chunk_size = 1000

metrics = e2_config.metrics()
replications = 1#e2_config.e2_replications()
base_detectors = e2_config.e2_clfs(1)

pbar = tqdm(total=replications*len(filepaths))

for i, filepath in enumerate(filepaths):
    stream = ARFFParser(filepath, chunk_size=chunk_size, n_chunks=n_chunks[i])
 
    results_clf = np.zeros((replications, len(base_detectors), n_chunks[i]-1))
    results_drf_arrs = np.zeros((replications, len(base_detectors), 2, n_chunks[i]-1))
    # replications x detectors x (real_drf, detected_drf) x chunks

    for replication in range(replications):

        detectors = e2_config.e2_clfs(n_features[i])

        str_name = filepath.split("/")[1]
        
        print("replication: %i, stream: %s" % (replication, str_name))

        eval = sl.evaluators.TestThenTrain(metrics=metrics)
        eval.process(stream, detectors)

        scores = eval.scores
        results_clf[replication] = scores[:,:,0]

        for det_id in range(len(detectors)):
            results_drf_arrs[replication, det_id, 1] = np.array(detectors[det_id].detector.drift)

        pbar.update(1)

    np.save('results_ex2_real/clf_%s' % str_name, results_clf)
    np.save('results_ex2_real/drf_arr_%s' % str_name, results_drf_arrs)

pbar.close()
