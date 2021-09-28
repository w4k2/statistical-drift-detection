import numpy as np

def e1_subspace_sizes():
    return [1,2,3]

def e1_n_detectors():
    return np.linspace(10,20,11)

def e1_n_detectors2():
    return np.linspace(20,100,11)

def e1_drf_threshold():
    return np.linspace(0.0,1.0,11)

def e1_replications():
    return 5

def e1_static():
    return {
        'n_drifts': 5,
        'n_chunks': 100,
        'chunk_size': 200,
        'n_features': 15,
        'n_informative': 15,
        'n_redundant': 0,
        'recurring': False
    }

def e1_drift_types():
    return {
    'sudden': {},
    'gradual': {
    'concept_sigmoid_spacing': 5
        },
    'incremental': {
    'concept_sigmoid_spacing': 5,
    'incremental': True
        },
    }
