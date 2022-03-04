import strlearn as sl
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from strlearn.ensembles import SEA
import numpy as np
import hashlib
from methods import DDM, EDDM, ADWIN, ALWAYS, NEVER, Meta, SDDM, SDDE, HDDM_AA, HDDM_WW

def e2_methods():
    return [
        GaussianNB(),
        # MLPClassifier(),
        # SEA(base_estimator = SVC(probability=True)),
        # SEA(base_estimator = DecisionTreeClassifier()),
        # SEA(base_estimator = KNeighborsClassifier()),
    ]

def e2_methods_labels():
    return [
        'GNB',
        # 'MLP',
        # 'SVC',
        # 'DTs',
        # 'kNN'
    ]

def metrics():
    return [
        sl.metrics.balanced_accuracy_score
        ]

def metrics_names():
    return [
        "BAC"
        ]

def e2_replications():
    return 10

def e2_static():
    return {
        'n_drifts': 5,
        'n_chunks': 200,
        'chunk_size': 250,
        'n_features': 15,
        'n_informative': 15,
        'n_redundant': 0
        }

def e2_static2():
    return {
        # 'n_drifts': 5,
        'n_chunks': 200,
        'chunk_size': 250,
        # 'n_features': 15,
        # 'n_informative': 15,
        'n_redundant': 0
        }

def e2_drift_types():
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

def e2_recurring():
    return {
        'recurring': {
            'recurring': True
        },
        'not-recurring': {}
    }

def e2_n_drifts():
    return {
        # 1: { 'n_drifts': 1},
        3: { 'n_drifts': 3},
        5: { 'n_drifts': 5},
        7: { 'n_drifts': 7},
        # 9: { 'n_drifts': 9},
    }

def e2_n_features():
    return {
        10: { 'n_features': 10, 'n_informative': 10},
        15: { 'n_features': 15, 'n_informative': 15},
        20: { 'n_features': 20, 'n_informative': 20},
        # 25: { 'n_features': 25, 'n_informative': 25},
        # 30: { 'n_features': 30, 'n_informative': 30},
    }

def e2_clfs(sdde_n_det=0, sdde_sensitivity=0):
    clfs = []

    # HDDM_W
    for m in e2_methods():
        clfs.append(Meta(detector = HDDM_WW(), base_clf = m))

    # HDDM_A
    for m in e2_methods():
        clfs.append(Meta(detector = HDDM_AA(), base_clf = m))



    return clfs
