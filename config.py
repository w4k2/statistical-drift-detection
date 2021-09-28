import strlearn as sl
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from strlearn.ensembles import SEA
import numpy as np
import hashlib
from methods import DDM, EDDM, ADWIN, ALWAYS, NEVER, Meta, SDDM, KDDDE

def e3_methods():
    return [
        GaussianNB(),
        # MLPClassifier(),
        # SEA(base_estimator = SVC(probability=True)),
        # SEA(base_estimator = DecisionTreeClassifier()),
        # SEA(base_estimator = KNeighborsClassifier()),
    ]

def e3_methods_labels():
    return [
        'GNB',
        # 'MLP',
        # 'SVC',
        # 'DTs',
        # 'kNN'
    ]

def e3_streams(random_state):
    #N_CHUNKS = 500
    #CHUNK_SIZE = 200

    N_CHUNKS = 200
    CHUNK_SIZE = 200

    N_FEATURES = 10
    N_INFORMATIVE = 10
    N_REDUNDANT=0
    N_REPEATED=0
    N_CLUSTERS_PER_CLASS=1

    RECURRING = [True, False]
    CONCEPT_SIGMOID_SPACING = [5., 500., 999.]
    N_DRIFS = [1,3,5,7,9]

    WEIGHTS = [0.5, 0.5]
    Y_FLIP = .01
    N_CLASSES = 2

    streams = {}

    for rec in RECURRING:
        for css in CONCEPT_SIGMOID_SPACING:
            for drifts in N_DRIFS:
                stream = sl.streams.StreamGenerator(
                        recurring = rec,
                        concept_sigmoid_spacing = css,
                        n_drifts=drifts,
                        random_state=random_state,
                        weights=WEIGHTS,
                        y_flip=Y_FLIP,
                        n_features=N_FEATURES,
                        n_informative=N_INFORMATIVE,
                        n_redundant=N_REDUNDANT,
                        n_repeated=N_REPEATED,
                        n_clusters_per_class=N_CLUSTERS_PER_CLASS,
                        n_chunks=N_CHUNKS,
                        chunk_size=CHUNK_SIZE,
                        n_classes = N_CLASSES
                    )
                streams.update({hash(str(stream)): stream})

    return streams

def hash(string):
    return hashlib.md5(string.encode("utf-8")).hexdigest()

def e3_clfs():
    clfs = []
    #DDM
    for m in e3_methods():
        clfs.append(Meta(detector = DDM(), base_clf = m))

    #EDDM
    for m in e3_methods():
        clfs.append(Meta(detector = EDDM(), base_clf = m))

    #ADWIN
    for m in e3_methods():
        clfs.append(Meta(detector = ADWIN(), base_clf = m))

    #SDDM
    for m in e3_methods():
        clfs.append(Meta(detector = SDDM(), base_clf = m))

    #KDDDE
    for m in e3_methods():
        clfs.append(Meta(detector = KDDDE(), base_clf = m))

    #ALWAYS
    for m in e3_methods():
        clfs.append(Meta(detector = ALWAYS(), base_clf = m))

    #NEVER
    for m in e3_methods():
        clfs.append(Meta(detector = NEVER(), base_clf = m))


    return clfs

def e3_clf_names():
    names = []
    detectors = ['DDM', 'EDDM', 'ADWIN', "SDDM", "KDDDE", "ALWAYS", "NEVER"]

    for d in detectors:
        for m in e3_methods_labels():
            names.append(d + "-" + m)

    return names


def metrics():
    return [
        sl.metrics.f1_score,
        sl.metrics.geometric_mean_score_1
        ]

def metrics_names():
    return [
        "F1",
        "G-mean"
        ]
