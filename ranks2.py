import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from cd_utils import compute_CD, graph_ranks


detector_names = ['DDM', 'EDDM', 'ADWIN', 'SDDE',
                  ' HDDM_W', 'HDDM_A', 'ALWAYS', 'NEVER']
drf_type = ["sudden", "gradual", "incremental"]
recs = ["reccuring", "not-recurring"]
metric_names = ["Accuracy", "d1", "d2", "cnt_ratio"]

# 10, 2, 3, 3, 3, 8, 4
# reps x rec x drf type x features x drf num x detectors x metrics
scores = np.load("results_2.npy")
print(scores.shape)

for metric_id, metric in enumerate(metric_names):
    print("%s" % metric)
    print("––––––––––––––––––")
    # reps x rec x drf type x features x drf num x detectors
    metric_scores = scores[:, :, :, :, :, [0, 1, 2, 3, 4, 5], metric_id]
    for rec_id, rec in enumerate(recs):
        # reps x drf type x features x drf num x detectors
        rec_scores = metric_scores[:, rec_id]
        for drf_id, drf in enumerate(drf_type):
            # reps x features x drf num x detectors
            drf_scores = rec_scores[:, drf_id]
            drf_scores = drf_scores.reshape(
                (drf_scores.shape[0]*drf_scores.shape[1]*drf_scores.shape[2], drf_scores.shape[3]))
            # print(drf_scores, drf_scores.shape)
            if metric_id != 0:
                drf_scores = drf_scores * -1
            ranks = []
            for row in drf_scores:
                ranks.append(rankdata(row).tolist())
            ranks = np.array(ranks)
            av_ranks = np.mean(ranks, axis=0)
            cd = compute_CD(av_ranks, 90)
            graph_ranks(av_ranks, detector_names,
                        cd=cd, width=6, textspace=1.5)
            plt.savefig("figures_cd/exp2/%s_%s_%s.png" % (metric, rec, drf))
            plt.savefig("figures_cd/exp2/%s_%s_%s.eps" % (metric, rec, drf))
            plt.close()
