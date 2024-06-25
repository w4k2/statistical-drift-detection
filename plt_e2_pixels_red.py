"""
Plot detection moments of all replications and all methods (exp 2 synthetic)
"""
import e2_config
from e2_config_hddm import e2_n_drifts, e2_n_features
import numpy as np
import matplotlib.pyplot as plt
from strlearn.streams import StreamGenerator

drf_types = e2_config.e2_drift_types()
recurring = e2_config.e2_recurring()
detectors = ['DDM', 'EDDM', "ADWIN", 'SDDE', 'HDDM_W', 'HDDM_A']

drifts_n = e2_n_drifts()
features_n = e2_n_features()

res_all = np.zeros((len(recurring), len(drf_types), len(features_n), len(drifts_n), 10, 8, 2, 199))
conceptsss = np.zeros((len(recurring), len(drf_types), len(features_n), len(drifts_n), 200*250))
# rec, drf_types, features, n_drifts, reps, detector x(real, detected), chunks

cps = np.linspace(0, 200, 200*250)

for rec_id, rec in enumerate(recurring):
    for drf_id, drf_type in enumerate(drf_types):
        for f_id, f in enumerate(features_n):
            for d_id, d in enumerate(drifts_n):
                res_arr = np.load('results_ex2_d_f_45/drf_arr_%ifeat_%idrifts_%s_%s_all.npy' %(f, d, drf_type, rec))
                res_all[rec_id,drf_id,f_id,d_id] = res_arr
                
                
                sc = {'n_features':1, 'n_informative':1,'n_clusters_per_class':1, 'n_redundant':0 }
                s = StreamGenerator(**sc, **recurring[rec], **drf_types[drf_type], **drifts_n[d])
                s._make_classification()
                conceptsss[rec_id,drf_id,f_id,d_id] = (s.concept_probabilities)


print(res_all.shape)
res_all = res_all[:,:,:,:,:,[0,1,2,3,6,7]]

# rec, drf_types, features, n_drifts, reps, detector x(real, detected), chunks
for rec_id, rec in enumerate(recurring):
    for drf_id, drf_type in enumerate(drf_types):
        
        fig, ax = plt.subplots(len(drifts_n), len(features_n), figsize=(12, 9), sharex=True, sharey=True)
        fig.suptitle("%s | %s" % (rec if rec!='not-recurring' else 'non-recurring', drf_type), fontsize=12)

        for f_id, f in enumerate(features_n):
            for d_id, d in enumerate(drifts_n):
                
                aa = ax[d_id, f_id]
                r = res_all[rec_id, drf_id, f_id, d_id,:,:,1]
                print(r.shape)
                for det in range(len(detectors)):
                    print(det, np.unique(res_all[rec_id, drf_id, f_id, d_id,:,det,1], return_counts=True))

                    for rep in range(10):
                        step = 1
                        start = det*10 + step*rep
                        stop = det*10 + step*(rep+1)
                        detections = np.argwhere(r[rep,det]>1).flatten()
                        
                        aa.vlines(detections, start, stop, color='black' if det != 3 else 'red')
                        
                aa.plot(cps, conceptsss[rec_id,drf_id,f_id,d_id]*5-7.5, c='red')
                aa.grid(ls=":")
            
                if d_id==0:          
                    aa.set_title('%i dim' % (f))
                if f_id==0:
                    aa.set_ylabel('%s drifts' % (d), fontsize=12)
                
                drfs = np.argwhere(res_all[rec_id, drf_id, f_id, d_id,0,0,0]>0).flatten()
                
                aa.set_xticks(drfs, ['D%i' % i for i in range(len(drfs))])
                aa.set_yticks([(10*i)-5 
                            for i in range(len(detectors)+1)], 
                            ['concept']+detectors)
                aa.spines['top'].set_visible(False)
                aa.spines['right'].set_visible(False)
                aa.spines['bottom'].set_visible(False)
                
            plt.tight_layout()
            plt.savefig('foo.png')
            plt.savefig("figures_ex2/d_f_pix_%s_%s.png" % (rec, drf_type))
            plt.savefig("pub_figures/d_f_pix_%s_%s.eps" % (rec, drf_type))
        
            # exit()