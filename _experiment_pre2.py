"""
nic z tego nie będzie
"""


from strlearn.streams import StreamGenerator
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from detectors.CDDD import CentroidDistanceDriftDetector
from detectors.OCDD import OneClassDriftDetector
from detectors.adwin import ADWIN
from detectors.ddm import DDM 

np.random.seed(18382)

n_reps = 10
rs = np.random.randint(10,100000,n_reps).astype(int)

n_chunks = 100
chunk_size = 500

dets_supp = np.zeros((n_reps, n_chunks))
dets_unsupp = np.zeros((n_reps, n_chunks))

for n in tqdm(range(n_reps)):
    stream = StreamGenerator(n_chunks=n_chunks,
                             n_features=20,
                         chunk_size=chunk_size,
                         random_state=rs[n],
                         n_drifts=1,
                         concept_sigmoid_spacing=5,
                         incremental=True)

    clf = GaussianNB()
    
    det_supp = DDM(drift_lvl=1., skip=10)
    det_unsupp = OneClassDriftDetector(size = 500, 
                                       dim = 20, 
                                       percent = 0.7, 
                                       nu=0.5)
    
    for c in range(n_chunks):
        X, y = stream.get_chunk()
        
        if c>0:
            pred = clf.predict(X)
            det_supp.process(X, y, pred)
                
        clf.partial_fit(X, y, np.unique(y))
        det_unsupp.process(X)

        dets_supp[n, c] = det_supp._is_drift
        if det_supp._is_drift:
            clf = GaussianNB()
            clf.partial_fit(X, y, np.unique(y))
        dets_unsupp[n, c] = det_unsupp._is_drift
        
np.save('preliminary/res_pre2_s.npy', dets_supp)
np.save('preliminary/res_pre2_uns.npy', dets_unsupp)

fig, ax = plt.subplots(1,1,figsize=(8,4), sharex=True, sharey=True)

idx_supp = np.argwhere(dets_supp>0)
idx_unsupp = np.argwhere(dets_unsupp>0)

ax.scatter(idx_supp[:,1], idx_supp[:,0], c='red', alpha=0.5)
ax.scatter(idx_unsupp[:,1], idx_unsupp[:,0], c='blue', alpha=0.5)

ax2 = ax.twinx()
ax2.plot(stream.concept_probabilities[::chunk_size], color='black', ls=':')

ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_yticks([])

ax.grid(ls=':')
ax.set_xlim(0,100)

ax.set_xlabel('chunk')
ax.set_ylabel('replication')

handles, labels = plt.gca().get_legend_handles_labels()
line1 = Line2D([0], [0], label='supervised', color='red', 
               marker='o', alpha=0.5, linestyle='')
line2 = Line2D([0], [0], label='unsupervised', color='blue',
                marker='o', alpha=0.5, linestyle='')
line3 = Line2D([0], [0], label='concept probability', color='black', ls=':')
handles.extend([line1, line2, line3])
plt.legend(handles=handles, frameon=False)


plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('preliminary/pre_acc.png')
