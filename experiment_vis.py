import numpy as np
import matplotlib.pyplot as plt

def get_real_drift(n_ch, n_d):
    real_drifts = np.linspace(0,n_ch,n_d+1)[:-1]
    real_drifts += (real_drifts[1]/2)
    return real_drifts


deltas = [1, 10, 20, 50]
drifts = [5, 10, 15]
n_epochs_mlp = [1,50,50,50]
n_chunks = 500

frameworks = ['CR',
              'TR-S',
              'TR-U',
              'TR-UR']
clfs = ['MLP', 'HT', 'GNB']
dets = ['Oracle', 'Real']

results = np.load('results.npy')
# reps, deltas, frameworks, classifiers, drifts, detectors, chunks, (bac, detections, trainings)

print(results.shape)
        
rep = 0 
clf_id = 0
dritfs_id = 2
detectors_id = 0

fig, ax = plt.subplots(4,4, figsize=(30,8), sharex=True, sharey=True)

for d_id, d in enumerate(deltas):
    for f_id, f in enumerate(frameworks):
    
        ax[f_id, d_id].vlines(results[rep, d_id, f_id, clf_id, dritfs_id, detectors_id, :, 1], 0, 1, label = 'label requests', color='r', alpha=0.3, ls=':')
        ax[f_id, d_id].vlines(results[rep, d_id, f_id, clf_id, dritfs_id, detectors_id, :, 2],0, 1, label = 'trainings', color='g', alpha=0.3, ls=':')
        ax[f_id, d_id].plot(np.arange(1,500),
            results[rep, d_id, f_id, clf_id, dritfs_id, detectors_id, :, 0], 
            label = 'balanced accuracy', color='b')
                
        ax[f_id, d_id].grid(ls=':')
        ax[f_id, d_id].set_xticks(get_real_drift(n_chunks, drifts[dritfs_id]).astype(int))

        if f_id==0:
            ax[f_id, d_id].set_title('delta = %i' % (d))
        if d_id==0:
            ax[f_id, d_id].set_ylabel(f)

        
ax[-1,-1].legend(frameon=False, ncols=3)
plt.tight_layout()
plt.savefig('foo.png')

