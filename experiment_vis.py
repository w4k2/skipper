import numpy as np
import matplotlib.pyplot as plt

def get_real_drift(n_ch, n_d):
    real_drifts = np.linspace(0,n_ch,n_d+1)[:-1]
    real_drifts += (real_drifts[1]/2)
    return real_drifts


deltas = [1, 10, 20, 60]
drifts = [5, 10, 15]
n_epochs_mlp = [1,50,50,50]
n_chunks = 500

frameworks = ['CR',
              'TR-S',
              'TR-U',
              'TR-P']
clfs = ['MLP', 'HT', 'GNB']
dets = ['Oracle', 'Real']

results = np.load('results/results.npy')
# reps, deltas, frameworks, classifiers, drifts, detectors, chunks, (bac, detections, trainings)

print(results.shape)
        
mag = 2.25

for rep in range(3):
    for clf_id, clf in enumerate(clfs):
        for n_d_id, n_d in enumerate(drifts):
            for det_id, det in enumerate(dets):

                fig, ax = plt.subplots(4,4, figsize=(4.5*mag,mag*4.5/1.618), sharex=True, sharey=True)
                
                title = '%s classifier with %s detector on data stream with %i concept drifts' % (clf, det, n_d)
                
                plt.suptitle(title)
                
                for d_id, d in enumerate(deltas):
                    for f_id, f in enumerate(frameworks):
                    
                        ax[f_id, d_id].vlines(
                            results[rep, d_id, f_id, clf_id, n_d_id, det_id, :, 1], 
                            0, 
                            1, 
                            label = 'Label request', 
                            color='tomato', lw=1)
                        ax[f_id, d_id].vlines(
                            results[rep, d_id, f_id, clf_id, n_d_id, det_id, :, 2],
                            0, 
                            1, 
                            label = 'Classifier training', color='cornflowerblue', alpha=1, ls=':', lw=1)

                        ax[f_id, d_id].plot(
                            np.arange(1,500),
                            results[rep, d_id, f_id, clf_id, n_d_id, det_id, :, 0], 
                            label = 'BAC', 
                            color='k', lw=1)
                                
                        ax[f_id, d_id].grid(ls=':')
                        ax[f_id, d_id].set_xticks(
                            get_real_drift(n_chunks, n_d).astype(int))

                        if f_id==0:
                            ax[f_id, d_id].set_title('$\delta = %i$' % (d))
                            ax[-1, d_id].set_xlabel('$chunk$')
                        if d_id==0:
                            ax[f_id, d_id].set_ylabel('$%s$' % f)
                            
                        ax[f_id, d_id].spines['top'].set_visible(False)
                        ax[f_id, d_id].spines['right'].set_visible(False)
                        ax[f_id, d_id].set_xlim(0,n_chunks)


                handles, labels = ax[-1,-1].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
                        
                plt.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.13)
                # ax[-1,-1].legend(frameon=False, ncols=3)
                # plt.tight_layout()
                plt.savefig('foo.png', dpi=500)
                plt.savefig('fig_frameworks/acc_%s_%s_%sd_%i.png' % (clf, det, n_d, rep))
                plt.savefig('fig_frameworks/acc_%s_%s_%sd_%i.eps' % (clf, det, n_d, rep))

                # exit()