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

frameworks = ['Continuous Rebuild (CR)',
              'Triggered Rebuild Supervised (TR-S)',
              'Triggered Rebuild Unsupervised (TR-U)',
              'Triggered Rebuild Partially Unsupervised (TR-P)']
clfs = ['MLP', 'HT', 'GNB']
dets = ['Oracle', 'Real']

results = np.load('results/results.npy')
# reps, deltas, frameworks, classifiers, drifts, detectors, chunks, (bac, detections, trainings)

print(results.shape)

selected_results = results[0,:,:,0,0,0] # delatas (4), frameworks(4), chunks, metrics
print(selected_results.shape)
        

for f_id, f in enumerate(frameworks):
    fig, ax = plt.subplots(4, 1, figsize=(7,10), sharex=True, sharey=True)
    plt.suptitle('%s \n MLP | Oracle | 5 concept drifts' % f)


    for d_id, d in enumerate(deltas):
    
        x1 = selected_results[d_id, f_id, :, 1]
        x2 = selected_results[d_id, f_id, :, 2]
        ax_twin = ax[d_id].twinx()
        
        ax_twin.scatter(
            x1, np.full_like(x1, 0),
            color=plt.cm.coolwarm([1.0]), s=50, alpha=1)
        ax_twin.scatter(
            x2, np.full_like(x2, 1),
            color=plt.cm.coolwarm([0.0]), s=50, alpha=1)
        ax_twin.set_ylim(-0.5,2.3)
        
        ax_twin.set_yticks([0,1], ['Label request', 'Training'], rotation=0)
        colors = plt.cm.coolwarm([0.0, 1.0])
        ax_twin.get_yticklabels()[0].set_color(colors[1])
        ax_twin.get_yticklabels()[1].set_color(colors[0])

        ax[d_id].plot(
            np.arange(1,500),
            selected_results[d_id, f_id, :, 0],
            color='k', lw=1)
                
        ax[d_id].grid(ls=':')
        ax[d_id].set_xticks(
            get_real_drift(n_chunks, 5).astype(int))

        ax[d_id].set_ylabel('$\delta = %i$ \n balanced accuracy' % (d))
        
        if d_id==3:
            ax[d_id].set_xlabel('$chunk$')
            
        ax[d_id].spines['top'].set_visible(False)
        ax[d_id].spines['right'].set_visible(False)
        ax[d_id].set_xlim(0,n_chunks)
        
        ax_twin.spines['top'].set_visible(False)
        ax_twin.spines['right'].set_visible(False)
        ax_twin.set_xlim(0,n_chunks)
        
        ax_twin.set_zorder(1)
        ax[d_id].patch.set_visible(False) # hide the 'canvas'
        ax[d_id].set_zorder(2)

    plt.tight_layout()
    plt.savefig('foo.png', dpi=500)
    plt.savefig('fig_frameworks/vis_single_%i.png' % f_id)
    plt.savefig('fig_frameworks/vis_single_%i.eps' % f_id)

    # exit()