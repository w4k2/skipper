import numpy as np
import matplotlib.pyplot as plt

def get_real_drift(n_ch, n_d):
    real_drifts = np.linspace(0,n_ch,n_d+1)[:-1]
    real_drifts += (real_drifts[1]/2)
    return real_drifts


delta = 100
d = 5
n_chunks = 5000

frameworks = ['Continuous Rebuild (CR)',
              'Triggered Rebuild Supervised (TR-S)',
              'Triggered Rebuild Unsupervised (TR-U)',
              'Triggered Rebuild Partially Unsupervised (TR-P)']

results = np.load('results/results_volume.npy')
# frameworks, chunks, (bac, detections, trainings)

fig, ax = plt.subplots(4, 1, figsize=(6,8), sharex=True, sharey=True)
plt.suptitle('MLP | Oracle | 5 concept drifts')
    
            
for f_id, f in enumerate(frameworks):
    
    ax[f_id].set_title(f)

    x1 = results[f_id, :, 1]
    x2 = results[f_id, :, 2]
    ax_twin = ax[f_id].twinx()
    
    ax_twin.scatter(
        x1, np.full_like(x1, 0),
        color=plt.cm.coolwarm([1.0]), s=50, alpha=1)
    ax_twin.scatter(
        x2, np.full_like(x2, 1),
        color=plt.cm.coolwarm([0.1]), s=50, alpha=1)
    ax_twin.set_ylim(-0.5,2.3)
    
    ax_twin.set_yticks([0,1], ['Label request', 'Training'], rotation=0)
    colors = plt.cm.coolwarm([0.0, 1.0])
    ax_twin.get_yticklabels()[0].set_color(colors[1])
    ax_twin.get_yticklabels()[1].set_color(colors[0])

    ax[f_id].plot(
        np.arange(1,5000),
        results[f_id, :, 0],
        color='k', lw=1)
            
    ax[f_id].grid(ls=':')
    ax[f_id].set_xticks(
        get_real_drift(n_chunks, 5).astype(int))

    ax[f_id].set_ylabel('$\delta = %i$ \n balanced accuracy' % (d))
    
    if f_id==3:
        ax[f_id].set_xlabel('$chunk$')
        
    ax[f_id].spines['top'].set_visible(False)
    ax[f_id].spines['right'].set_visible(False)
    ax[f_id].set_xlim(0,n_chunks)
    
    ax_twin.spines['top'].set_visible(False)
    ax_twin.spines['right'].set_visible(False)
    ax_twin.set_xlim(0,n_chunks)
    
    ax_twin.set_zorder(1)
    ax[f_id].patch.set_visible(False) # hide the 'canvas'
    ax[f_id].set_zorder(2)

plt.tight_layout()
plt.savefig('foo.png', dpi=500)
plt.savefig('fig_frameworks/vis_volume.png', dpi=500)
plt.savefig('fig_frameworks/vis_volume.eps')

# exit()