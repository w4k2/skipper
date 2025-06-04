import numpy as np
import matplotlib.pyplot as plt

deltas = [1, 10, 20, 60]

frameworks = ['Continuous Rebuild (CR)',
              'Triggered Rebuild Supervised (TR-S)',
              'Triggered Rebuild Unsupervised (TR-U)',
              'Triggered Rebuild Partially Unsupervised (TR-P)']

results = np.load('results/res_covtype.npy')
n_chunks = results.shape[2]

print(results.shape) # deltas (4), frameworks (4), chunks, metrics        

fig, ax = plt.subplots(4,4, figsize=(10,7), sharex=True, sharey=True)

for f_id, f in enumerate(frameworks):
    fig, ax = plt.subplots(4, 1, figsize=(6,8), sharex=True, sharey=True)
    plt.suptitle('%s \n MLP | Oracle | 5 concept drifts' % f)


    for d_id, d in enumerate(deltas):
    
        x1 = results[d_id, f_id, :, 1]
        x2 = results[d_id, f_id, :, 2]
        ax_twin = ax[d_id].twinx()
        
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

        ax[d_id].plot(
            np.arange(n_chunks),
            results[d_id, f_id, :, 0],
            color='k', lw=1)
                
        ax[d_id].grid(ls=':')
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
    plt.savefig('fig_frameworks/covtype_%i.png' % f_id)
    plt.savefig('fig_frameworks/covtype_%i.eps' % f_id)

    # exit()