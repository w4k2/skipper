import numpy as np
import matplotlib.pyplot as plt
from strlearn.streams import StreamGenerator

# np.random.seed(12213)

stream = StreamGenerator(
    n_chunks=500,
    chunk_size=500,
    n_drifts=2,
    n_features=2,
    n_redundant=0,
    n_informative=2)

n_plots = 5
                        
# chunk_vis = np.linspace(0,500-1,n_plots).astype(int)
chunk_vis = np.arange(0,500,100).astype(int)
chunk_dist = []

for c in range(500):
    X, y = stream.get_chunk()
    if c in chunk_vis:
        chunk_dist.append([X, y])
        
fig, ax = plt.subplots(1, n_plots, figsize=(10,2.4), sharex=True, sharey=True)
    
for cv_id, cv in enumerate(chunk_vis):
    ax[cv_id].set_title('chunk: %i' % cv)
    X, y = chunk_dist[cv_id]
    ax[cv_id].scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', s=3)
    
for aa in ax:
    aa.grid(ls=':')
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    
    
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig_frameworks/stream_vis.png')
plt.savefig('fig_frameworks/stream_vis.eps')
    
    
        