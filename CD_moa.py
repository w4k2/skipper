    
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from cd_plots import compute_CD, graph_ranks
import os

# Constant
n_chunks = 1000

# Variables
training_intervals = [10, 50, 100]

datasets = os.listdir('moa_streams')
try:
    datasets.remove('raw')
    datasets.remove('.DS_Store')
except:
    pass


datasets = [a.replace('_5_', '_05_') for a in datasets]
order = np.argsort(datasets)

print(order)
datasets = np.array(datasets)[order]
print(datasets)


method_names = [ 'SEA', 'AWE', 'AUE', 'WAE', 'DWM', 'KUE', 'ROSE', 'GNB', 'MLP']

res = np.load('res_moa.npy')
res = res[order]

res = res.reshape(4,4,3,9,999) # generators, drifts, training, methods, chunks
print(res.shape)


fig2, ax2 = plt.subplots(2, 2, figsize=(10,6), dpi=200)
ax2 = ax2.ravel()

for generator_id, generator in enumerate(['Agrawal', 'Hyperplane', 'LED', 'SEA']):
        
    # Tutaj powinny być wyniki o kształcie: (zbiory, metody)                                    
    temp_res = res[generator_id]
    temp_res_stream = np.nanmean(temp_res, axis=-1).reshape(-1,9)

    print(temp_res_stream.shape) # 12, 9

    # exit()
    ranks = []

    for row in temp_res_stream:
        ranks.append(rankdata(row).tolist())
    ranks = np.array(ranks)

    av_ranks = np.mean(ranks, axis=0)
    cd = compute_CD(av_ranks, temp_res_stream.shape[0])

    graph_ranks(av_ranks, method_names, cd=cd, width=6, textspace=1.2)
    plt.savefig("foo.png")

    img = plt.imread('foo.png')

    ax2[generator_id].imshow(img)
    ax2[generator_id].set_title(generator)
    ax2[generator_id].spines['top'].set_visible(False)
    ax2[generator_id].spines['bottom'].set_visible(False)
    ax2[generator_id].spines['right'].set_visible(False)
    ax2[generator_id].spines['left'].set_visible(False)
    ax2[generator_id].set_xticks([])
    ax2[generator_id].set_yticks([])

fig2.tight_layout()
fig2.savefig("foo.png")
fig2.savefig("cd/moa.png")
