    
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from cd_plots import compute_CD, graph_ranks
import os

# Constant
n_chunks = 1000

# Variables
training_intervals = [10, 50, 100]

datasets = os.listdir('real_streams')
try:
    datasets.remove('.DS_Store')
except:
    pass

# print(datasets)
datasets = ['INSECTS-a', 'INSECTS-g', 'Covtype', 'INSECTS-i-5', 'Poker',
            'INSECTS-i', 'INSECTS-a-5', 'INSECTS-g-5']


method_names = [ 'SEA', 'AWE', 'AUE', 'WAE', 'DWM', 'KUE', 'ROSE', 'GNB', 'MLP']

res = np.load('res_real.npy') # streams, training_int, methods, chunks
print(res.shape)

fig2, ax2 = plt.subplots(1,1, figsize=(5,3), dpi=200)


# Tutaj powinny być wyniki o kształcie: (zbiory, metody)                                    
temp_res = res
temp_res_stream = np.nanmean(temp_res, axis=-1).reshape(-1,9)

print(temp_res_stream.shape) # 24, 9

ranks = []

for row in temp_res_stream:
    ranks.append(rankdata(row).tolist())
ranks = np.array(ranks)

av_ranks = np.mean(ranks, axis=0)
cd = compute_CD(av_ranks, temp_res_stream.shape[0])

graph_ranks(av_ranks, method_names, cd=cd, width=6, textspace=1.2)
plt.savefig("foo.png")

img = plt.imread('foo.png')

ax2.imshow(img)
# ax2.set_title('T%i_D%i' % (training_int, n_d))
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])

fig2.tight_layout()
fig2.savefig("foo.png")
fig2.savefig("cd/real.png")
