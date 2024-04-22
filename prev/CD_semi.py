    
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from cd_plots import compute_CD, graph_ranks
import os

# Constant
n_chunks = 1000

# Variables
training_intervals = [10, 50, 100]
n_drifts = [5,10,15,30]

datasets = os.listdir('static')
try:
    datasets.remove('.DS_Store')
except:
    pass

method_names = [ 'SEA', 'AWE', 'AUE', 'WAE', 'DWM', 'KUE', 'ROSE', 'GNB', 'MLP']

res = np.load('res_semi.npy') # datasets, training_int, rs_id, n_drifts, methods, chunks

print(res.shape)

fig2, ax2 = plt.subplots(len(training_intervals), len(n_drifts), figsize=(15,8), dpi=200)

for training_int_id, training_int in enumerate(training_intervals):          
    for n_d_id, n_d in enumerate(n_drifts):
        
        # Tutaj powinny być wyniki o kształcie: (zbiory, metody)                                    
        temp_res = res[:, training_int_id, :, n_d_id]
        temp_res_stream = np.mean(temp_res, axis=-1).reshape(-1,9)
        
        print(temp_res_stream.shape) # 70, 9
        
        ranks = []

        for row in temp_res_stream:
            ranks.append(rankdata(row).tolist())
        ranks = np.array(ranks)

        av_ranks = np.mean(ranks, axis=0)
        cd = compute_CD(av_ranks, temp_res_stream.shape[0])

        graph_ranks(av_ranks, method_names, cd=cd, width=6, textspace=1.2)
        plt.savefig("foo.png")
        
        img = plt.imread('foo.png')
        
        ax2[training_int_id, n_d_id].imshow(img)
        ax2[training_int_id, n_d_id].set_title('T%i_D%i' % (training_int, n_d))
        ax2[training_int_id, n_d_id].spines['top'].set_visible(False)
        ax2[training_int_id, n_d_id].spines['bottom'].set_visible(False)
        ax2[training_int_id, n_d_id].spines['right'].set_visible(False)
        ax2[training_int_id, n_d_id].spines['left'].set_visible(False)
        ax2[training_int_id, n_d_id].set_xticks([])
        ax2[training_int_id, n_d_id].set_yticks([])

fig2.tight_layout()
fig2.savefig("foo.png")
fig2.savefig("cd/semi.png")
