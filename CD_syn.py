    
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from cd_plots import compute_CD, graph_ranks


# Constant
n_chunks = 1000

# Variables
chunk_sizes = [250,500]
training_intervals = [10, 50, 100]

n_features = [8, 16, 32, 64]
y_noises = [0.0, 0.05]

n_drifts = [5,10,15,30]

method_names = [ 'SEA', 'AWE', 'AUE', 'WAE', 'DWM', 'KUE', 'ROSE', 'GNB', 'MLP']

res = np.load('res_syn_fix.npy') # training_int , rs_id, chunk_size, n_f, y_noise, n_drifts, methods, chunks

print(res.shape)

for training_int_id, training_int in enumerate(training_intervals):

    fig2, ax2 = plt.subplots(len(chunk_sizes), len(n_drifts), figsize=(15,5), dpi=200)
          
    for chunk_size_id, chunk_size in enumerate(chunk_sizes):            
        for n_d_id, n_d in enumerate(n_drifts):
            
            # Tutaj powinny być wyniki o kształcie: (zbiory, metody)                                    
            temp_res = res[training_int_id, :, chunk_size_id, :, :, n_d_id]
            temp_res_stream = np.mean(temp_res, axis=-1).reshape(-1,9)
            
            print(temp_res_stream.shape) # 80, 9
            
            ranks = []

            for row in temp_res_stream:
                ranks.append(rankdata(row).tolist())
            ranks = np.array(ranks)

            av_ranks = np.mean(ranks, axis=0)
            cd = compute_CD(av_ranks, temp_res_stream.shape[0])

            graph_ranks(av_ranks, method_names, cd=cd, width=6, textspace=1.2)
            plt.savefig("foo.png")
            
            img = plt.imread('foo.png')
            
            ax2[chunk_size_id, n_d_id].imshow(img)
            ax2[chunk_size_id, n_d_id].set_title('CS%i_D%i' % (chunk_size, n_d))
            ax2[chunk_size_id, n_d_id].spines['top'].set_visible(False)
            ax2[chunk_size_id, n_d_id].spines['bottom'].set_visible(False)
            ax2[chunk_size_id, n_d_id].spines['right'].set_visible(False)
            ax2[chunk_size_id, n_d_id].spines['left'].set_visible(False)
            ax2[chunk_size_id, n_d_id].set_xticks([])
            ax2[chunk_size_id, n_d_id].set_yticks([])

    fig2.suptitle('Training every %i chunks' % training_int, fontsize=15)
    fig2.tight_layout()
    fig2.savefig("foo.png")
    fig2.savefig("cd/syn_tr%i.png" % training_int)
 