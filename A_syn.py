import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.ndimage import gaussian_filter1d

def get_real_drift(n_ch, n_d):
    real_drifts = np.linspace(0,n_ch,n_d+1)[:-1]
    real_drifts += (real_drifts[1]/2)
    return real_drifts

# Constant
n_chunks = 1000

# Variables
chunk_sizes = [250,500]
training_intervals = [10, 50, 100]

n_features = [8, 16, 32, 64]
y_noises = [0.0, 0.05]

n_drifts = [5,10,15,30]

method_names = [ 'SEA', 'AWE', 'AUE', 'WAE', 'DWM', 'KUE', 'ROSE', 'GNB', 'MLP']

colors = [
    'black', 'black',
    'green', 'green',
    'red', 'red', 'red',
    'blue', 'blue'
]
ls = [
    '-', ':',
    '-', ':',
    '-', ':', '-.',
    '-', ':',
]
lw = [
    1,1,
    1,1,
    1,1,1,
    1,1
]

order = [7, 8,      # GNB, MLP 
         0, 4,      # SEA, DWM 
         1, 2, 3,   # AWE, AUE, WAE 
         5, 6]      # KUE, ROSE

cols = plt.cm.jet(np.linspace(0,1,len(method_names)))

res = np.load('res_syn_fix.npy') # training_int , rs_id, chunk_size, n_f, y_noise, n_drifts, methods, chunks

print(res.shape)

mean_res = np.nanmean(res, axis=1) #training_int , chunk_size, n_f, y_noise, n_drifts, methods, chunks                      
print(mean_res.shape)

for chunk_size_id, chunk_size in enumerate(chunk_sizes):
    for y_noise_id, y_noise in enumerate(y_noises):
        for training_int_id, training_int in enumerate(training_intervals):
            
            fig, ax = plt.subplots(4,4, figsize=(20,10), sharey=True, dpi=300)
            plt.suptitle('Chunk size: %i | label noise: %.2f | training every %i chunks' % (chunk_size, y_noise, training_int))
            
            for n_f_id, n_f in enumerate(n_features):
                for n_d_id, n_d in enumerate(n_drifts):
                    
                    aa = ax[n_f_id, n_d_id]
                    
                    # aa.set_title('%i f | %i d' % (n_f, n_d))
                    
                    for o_id, method_id in enumerate(order):
                        method = method_names[method_id]
                        temp = mean_res[training_int_id, chunk_size_id, n_f_id, y_noise_id, n_d_id, method_id]
                        aa.plot(gaussian_filter1d(temp, 3), 
                                color=colors[o_id],# cols[method_id], 
                                label=method,
                                ls=ls[o_id],
                                linewidth=lw[o_id])
                    
                    #for method_id, method in enumerate(method_names):
                    #    temp = mean_res[training_int_id, chunk_size_id, n_f_id, y_noise_id, n_d_id, method_id]
                    #    aa.plot(gaussian_filter1d(temp, 3), color=cols[method_id], label=method, linewidth=0.75)
                    
                    l = get_real_drift(1000, n_d).astype(int)
                    skip = [1, 2, 3, 4][n_d_id]
                    ll = [i if i_id % skip == 0 else '' 
                          for i_id, i in enumerate(l)]
                    
                    aa.set_xticks((l-1), ll)
                    aa.grid(ls=':')
                    aa.spines['top'].set_visible(False)
                    aa.spines['right'].set_visible(False)
                    
                    if n_f_id==len(n_features)-1:
                        aa.set_xlabel('%i drifts' % n_d)
                    if n_d_id==0:
                        aa.set_ylabel('%i features' % n_f)
                        
            
            plt.legend(bbox_to_anchor=(-1, -0.32), loc='upper center', ncol=9, frameon=False, fontsize=12)
            plt.subplots_adjust(bottom=0.15, top=0.95, right=0.98, left=0.05, hspace=0.2, wspace=0.05)

            plt.tight_layout()
            plt.savefig('fig/cs%i_n%i_t%i.png' % (chunk_size, y_noise_id, training_int))
            plt.savefig('fig/cs%i_n%i_t%i.eps' % (chunk_size, y_noise_id, training_int))
            plt.savefig('foo.png')
            
            # time.sleep(1)
            # exit()
                 
                        
                        