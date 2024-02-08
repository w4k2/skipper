import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.ndimage import gaussian_filter1d
import os

def get_real_drift(n_ch, n_d):
    real_drifts = np.linspace(0,n_ch,n_d+1)[:-1]
    real_drifts += (real_drifts[1]/2)
    return real_drifts

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
cols = plt.cm.jet(np.linspace(0,1,len(method_names)))

res = np.load('res_semi.npy') # datasets, training_int , rs_id, n_drifts, methods, chunks

print(res.shape)

mean_res = np.nanmean(res, axis=2) # datasets, training_int , n_drifts, methods, chunks       
print(mean_res.shape)

for training_int_id, training_int in enumerate(training_intervals):
    
    fig, ax = plt.subplots(7,4, figsize=(20,20), sharey=True)
    plt.suptitle('Training every %i chunks' % (training_int))
    
    for data_id, data_name in enumerate(datasets):
        for n_d_id, n_d in enumerate(n_drifts):
            
            aa = ax[data_id, n_d_id]
                        
            for method_id, method in enumerate(method_names):
                temp = mean_res[data_id, training_int_id, n_d_id, method_id]
                aa.plot(gaussian_filter1d(temp, 3), color=cols[method_id], label=method, linewidth=0.75)
            
            l = get_real_drift(1000, n_d).astype(int)
            skip = [1, 2, 3, 4][n_d_id]
            ll = [ i if i_id%skip==0 else '' for i_id, i in enumerate(l)]
            aa.set_xticks((l-1), ll)
            aa.grid(ls=':')
            aa.spines['top'].set_visible(False)
            aa.spines['right'].set_visible(False)
            
            if data_id==len(datasets)-1:
                aa.set_xlabel('%i drifts' % n_d)
            if n_d_id==0:
                aa.set_ylabel('%s' % data_name.split('.')[0])
                
    
    plt.legend(bbox_to_anchor=(-1, -0.32), loc='upper center', ncol=9, frameon=False, fontsize=12)
    plt.subplots_adjust(bottom=0.07, top=0.95, right=0.98, left=0.05, hspace=0.2, wspace=0.05)

    plt.tight_layout()
    plt.savefig('fig/semi_t%i.png' % (training_int))
    plt.savefig('foo.png')
            
    # exit()
                 
                        
                        