import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os

def get_real_drift(n_ch, n_d):
    real_drifts = np.linspace(0,n_ch,n_d+1)[:-1]
    real_drifts += (real_drifts[1]/2)
    return real_drifts

# W Kolumnach dryfy, w wierszach generatory

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
print(datasets)
datasets = np.array(datasets)[order]


method_names = [ 'SEA', 'AWE', 'AUE', 'WAE', 'DWM', 'KUE', 'ROSE', 'GNB', 'MLP']
cols = plt.cm.jet(np.linspace(0,1,len(method_names)))

res = np.load('res_moa.npy')
print(res.shape)  # streams, training_int, methods, chunks


res = res[order]
    
for training_int_id, training_int in enumerate(training_intervals):
    
    fig, ax = plt.subplots(4, 4, figsize=(20,11), sharey=True)
    ax = ax.ravel()
    plt.suptitle('Training every %i chunks' % (training_int))

    for data_id, data_name in enumerate(datasets):
            
        aa = ax[data_id]
                    
        for method_id, method in enumerate(method_names):
            temp = res[data_id, training_int_id, method_id]
            aa.plot(gaussian_filter1d(temp, 3), color=cols[method_id], label=method, linewidth=0.75)
        
        aa.grid(ls=':')
        aa.spines['top'].set_visible(False)
        aa.spines['right'].set_visible(False)
        
        n_d = [5,10,15,30][data_id%4]
        l = get_real_drift(1000, n_d).astype(int)
        skip = [1, 2, 3, 4][data_id%4]
        ll = [ i if i_id%skip==0 else '' for i_id, i in enumerate(l)]
        aa.set_xticks((l-1), ll)
        aa.set_ylabel('%s' % data_name.split('.')[0])
            

    plt.legend(bbox_to_anchor=(-1, -0.15), loc='upper center', ncol=9, frameon=False, fontsize=12)
    plt.subplots_adjust(bottom=0.07, top=0.95, right=0.98, left=0.05, hspace=0.2, wspace=0.05)

    # plt.tight_layout()
    plt.savefig('fig/moa_t%i.png' % training_int)
    plt.savefig('foo.png')
            
    # exit()
                    
                    
                    