import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os

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
cols = plt.cm.jet(np.linspace(0,1,len(method_names)))

res = np.load('res_real.npy')
print(res.shape)  # streams, training_int, methods, chunks

fig, ax = plt.subplots(8, 3, figsize=(20,20), sharey=True)
    
for training_int_id, training_int in enumerate(training_intervals):
    for data_id, data_name in enumerate(datasets):
            
        aa = ax[data_id, training_int_id]
                    
        for method_id, method in enumerate(method_names):
            temp = res[data_id, training_int_id, method_id]
            aa.plot(gaussian_filter1d(temp, 3), color=cols[method_id], label=method, linewidth=0.75)
        
        aa.grid(ls=':')
        aa.spines['top'].set_visible(False)
        aa.spines['right'].set_visible(False)
        
        if data_id==len(datasets)-1:
            aa.set_xlabel('treining every %i' % training_int)
        if training_int_id==0:
            aa.set_ylabel('%s' % data_name.split('.')[0])
            

plt.legend(bbox_to_anchor=(-1, -0.32), loc='upper center', ncol=9, frameon=False, fontsize=12)
plt.subplots_adjust(bottom=0.07, top=0.95, right=0.98, left=0.05, hspace=0.2, wspace=0.05)

plt.tight_layout()
plt.savefig('fig/real.png')
plt.savefig('foo.png')
        
exit()
                
                    
                    