import numpy as np
import os
from tabulate import tabulate
import io

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

res = np.load('res_moa.npy')
res = res[order]


print(res.shape)  # streams, training_int, methods, chunks
    
for training_int_id, training_int in enumerate(training_intervals):
    
    rows = []
    rows.append(['Stream', 'SEA', 'AWE', 'AUE', 'WAE', 'DWM', 'KUE', 'ROSE', 'GNB', 'MLP'])
    
    for data_id, data_name in enumerate(datasets):
            
        temp_res = res[data_id, training_int_id]        
        mean_res = np.nanmean(temp_res, axis=-1)
                           
        rows.append([
            '%s' % (data_name.split('.')[0]),
            '%.3f' % (mean_res[0]),
            '%.3f' % (mean_res[1]),
            '%.3f' % (mean_res[2]),
            '%.3f' % (mean_res[3]),
            '%.3f' % (mean_res[4]),
            '%.3f' % (mean_res[5]),
            '%.3f' % (mean_res[6]),
            '%.3f' % (mean_res[7]),
            '%.3f' % (mean_res[8])
        ])            
                
    table = tabulate(rows, tablefmt='latex') 
    with io.open('tables/moa_tr%i.txt' % training_int, 'w') as file:
        file.write(table)
            