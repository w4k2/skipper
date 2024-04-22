import numpy as np
import os
from tabulate import tabulate
import io

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

res = np.load('res_real.npy')
print(res.shape)  # streams, training_int, methods, chunks
    
for training_int_id, training_int in enumerate(training_intervals):
    
    rows = []
    rows.append(['Stream', 'SEA', 'AWE', 'AUE', 'WAE', 'DWM', 'KUE', 'ROSE', 'GNB', 'MLP'])
    
    for data_id, data_name in enumerate(datasets):
            
        temp_res = res[data_id, training_int_id]
        print(data_name, np.sum(np.isnan(temp_res)))
        
        mean_res = np.nanmean(temp_res, axis=-1)
                           
        rows.append([
            '%s' % (data_name),
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
    with io.open('tables/real_tr%i.txt' % training_int, 'w') as file:
        file.write(table)
            