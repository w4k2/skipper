import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import io
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

res = np.load('res_semi.npy') # datasets, training_int, rs_id, n_drifts, methods, chunks

# print(res.shape)
# exit()

for training_int_id, training_int in enumerate(training_intervals):
            
    rows = []
    rows.append(['Stream', 'SEA', 'AWE', 'AUE', 'WAE', 'DWM', 'KUE', 'ROSE', 'GNB', 'MLP'])
    
    for data_id, data_name in enumerate(datasets):
        for n_d_id, n_d in enumerate(n_drifts):
                
            temp_res = res[data_id, training_int_id, :, n_d_id]
            temp_res_stream = np.mean(temp_res, axis=-1)

            mean_res = np.round(np.mean(temp_res_stream, axis=0),3)
            std_res = np.round(np.std(temp_res_stream, axis=0),3)
                    
            
            rows.append([
                '%s_D%i' % (data_name.split('.')[0], n_d),
                '%.3f (%.3f)' % (mean_res[0], std_res[0]),
                '%.3f (%.3f)' % (mean_res[1], std_res[1]),
                '%.3f (%.3f)' % (mean_res[2], std_res[2]),
                '%.3f (%.3f)' % (mean_res[3], std_res[3]),
                '%.3f (%.3f)' % (mean_res[4], std_res[4]),
                '%.3f (%.3f)' % (mean_res[5], std_res[5]),
                '%.3f (%.3f)' % (mean_res[6], std_res[6]),
                '%.3f (%.3f)' % (mean_res[7], std_res[7]),
                '%.3f (%.3f)' % (mean_res[8], std_res[8])
            ])            
                    
    table = tabulate(rows, tablefmt='latex') 
    with io.open('tables/semi_tr%i.txt' % training_int, 'w') as file:
        file.write(table)
        
    file.close()
    # print(table)
    # exit()
