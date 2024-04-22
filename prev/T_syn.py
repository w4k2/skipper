import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import io

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
            
    rows = []
    rows.append(['Stream', 'SEA', 'AWE', 'AUE', 'WAE', 'DWM', 'KUE', 'ROSE', 'GNB', 'MLP'])
          
    for chunk_size_id, chunk_size in enumerate(chunk_sizes):
        for y_noise_id, y_noise in enumerate(y_noises):
            
            for n_f_id, n_f in enumerate(n_features):
                for n_d_id, n_d in enumerate(n_drifts):
                                        
                    temp_res = res[training_int_id, :, chunk_size_id, n_f_id, y_noise_id, n_d_id]
                    temp_res_stream = np.mean(temp_res, axis=-1)
                    
                    mean_res = np.round(np.mean(temp_res_stream, axis=0),3)
                    std_res = np.round(np.std(temp_res_stream, axis=0),3)                    
                    
                    rows.append([
                        'CS%i_F%i_D%i_N%.2f' % (chunk_size, n_f, n_d, y_noise),
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
    with io.open('tables/syn_tr%i.txt' % training_int, 'w') as file:
        file.write(table)
        
    file.close()
    # print(table)
    # exit()
