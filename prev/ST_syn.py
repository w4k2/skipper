
# Shaker Tables

from shaker import DriftEvaluator
import numpy as np
from tabulate import tabulate
import io

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

res = np.load('res/res_syn_fix.npy') # training_int , rs_id, chunk_size, n_f, y_noise, n_drifts, methods, chunks

print(res.shape)

for training_int_id, training_int in enumerate(training_intervals):
            
    rows_rec_len = []
    rows_perf_loss = []
    rows_rec_len.append(['Stream', 'SEA', 'AWE', 'AUE', 'WAE', 'DWM', 'KUE', 'ROSE', 'GNB', 'MLP'])
    rows_perf_loss.append(['Stream', 'SEA', 'AWE', 'AUE', 'WAE', 'DWM', 'KUE', 'ROSE', 'GNB', 'MLP'])
          
    for chunk_size_id, chunk_size in enumerate(chunk_sizes):
        for y_noise_id, y_noise in enumerate(y_noises):
            
            for n_f_id, n_f in enumerate(n_features):
                for n_d_id, n_d in enumerate(n_drifts):
                                        
                    temp_res = res[training_int_id, :, chunk_size_id, n_f_id, y_noise_id, n_d_id]
                    
                    print(temp_res.shape)
                    
                    rec_len = np.zeros((10, 9, n_d))
                    perf_loss = np.zeros((10,9, n_d))
                    for r in range(10):
                        for m in range(9):
                            de = DriftEvaluator(temp_res[r, m], (get_real_drift(1000, n_d).astype(int))-1, 1000)
                            rec_len[r, m] = de.calculate_recovery_lengths()
                            perf_loss[r, m] = de.calculate_performance_loss()
                    
                    rec_len_mean = np.mean(rec_len, axis=(0,-1))
                    rec_len_std = np.std(rec_len, axis=(0,-1))
                    
                    perf_loss_mean = np.mean(perf_loss, axis=(0,-1))
                    perf_loss_std = np.std(perf_loss, axis=(0,-1))                    
                    
                    rows_rec_len.append([
                        'CS%i_F%i_D%i_N%.2f' % (chunk_size, n_f, n_d, y_noise),
                        '%.3f (%.3f)' % (rec_len_mean[0], rec_len_std[0]),
                        '%.3f (%.3f)' % (rec_len_mean[1], rec_len_std[1]),
                        '%.3f (%.3f)' % (rec_len_mean[2], rec_len_std[2]),
                        '%.3f (%.3f)' % (rec_len_mean[3], rec_len_std[3]),
                        '%.3f (%.3f)' % (rec_len_mean[4], rec_len_std[4]),
                        '%.3f (%.3f)' % (rec_len_mean[5], rec_len_std[5]),
                        '%.3f (%.3f)' % (rec_len_mean[6], rec_len_std[6]),
                        '%.3f (%.3f)' % (rec_len_mean[7], rec_len_std[7]),
                        '%.3f (%.3f)' % (rec_len_mean[8], rec_len_std[8])
                    ])   
                                        
                    rows_perf_loss.append([
                        'CS%i_F%i_D%i_N%.2f' % (chunk_size, n_f, n_d, y_noise),
                        '%.3f (%.3f)' % (perf_loss_mean[0], perf_loss_std[0]),
                        '%.3f (%.3f)' % (perf_loss_mean[1], perf_loss_std[1]),
                        '%.3f (%.3f)' % (perf_loss_mean[2], perf_loss_std[2]),
                        '%.3f (%.3f)' % (perf_loss_mean[3], perf_loss_std[3]),
                        '%.3f (%.3f)' % (perf_loss_mean[4], perf_loss_std[4]),
                        '%.3f (%.3f)' % (perf_loss_mean[5], perf_loss_std[5]),
                        '%.3f (%.3f)' % (perf_loss_mean[6], perf_loss_std[6]),
                        '%.3f (%.3f)' % (perf_loss_mean[7], perf_loss_std[7]),
                        '%.3f (%.3f)' % (perf_loss_mean[8], perf_loss_std[8])
                    ])            
                    
    table_rec_len = tabulate(rows_rec_len, tablefmt='latex') 
    with io.open('tables/syn_tr%i_reclen.txt' % training_int, 'w') as file:
        file.write(table_rec_len)
    
    file.close()

    table_perf_loss = tabulate(rows_perf_loss, tablefmt='latex') 
    with io.open('tables/syn_tr%i_perfloss.txt' % training_int, 'w') as file:
        file.write(table_perf_loss)
    
    file.close()

    # print(table)
    # exit()
