import numpy as np
from tabulate import tabulate

deltas = [1, 10, 20, 50]
drifts = [5, 10, 15]
n_epochs_mlp = [1,50,50,50]
n_chunks = 500

frameworks = ['CR',
              'TR-S',
              'TR-U',
              'TR-UR']
clfs = ['MLP', 'HT', 'GNB']
dets = ['Oracle', 'Real']

results = np.load('results.npy')
# reps, deltas, frameworks, classifiers, drifts, detectors, chunks, (bac, detections, trainings)

m_accs = np.nanmean(results[...,0], axis=(-1,0))
m_requests = (np.mean(np.sum((np.isnan(results[...,1])==False), axis=-1), axis=0)/n_chunks)*100
m_trainings = (np.mean(np.sum((np.isnan(results[...,2])==False), axis=-1), axis=0)/n_chunks)*100

print(m_accs.shape) # deltas, frameworks, classifiers, drifts, detectors
print(m_requests.shape)
print(m_requests[0,:,0,0])
print(m_trainings.shape)

rows = []
rows.append(['Environment', 
             'ACC MLP O', 'ACC MLP R', 'ACC HT O', 'ACC HT R', 'ACC GNB O', 'ACC GNB R',
             'REQ MLP O', 'REQ MLP R', 'REQ HT O', 'REQ HT R', 'REQ GNB O', 'REQ GNB R',
             'TRG MLP O', 'TRG MLP R', 'TRG HT O', 'TRG HT R', 'TRG GNB O', 'TRG GNB R',
             ])

for framework_id, framework in enumerate(frameworks):
    for n_drifts_id, n_drifts in enumerate(drifts):
        for delta_id, delta in enumerate(deltas):
            
            temp_acc = np.round(m_accs[delta_id, framework_id, n_drifts_id],3)
            temp_req = np.round(m_requests[delta_id, framework_id, n_drifts_id],3)
            temp_trg = np.round(m_trainings[delta_id, framework_id, n_drifts_id],3)
            
            env_name = '%s | %i DRF | %i Delta' % (framework, n_drifts, delta)
            rows.append([env_name,
                         temp_acc[0,0], temp_acc[0,1], temp_acc[1,0], temp_acc[1,1], temp_acc[2,0], temp_acc[2,1],     
                         temp_req[0,0], temp_req[0,1], temp_req[1,0], temp_req[1,1], temp_req[2,0], temp_req[2,1],     
                         temp_trg[0,0], temp_trg[0,1], temp_trg[1,0], temp_trg[1,1], temp_trg[2,0], temp_trg[2,1],     
                        ])
            
tab = tabulate(rows)
print(tab)

f = open("table.txt", "w")
f.write(tab)
f.close()

            
            