import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

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
m_requests = (np.mean(np.sum((np.isnan(results[...,1])==False), axis=-1), axis=0)/n_chunks)
m_trainings = (np.mean(np.sum((np.isnan(results[...,2])==False), axis=-1), axis=0)/n_chunks)

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
            
tab = tabulate(rows, tablefmt="latex")
print(tab)

f = open("table.txt", "w")
f.write(tab)
f.close()

# ---

tab_data = rows[1:]
env_names = [r[0] for r in tab_data]
tab_data = [r[1:] for r in tab_data]

tab_data = np.array(tab_data)
print(tab_data.shape)

fig, ax = plt.subplots(1,3, figsize=(15,15), sharex=True, sharey=True)

ax[0].imshow(tab_data[:,:6], cmap='coolwarm', aspect='auto')
ax[0].set_title('Balanced Accuracy')

ax[1].imshow(tab_data[:,6:12], 
             cmap='coolwarm', aspect='auto',
             vmin=0, vmax=0.04)
ax[1].set_title('Label request in chunk')

ax[2].imshow(tab_data[:,12:], 
             cmap='coolwarm', aspect='auto',
             vmin=0, vmax=0.04)
ax[2].set_title('Training in chunk')

# continous
for _a in range(12):
    for _b in range(3):
        ax[0].text(2*_b+0.5, _a, "%.3f" % (
            tab_data[_a, 2*_b]
            ) , va='center', ha='center', 
                   c='white' if (tab_data[_a, 2*_b] >0.75) or (tab_data[_a, 2*_b]<0.55) else 'black', 
                   fontsize=11)

# remaining
for _a in range(12,48):
    for _b in range(6):
        ax[0].text(_b, _a, "%.3f" % (
            tab_data[_a, _b]
            ) , va='center', ha='center', 
                   c='white' if (tab_data[_a, _b] >0.75) or (tab_data[_a, _b]<0.55) else 'black', 
                   fontsize=11)
        


# continous
for _a in range(12):
    for _b in range(3):
        aa = 6
        ax[1].text(2*_b+0.5, _a, "%.3f" % (
            tab_data[_a, 2*_b+aa]
            ) , va='center', ha='center', 
                   c='white' if (tab_data[_a, 2*_b+aa] > 0.75) else 'black', 
                   fontsize=11)

# remaining
for _a in range(12,48):
    for _b in range(6):
        aa = 6
        ax[1].text(_b, _a, "%.3f" % (
            tab_data[_a, _b + aa]
            ) , va='center', ha='center', 
                   c='white' if (tab_data[_a, _b+aa] > 0.75) else 'black', 
                   fontsize=11)



# continous
for _a in range(12):
    for _b in range(3):
        aa = 12
        ax[2].text(2*_b+0.5, _a, "%.3f" % (
            tab_data[_a, 2*_b+aa]
            ) , va='center', ha='center', 
                   c='white',
                   fontsize=11)

# remaining
for _a in range(12,48):
    for _b in range(6):
        aa = 12
        ax[2].text(_b, _a, "%.3f" % (
            tab_data[_a, _b + aa]
            ) , va='center', ha='center', 
                   c='black', 
                   fontsize=11)


for aa in ax:
    aa.set_xticks(np.arange(6), [r[4:] for r in rows[0][1:7]], rotation=90)
    aa.set_yticks(np.arange(len(env_names)), env_names)
plt.tight_layout()
plt.savefig('table_vis.png') 
            