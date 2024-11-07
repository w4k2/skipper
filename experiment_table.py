import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

deltas = [1, 10, 20, 60]
drifts = [5, 10, 15]
n_epochs_mlp = [1,50,50,50]
n_chunks = 500

frameworks = ['CR',
              'TR-S',
              'TR-U',
              'TR-P']
clfs = ['MLP', 'HT', 'GNB']
dets = ['Oracle', 'Real']

results = np.load('results/results.npy')
# reps, deltas, frameworks, classifiers, drifts, detectors, chunks, (bac, detections, trainings)

m_accs = np.nanmean(results[...,0], axis=(-1,0))
m_requests = (np.mean(np.sum((np.isnan(results[...,1])==False), axis=-1), axis=0)/n_chunks)
m_trainings = (np.mean(np.sum((np.isnan(results[...,2])==False), axis=-1), axis=0)/n_chunks)

# print(m_accs.shape) # deltas, frameworks, classifiers, drifts, detectors
# print(m_requests.shape)
# print(m_requests[0,:,0,0])
# print(m_trainings.shape)


rows = []
rows.append(['Environment', 
             'ACC MLP O', 'ACC MLP R', 'ACC HT O', 'ACC HT R', 'ACC GNB O', 'ACC GNB R',
             'REQ MLP O', 'REQ MLP R', 'REQ HT O', 'REQ HT R', 'REQ GNB O', 'REQ GNB R',
             'TRG MLP O', 'TRG MLP R', 'TRG HT O', 'TRG HT R', 'TRG GNB O', 'TRG GNB R',
             ])

for framework_id, framework in enumerate(frameworks):
    for n_drifts_id, n_drifts in enumerate(drifts):
        for delta_id, delta in enumerate(deltas):
            
            temp_acc = np.round(m_accs[delta_id, framework_id, :, n_drifts_id],3)
            temp_req = np.round(m_requests[delta_id, framework_id, :, n_drifts_id],3)
            temp_trg = np.round(m_trainings[delta_id, framework_id, :, n_drifts_id],3)
            
            env_name = '%s (%02d) | %02d Drf' % (framework, delta, n_drifts)
            rows.append([env_name,
                         temp_acc[0,0], temp_acc[0,1], temp_acc[1,0], temp_acc[1,1], temp_acc[2,0], temp_acc[2,1],     
                         temp_req[0,0], temp_req[0,1], temp_req[1,0], temp_req[1,1], temp_req[2,0], temp_req[2,1],     
                         temp_trg[0,0], temp_trg[0,1], temp_trg[1,0], temp_trg[1,1], temp_trg[2,0], temp_trg[2,1],     
                        ])
            
tab = tabulate(rows, tablefmt="latex")
print(tab)

f = open("tables/table.txt", "w")
f.write(tab)
f.close()

# --- ---- ----- ----- ----- ----- ----- 

tab_data = rows[1:]
env_names = [r[0] for r in tab_data]
tab_data = [r[1:] for r in tab_data]

tab_data = np.array(tab_data)
print(tab_data.shape)

"""
HERE COMES THE SUN
"""
fig, ax = plt.subplots(4,4, figsize=(11,11), sharex=False, sharey=False,
                       width_ratios=[1,1,1,.4])


# Right legend
for z in range(4):
    ax[z,-1].set_ylim(-.5,11.5)

    ax[z, -1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax[z, -1].set_xlim(-.5, 1.5)
    # ax[z, -1].spines['right'].set_visible(False)
    ax[z, -1].set_xticks([0,1], ['$\delta$', 'Drifts'])
    ax[z, -1].set_yticks([3.5,7.5],['',''])
    ax[z, -1].grid(ls=":", axis='y')


    for i, v_drifts in enumerate([5,10,15]):
        ax[z, -1].text(1,9.5-i*4,v_drifts, ha='center', va='center')
        for j, v_delta in enumerate([1,10,20,60]):        
            ax[z, -1].text(0,11-(j+i*4),v_delta, ha='center', va='center')

rows_per_framework = 12
for framework_id in range(4):
    start = framework_id*rows_per_framework
    stop = (framework_id+1)*rows_per_framework
    
    ax[framework_id, 0].imshow(tab_data[start:stop,:6], cmap='coolwarm', aspect='auto',
                vmin=0.5, vmax=0.8)

    ax[framework_id, 1].imshow(tab_data[start:stop,6:12], 
                cmap='coolwarm', aspect='auto',
                vmin=0, vmax=0.04)

    ax[framework_id, 2].imshow(tab_data[start:stop,12:], 
                cmap='coolwarm', aspect='auto',
                vmin=0, vmax=0.04)
    
    if framework_id == 0:
        ax[framework_id, 0].set_title('Balanced Accuracy')
        ax[framework_id, 1].set_title('Label request in chunk')
        ax[framework_id, 2].set_title('Training in chunk')

# continous
for _a in range(12):
    for _b in range(3):
        ax[0, 0].text(2*_b+0.5, _a, ("%.3f" % (
            tab_data[_a, 2*_b]
            ))[1:] , va='center', ha='center', 
                c='white' if (tab_data[_a, 2*_b] >0.75) or (tab_data[_a, 2*_b]<0.55) else 'black')

# remaining
for _a in range(12,48):
    for _b in range(6):
        framework_id = _a // rows_per_framework
        # print(framework_id, _a, _b)
        
        ax[framework_id, 0].text(_b, _a%rows_per_framework, ('%.3f' %tab_data[_a, _b])[1:],
                                 ha='center', va='center',
                                 c='white' if (tab_data[_a, _b] >0.75) or (tab_data[_a, _b]<0.55) else 'black')

# continous
for _a in range(12):
    for _b in range(3):
        aa = 6
        ax[0, 1].text(2*_b+0.5, _a, ("%.3f" % (
            tab_data[_a, 2*_b+aa]
            ))[1:] , va='center', ha='center', 
                   c='white' if (tab_data[_a, 2*_b+aa] > 0.75) else 'black')

# remaining
for _a in range(12,48):
    for _b in range(6):
        framework_id = _a // rows_per_framework
        
        aa = 6
        ax[framework_id, 1].text(_b, _a%rows_per_framework, ("%.3f" % (
            tab_data[_a, _b + aa]
            ))[1:] , va='center', ha='center', 
                   c='white' if (tab_data[_a, _b+aa] > 0.75) else 'black')



# continous
for _a in range(12):
    for _b in range(3):
        aa = 12
        ax[0, 2].text(2*_b+0.5, _a, ("%.3f" % (
            tab_data[_a, 2*_b+aa]
            ))[1:] , va='center', ha='center', 
                   c='white')

# remaining
for _a in range(12,48):
    for _b in range(6):
        framework_id = _a // rows_per_framework
        
        aa = 12
        
        v = "%.3f" % (
            tab_data[_a, _b + aa]
        )
        v = v[1:]
        
        ax[framework_id, 2].text(_b, _a%rows_per_framework, v , va='center', ha='center', 
                   c='black')

for i, raa in enumerate(ax):
    for j, aa in enumerate(raa[:-1]):
        aa.set_xticks(np.arange(6), [r[4:] for r in rows[0][1:7]], rotation=90)
        # aa.set_yticks(np.arange(len(env_names)), env_names, fontsize=15)
        
        start = i*rows_per_framework
        stop = (i+1)*rows_per_framework
        
        _env_names = env_names[start:stop]
        
        # print(_env_names)
        
        # print('A', i, j, len(env_names), start, stop, len(_env_names))
        
        # aa.set_yticks(np.arange(len(_env_names)), _env_names, fontsize=15)
        
for i in range(4):        
    start = i*rows_per_framework
    stop = (i+1)*rows_per_framework
    
    _env_names = env_names[start:stop]
    
    # print(i, _env_names)
    
    # ax[i,0].set_yticks(np.arange(len(_env_names)), _env_names, fontsize=15)

# ax[0,0].set_ylabel('\nContinous Rebuild')
# ax[1,0].set_ylabel('Triggered Rebuild\nSupervised drift detection')
# ax[2,0].set_ylabel('Triggered Rebuild\nUnsupervised drift detection')
# ax[3,0].set_ylabel('Triggered Rebuild\nSemi-supervised drift detection')
ax[0,0].set_ylabel('$CR$')
ax[1,0].set_ylabel('$TR-S$')
ax[2,0].set_ylabel('$TR-U$')
ax[3,0].set_ylabel('$TR-P$')


for i in range(4):
    for j in range(3):
        # ax[i,j].spines['top'].set_visible(False)
        # ax[i,j].spines['right'].set_visible(False)
        # ax[i,j].spines['left'].set_visible(False)
        # ax[i,j].spines['bottom'].set_visible(False)
        
        ax[i,j].set_yticks([])
        
        ax[i,j].hlines([3.5,7.5], -.5, 5.5, lw=1, color='white', alpha=.5)
        
        ax[i,j].vlines([1.5, 3.5], -.5, 11.5, lw=1, color='white', alpha=.5)
        
        if i < 3:
            ax[i,j].set_xticks([])

#plt.tight_layout()

plt.subplots_adjust(left=0.05, right=.95, top=0.95, bottom=0.07)

plt.savefig('tables/table_vis.png') 
plt.savefig('tables/table_vis.eps') 
plt.savefig('foo.png', dpi = 500)
            