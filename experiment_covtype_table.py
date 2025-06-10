       
from scipy.stats import rankdata, ranksums, wilcoxon
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
    

deltas = [1, 10, 20, 60]
frameworks = ['CR', 'TR-S', 'TR-U', 'TR-P']
labels = []
for d in deltas:
    for f in frameworks:
        labels.append('%s-%02d' % (f, d))
print(labels)

results = np.load('results/res_covtype_ova.npy')
print(results.shape) # classes, deltas (4), frameworks (4), chunks, metrics        

# ACC
res_acc = np.mean(results[:,:,:,:,0], axis=3)
print(res_acc.shape) # 7, 4, 4
res_acc = res_acc.reshape(7, -1) # 7, (deltas x frameworks)


stat = np.zeros((len(labels), len(labels)))
p_val = np.zeros((len(labels), len(labels)))
better = np.zeros((len(labels), len(labels))).astype(bool)

for i in range(len(labels)):
    for j in range(len(labels)):
        if i!=j:
            stat[i,j], p_val[i,j] = wilcoxon(res_acc[:,i], res_acc[:,j])
            better[i,j] = np.mean(res_acc[:,i]) > np.mean(res_acc[:,j])

significant = p_val<0.05
significantly_better = significant*better

# print(significantly_better)
# for ds_id in range(7):
    
#     c = ['C%i' % ds_id]
#     c.extend(np.round(res_acc[ds_id], 3))
#     c2 = ['C%i' % ds_id]
#     for i in range(7):
#         better = np.argwhere(significantly_better[i])
#         print(better)
#     print(c)
#     exit()

# plt.imshow(res_acc, cmap='coolwarm')
# plt.savefig('foo.png')

datasets = ['C%i' %i for i in range(7)]
data1 = np.round(res_acc,3).astype(object)[:,:8]
data2 = np.round(res_acc,3).astype(object)[:,8:]

data1= np.column_stack((datasets, data1))
data2= np.column_stack((datasets, data2))

print(tabulate(data1, headers=labels[:8], tablefmt='latex'))
print(tabulate(data2, headers=labels[8:], tablefmt='latex'))