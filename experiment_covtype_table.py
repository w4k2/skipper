       
from scipy.stats import rankdata, ranksums, wilcoxon
import numpy as np
import matplotlib.pyplot as plt
    

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

# print(p_val)
# exit()
significant = p_val<0.05
significantly_better = significant*better

print(significantly_better)

exit()

# CLF trainign
res_trn = results[...,2]==0
n_chunks = res_trn.shape[-1]

res_trn = np.sum(res_trn, axis=-1)/n_chunks
print(res_trn.shape) # 7, 4, 4
res_trn = res_trn.reshape(7, -1) # 7, (deltas x frameworks)

ranks = []
for row in res_trn:
    ranks.append(rankdata(row).tolist())
ranks = np.array(ranks)

av_ranks = np.mean(ranks, axis=0)
cd = compute_CD(av_ranks, res_acc.shape[0])

fig = graph_ranks(av_ranks, labels, cd=cd, width=6, textspace=1.1, reverse=True, title='Classifier Training chunks', color=plt.cm.coolwarm(.9))
# plt.subplots_adjust(top=0.8)
plt.tight_layout()

plt.savefig("foo.png", dpi=300)
plt.savefig("fig_frameworks/CD_trn.png")
plt.savefig("fig_frameworks/CD_trn.pdf")
