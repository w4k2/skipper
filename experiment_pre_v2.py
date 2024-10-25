from strlearn.streams import StreamGenerator
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

np.random.seed(18382)

n_reps = 100
rs = np.random.randint(10,100000,n_reps).astype(int)

n_chunks = 100
chunk_size = 500
acc = np.zeros((n_reps, n_chunks-1))
dist_X0 = np.zeros((n_reps, n_chunks-1))
dist_X1 = np.zeros((n_reps, n_chunks-1))

for n in tqdm(range(n_reps)):
    stream = StreamGenerator(n_chunks=n_chunks,
                         chunk_size=chunk_size,
                         random_state=rs[n],
                         n_drifts=1,
                         concept_sigmoid_spacing=5,
                         incremental=True)

    clf = GaussianNB()
    
    for c in range(n_chunks):
        X, y = stream.get_chunk()
        
        if c==0:
            initial_meanX0 = np.mean(X[y==0], axis=0)
            initial_meanX1 = np.mean(X[y==1], axis=0)
        
        if c>0:
            acc[n, c-1] = accuracy_score(y,clf.predict(X))
            dist_X0[n, c-1] = np.sum(np.abs(np.mean(X[y==0], axis=0) - initial_meanX0))
            dist_X1[n, c-1] = np.sum(np.abs(np.mean(X[y==1], axis=0) - initial_meanX1))

        clf.partial_fit(X, y, np.unique(y))
        
mean_acc = np.mean(acc, axis=0)
std_acc = np.std(acc, axis=0)


mean_dist_X0 = np.mean(dist_X0, axis=0)
std_dist_X0 = np.std(dist_X0, axis=0)

mean_dist_X1 = np.mean(dist_X1, axis=0)
std_dist_X1 = np.std(dist_X1, axis=0)


# PLOT

fig, ax = plt.subplots(1, 1, figsize=(8,4))

ax.fill_between(np.arange(n_chunks-1), 
                mean_acc-std_acc, mean_acc+std_acc, 
                alpha=0.2, color='red',
                lw=0)
ax.plot(mean_acc, color='red')

ax2 = ax.twinx()
# ax2.plot(stream.concept_probabilities[::chunk_size], color='black', ls=':')
ax2.fill_between(np.arange(n_chunks-1), 
                mean_dist_X0-std_dist_X0, mean_dist_X0+std_dist_X0, 
                alpha=0.2, color='blue',
                lw=0)
ax2.plot(mean_dist_X0, color='blue')

ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# ax.spines['right'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.set_yticks([])

ax.grid(ls=':')
ax.set_xlim(0,98)
ax.set_ylim(0.5,1)

ax.set_xlabel('chunk')
ax.set_ylabel('accuracy', color='r')
ax2.set_ylabel('distance', color='b')

handles, labels = plt.gca().get_legend_handles_labels()
line1 = Line2D([0], [0], label='Gaussian Naive Bayes', color='red')
line2 = Line2D([0], [0], label='Distance from initial class centroid', color='blue')
handles.extend([line1, line2])
ax.legend(handles=handles, frameon=False, loc=4)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('preliminary/pre_acc.png')
plt.savefig('preliminary/pre_acc.pdf')
