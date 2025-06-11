import numpy as np
from sklearn import clone
from detectors.Oracle import Oracle
from frameworks.ContinousRebuild import ContinousRebuild
from frameworks.TriggeredRebuildSupervised import TriggeredRebuildSupervised
from frameworks.TriggeredRebuildUnsupervised import TriggeredRebuildUnsupervised
from frameworks.TriggeredRebuildPartiallyUnsupervised import TriggeredRebuildPartiallyUnsupervised
from sklearn.neural_network import MLPClassifier
from strlearn.streams import StreamGenerator
np.random.seed(12213)

class MLPwrap:
    def __init__(self, clf, n_epochs):
        self.clf = clone(clf)
        self.n_epochs = n_epochs
        
    def partial_fit(self, X, y, classes):
        [self.clf.partial_fit(X, y, classes) for i in range(self.n_epochs)]
        return self
    
    def predict(self, X):
        return self.clf.predict(X)
    

d = 300
drifts = 5
n_epochs_mlp = [1,50,50,50]

fit_partial_mode = True
n_chunks = 5000
chunk_size = 500
n_features = 50

results = np.full((4, n_chunks-1, 3),np.nan) # frameworks, chunks, metrics

frameworks = [
        ContinousRebuild(partial=fit_partial_mode, delta=d),
        TriggeredRebuildSupervised(partial=fit_partial_mode, delta=d),
        TriggeredRebuildUnsupervised(partial=fit_partial_mode, delta=d),
        TriggeredRebuildPartiallyUnsupervised(partial=fit_partial_mode, delta=d)
]


for f_id, f in enumerate(frameworks):
    print(f_id)
    clf = MLPwrap(MLPClassifier(random_state=997, hidden_layer_sizes=(10)), n_epochs=n_epochs_mlp[f_id])
    
    stream = StreamGenerator(
        n_chunks=n_chunks,
        chunk_size=chunk_size,
        n_drifts=drifts,
        n_features=n_features,
        n_redundant=0,
        random_state=3456,
        n_informative=int(0.3*n_features))
                  

    if f_id==0:
        f.process(stream=stream, clf=clf)
    else:
        f.process(stream=stream, clf=clf, det=Oracle(n_drifts=drifts, n_chunks=n_chunks))
        
    results[f_id,  :, 0] = f.scores
    results[f_id, :len(f.label_request_chunks), 1] = f.label_request_chunks
    results[f_id, :len(f.training_chunks), 2] = f.training_chunks

    print(results[f_id,:,:])

np.save('results/results_volume.npy', results)