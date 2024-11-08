import numpy as np

from sklearn import clone

from detectors.Oracle import Oracle
from detectors.ddm import DDM
from detectors.MD3 import MD3
from detectors.OCDD import OneClassDriftDetector

from frameworks.ContinousRebuild import ContinousRebuild
from frameworks.TriggeredRebuildSupervised import TriggeredRebuildSupervised
from frameworks.TriggeredRebuildUnsupervised import TriggeredRebuildUnsupervised
from frameworks.TriggeredRebuildPartiallyUnsupervised import TriggeredRebuildPartiallyUnsupervised
from skmultiflow.trees import HoeffdingTree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from strlearn.streams import StreamGenerator

from tqdm import tqdm


np.random.seed(12213)
r_states = np.random.choice(10000, size=10, replace=False)

class MLPwrap:
    def __init__(self, clf, n_epochs):
        self.clf = clone(clf)
        self.n_epochs = n_epochs
        
    def partial_fit(self, X, y, classes):
        [self.clf.partial_fit(X, y, classes) for i in range(self.n_epochs)]
        return self
    
    def predict(self, X):
        return self.clf.predict(X)
    


deltas = [1, 10, 20, 60]
drifts = [5, 10, 15]
n_epochs_mlp = [1,50,50,50]


fit_partial_mode = True
n_chunks = 500

pbar = tqdm(total=len(r_states)*len(deltas)*3*4*len(drifts))

results = np.full(
    (len(r_states), len(deltas), 4, 3, len(drifts), 2, n_chunks-1, 3),
    np.nan)

    # reps, deltas, frameworks, classifiers, drifts, detectors, chunks, (bac, detections, trainings)
for rs_id, rs in enumerate(r_states):
    for delta_id, d in enumerate(deltas):
        frameworks = [
                ContinousRebuild(partial=fit_partial_mode, delta=d),
                TriggeredRebuildSupervised(partial=fit_partial_mode, delta=d),
                TriggeredRebuildUnsupervised(partial=fit_partial_mode, delta=d),
                TriggeredRebuildPartiallyUnsupervised(partial=fit_partial_mode, delta=d)
        ]
        
        for f_id, f in enumerate(frameworks):
            
            for clf_id in range(3): # classifiers                
                for n_drifts_id, n_drifts in enumerate(drifts):
            
                    #Oracle
                    dets = Oracle(n_drifts=n_drifts, n_chunks=n_chunks)
                    
                    stream = StreamGenerator(
                        n_chunks=n_chunks,
                        chunk_size=250,
                        random_state=rs,
                        n_drifts=n_drifts,
                        n_features=20,
                        n_redundant=0,
                        n_informative=15)
                    
                    classifiers = [
                        MLPwrap(MLPClassifier(random_state=997, hidden_layer_sizes=(10)), n_epochs=n_epochs_mlp[f_id]),
                        HoeffdingTree(),
                        GaussianNB()
                        ]            
            
                    if f_id==0:
                        f.process(stream=stream, clf=classifiers[clf_id])
                    else:
                        f.process(stream=stream, clf=classifiers[clf_id], det=dets)
                        
                    results[rs_id, delta_id, f_id, clf_id, n_drifts_id, 0, :, 0] = f.scores
                    results[rs_id, delta_id, f_id, clf_id, n_drifts_id, 0, :len(f.label_request_chunks), 1] = f.label_request_chunks
                    results[rs_id, delta_id, f_id, clf_id, n_drifts_id, 0, :len(f.training_chunks), 2] = f.training_chunks

                    # Real detector
                    dets = [None, DDM(), OneClassDriftDetector(size = 250, dim = 20, percent = 0.995, nu=0.5), MD3()]
                    
                    stream = StreamGenerator(
                        n_chunks=n_chunks,
                        chunk_size=250,
                        random_state=rs,
                        n_drifts=n_drifts,
                        n_features=20,
                        n_redundant=0,
                        n_informative=15)
                        
                    classifiers = [
                        MLPwrap(MLPClassifier(random_state=997, hidden_layer_sizes=(10)), n_epochs=n_epochs_mlp[f_id]),
                        HoeffdingTree(),
                        GaussianNB()
                        ]            
            
                    if f_id==0:
                        f.process(stream=stream, clf=classifiers[clf_id])
                    else:
                        f.process(stream=stream, clf=classifiers[clf_id], det=dets[f_id])
                        
                    results[rs_id, delta_id, f_id, clf_id, n_drifts_id, 1, :, 0] = f.scores
                    results[rs_id, delta_id, f_id, clf_id, n_drifts_id, 1, :len(f.label_request_chunks), 1] = f.label_request_chunks
                    results[rs_id, delta_id, f_id, clf_id, n_drifts_id, 1, :len(f.training_chunks), 2] = f.training_chunks

                    print(results[rs_id, delta_id, f_id, clf_id, n_drifts_id])
                    pbar.update(1)

        np.save('results/results.npy', results)