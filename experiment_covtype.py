from matplotlib import pyplot as plt
import numpy as np
from sklearn import clone
from sklearn.datasets import fetch_covtype
from sklearn.neural_network import MLPClassifier
from strlearn.streams import NPYParser
from detectors.MD3 import MD3
from detectors.OCDD import OneClassDriftDetector
from detectors.ddm import DDM
from frameworks.ContinousRebuild import ContinousRebuild
from frameworks.TriggeredRebuildPartiallyUnsupervised import TriggeredRebuildPartiallyUnsupervised
from frameworks.TriggeredRebuildSupervised import TriggeredRebuildSupervised
from frameworks.TriggeredRebuildUnsupervised import TriggeredRebuildUnsupervised

class MLPwrap:
    def __init__(self, clf, n_epochs):
        self.clf = clone(clf)
        self.n_epochs = n_epochs
        
    def partial_fit(self, X, y, classes):
        if len(np.unique(y))==1:
            y[:2] = [0,1]
            
        [self.clf.partial_fit(X, y, [0,1]) for i in range(self.n_epochs)]
        return self
    
    def predict(self, X):
        return self.clf.predict(X)
    
    
X, y = fetch_covtype(return_X_y=True)
n_features = X.shape[1]
classes = np.unique(y)

print(X.shape)
print(np.unique(y, return_counts=True))

chunk_size = 500
n_chunks = X.shape[0]//chunk_size

deltas = [1, 10, 20, 60]
n_epochs = [1, 50, 50, 50]
results = np.zeros((len(classes), 4, 4, n_chunks, 3))

# OVA
for class_id, pos_class in enumerate(classes):

    y_new = np.zeros_like(y)
    y_new[y==pos_class] = 1
    
    print(np.unique(y_new, return_counts=True))
    data = np.column_stack([X,y_new])
    np.save('covtype.npy', data)



    # experiment
    for d_id, d in enumerate(deltas):
        frameworks = [
            ContinousRebuild(partial=True, delta=d),
            TriggeredRebuildSupervised(partial=True, delta=d),
            TriggeredRebuildUnsupervised(partial=True, delta=d),
            TriggeredRebuildPartiallyUnsupervised(partial=True, delta=d)
        ]

        dets = [None, 
                DDM(drift_lvl=2.0),
                OneClassDriftDetector(size = chunk_size,
                                    dim = n_features,
                                    percent = 0.8, 
                                    nu=0.5), 
                MD3(sigma=0.1)]

        for f_id in range(4):
            
            clf = MLPwrap(MLPClassifier(random_state=997, 
                                        hidden_layer_sizes=(100)), 
                                        n_epochs=n_epochs[f_id])
            f = frameworks[f_id]
            det = dets[f_id]
            
            stream = NPYParser('covtype.npy',
                        chunk_size=chunk_size, 
                        n_chunks=n_chunks)
            if f_id==0:
                f.process(stream=stream, clf=clf)
            else:
                f.process(stream=stream, clf=clf, det=det)
                
            results[class_id, d_id, f_id, 1:, 0] = f.scores
            results[class_id, d_id, f_id, :len(f.label_request_chunks), 1] = f.label_request_chunks
            results[class_id, d_id, f_id, :len(f.training_chunks), 2] = f.training_chunks

            print(f.label_request_chunks)
            print(f.training_chunks)
            np.save('results/res_covtype_ova.npy', results)