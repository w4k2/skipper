from sklearn import clone
from strlearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from SparseTrainDenseTest import SparseTrainDenseTest
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from strlearn.ensembles import *
import os
from strlearn.streams import ARFFParser, NPYParser

class MLPwrap:
    def __init__(self, clf, n_epochs=25):
        self.clf = clone(clf)
        self.n_epochs = n_epochs
        
    def partial_fit(self, X, y, classes):
        [self.clf.partial_fit(X, y, classes) for i in range(self.n_epochs)]
        return self
    
    def predict(self, X):
        return self.clf.predict(X)
    
    
# Config
np.random.seed(1772)

# Constant
chunk_size = 100

# Variables
training_intervals = [10, 50, 100]
n_methods = 9

streams = os.listdir('moa_streams')
try:
    streams.remove('raw')
    streams.remove('.DS_Store')
except:
    pass

print(streams)

# Experiment
pbar = tqdm(total=len(streams)*len(training_intervals))

n_chunks=1000
results = np.zeros((len(streams), len(training_intervals), n_methods, n_chunks-1))

for str_id, str_name in enumerate(streams):
    for _training_int_id, _training_int in enumerate(training_intervals):

        stream = NPYParser('moa_streams/%s' % str_name, chunk_size=chunk_size, n_chunks=n_chunks)

        methods = [
            SEA(base_estimator=GaussianNB()),
            AWE(base_estimator=GaussianNB()),
            AUE(base_estimator=GaussianNB()),
            WAE(base_estimator=GaussianNB()),
            DWM(base_estimator=GaussianNB()),
            KUE(base_estimator=GaussianNB()),
            ROSE(base_estimator=GaussianNB()),
            GaussianNB(),
            MLPwrap(clf=MLPClassifier(random_state=1223))
        ]

        evaluator = SparseTrainDenseTest(n_repeats = _training_int, metrics=balanced_accuracy_score, verbose=True)
        evaluator.process(stream, methods)
        
        results[str_id, _training_int_id] = evaluator.scores[:,:,0]

        pbar.update(1)
    
        res_str = np.array(results)
        np.save('res_moa.npy', res_str)
    
        
    
    