from sklearn import clone
from strlearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from SparseTrainDenseTest import SparseTrainDenseTest
from strlearn.streams import SemiSyntheticStreamGenerator
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from strlearn.ensembles import *
import os

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
n_chunks = 1000
n_features = 16
chunk_size = 100

# Variables
training_intervals = [10, 50, 100]
n_drifts = [5,10,15,30]

random_states = np.random.randint(100,10000, size=10)
n_methods = 9

datasets = os.listdir('static')
try:
    datasets.remove('.DS_Store')
except:
    pass

print(datasets)

# Experiment
results = np.full((len(datasets),len(training_intervals), len(random_states), len(n_drifts), n_methods, n_chunks-1), np.nan)
pbar = tqdm(total=len(datasets)*len(training_intervals)*len(random_states)*len(n_drifts))

for dataset_id, dataset_name in enumerate(datasets):
    data = np.loadtxt('static/%s' % dataset_name, delimiter=',')
    X, y = data[:,:-1], data[:, -1]
    
    for _training_int_id, _training_int in enumerate(training_intervals):
        
        print('Dataset: %s | Training intervals: %i' % (dataset_name, _training_int))
        print(X.shape, y.shape)
        
        for rs_id, rs in enumerate(random_states):
            for _n_drifts_id, _n_drifts in enumerate(n_drifts):
                
                stream = SemiSyntheticStreamGenerator(
                    X, y,
                    n_chunks=n_chunks,
                    chunk_size=chunk_size,
                    random_state=rs,
                    n_drifts=_n_drifts,
                    n_features=n_features
                )
                
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
                
                results[dataset_id, _training_int_id, rs_id, _n_drifts_id] = evaluator.scores[:,:,0]

                pbar.update(1)
                np.save('res_semi.npy', results)
                
                
            
            