from strlearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from SparseTrainDenseTest import SparseTrainDenseTest
from strlearn.streams import StreamGenerator
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from strlearn.ensembles import *

# Config
np.random.seed(1772)

# Constant
n_chunks = 1000
n_informative = 8


# Variables
chunk_sizes = [250,500]
training_intervals = [10, 50, 100]

n_features = [8, 16, 32, 64]
y_noises = [0.0, 0.05]

n_drifts = [5,10,15,30]

random_states = np.random.randint(100,10000, size=10)
n_methods = 9

# Experiment
results = np.full((len(training_intervals), len(random_states), len(chunk_sizes), len(n_features), len(y_noises), len(n_drifts), n_methods, n_chunks-1), np.nan)
pbar = tqdm(total=len(training_intervals)*len(random_states)*len(chunk_sizes)*len(n_features)*len(y_noises)*len(n_drifts))

for _training_int_id, _training_int in enumerate(training_intervals):
    
    print('Training intervals: %i' % _training_int)
    
    for rs_id, rs in enumerate(random_states):
        for _chunk_size_id, _chunk_size in enumerate(chunk_sizes):
                for _n_f_id, _n_f in enumerate(n_features):
                    for _y_noise_id, _y_noise in enumerate(y_noises):
                        for _n_drifts_id, _n_drifts in enumerate(n_drifts):
                            
                            stream = StreamGenerator(
                                n_chunks=n_chunks,
                                chunk_size=_chunk_size,
                                random_state=rs,
                                n_drifts=_n_drifts,
                                n_classes=2,
                                n_features=_n_f,
                                n_informative=n_informative,
                                n_redundant=_n_f-n_informative,
                                y_flip=_y_noise
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
                               MLPClassifier()
                            ]
                            
                            evaluator = SparseTrainDenseTest(n_repeats = _training_int, metrics=balanced_accuracy_score)
                            evaluator.process(stream, methods)
                            
                            results[_training_int_id, rs_id, _chunk_size_id, _n_f_id, _y_noise_id, _n_drifts_id] = evaluator.scores[:,:,0]

                            pbar.update(1)
                            
                            # print(results[_training_int_id, rs_id, _chunk_size_id, _n_f_id, _y_noise_id, _n_drifts_id,0])
                            np.save('res_syn.npy', results)
                            
                            
                        
                        