import numpy as np
from sklearn import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from strlearn.streams import StreamGenerator
from detectors.CDDD import CentroidDistanceDriftDetector
from detectors.adwin import ADWIN
from detectors.ddm import DDM
from detectors.MD3 import MD3

from ContinousRebuild import ContinousRebuild
from TriggeredRebuildSupervised import TriggeredRebuildSupervised
from TriggeredRebuildUnsupervised import TriggeredRebuildUnsupervised
from TriggeredRebuildUnsupervisedRequest import TriggeredRebuildUnsupervisedRequest

import matplotlib.pyplot as plt

def get_real_drift(n_ch, n_d):
    real_drifts = np.linspace(0,n_ch,n_d+1)[:-1]
    real_drifts += (real_drifts[1]/2)
    return real_drifts

class MLPwrap:
    def __init__(self, clf, n_epochs=50):
        self.clf = clone(clf)
        self.n_epochs = n_epochs
        
    def partial_fit(self, X, y, classes):
        [self.clf.partial_fit(X, y, classes) for i in range(self.n_epochs)]
        return self
    
    def predict(self, X):
        return self.clf.predict(X)
    

dets = [None, DDM(), CentroidDistanceDriftDetector(), MD3()]
# clf = MLPWrap(MLPClassifier(random_state=997, hidden_layer_sizes=(10)))
clf = GaussianNB()

frameworks = [
        ContinousRebuild(partial=False),
        TriggeredRebuildSupervised(partial=False),
        TriggeredRebuildUnsupervised(partial=False),
        TriggeredRebuildUnsupervisedRequest(partial=False)
]

for f_id, f_name in enumerate(['CR', 'TS', 'TU', 'TUR']):
        
        stream = StreamGenerator(
                n_chunks=500,
                chunk_size=200,
                random_state=233,
                n_drifts=7,
                n_features=20,
                n_redundant=0,
                n_informative=20)
        
        framework = frameworks[f_id]
        if f_id==0:
                framework.process(stream=stream, clf=clone(clf))
        else:
                framework.process(stream=stream, clf=clone(clf), det=dets[f_id])



        fig, ax = plt.subplots(1,1,figsize=(12,3))

        minm = np.min(framework.scores)
        maxm = np.max(framework.scores)

        ax.plot(framework.scores, color='b', label='accuracy')
        try:
                ax.vlines(framework.detections, minm, maxm, color='r', label='detections', alpha=0.2)
                ax.vlines(framework.training_chunks, minm, maxm, color='g', label='training', alpha=0.2)
                ax.vlines(framework.past_training_chunks, minm, maxm, color='g', label='training from', ls=':', alpha=0.2)
        except:
                pass

        ax.set_xticks(get_real_drift(500, 7).astype(int))
        ax.set_ylim(minm,1)
        ax.legend(frameon=False, ncols=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(ls=':')
        ax.set_ylim(0.5, 1)

        ax.set_title(framework)

        plt.tight_layout()
        plt.savefig('foo.png')
        plt.savefig('%s.png' % f_name)