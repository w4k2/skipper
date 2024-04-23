import numpy as np
from sklearn.naive_bayes import GaussianNB
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


stream = StreamGenerator(
        n_chunks=500,
        chunk_size=200,
        random_state=233,
        n_drifts=7,
        n_features=20,
        n_redundant=0,
        n_informative=20)


# # Continous rebuild
framework = ContinousRebuild()
framework.process(stream=stream, clf=GaussianNB())


# # Triggered rebuild -- supervised
# framework = TriggeredRebuildSupervised()
# framework.process(stream=stream, clf=GaussianNB(), det=DDM())


# # Triggered rebuild -- unsupervised
# framework = TriggeredRebuildUnsupervised()
# framework.process(stream=stream, clf=GaussianNB(), det=CentroidDistanceDriftDetector())


# # # Triggered rebuild -- unsupervised with label request
# framework = TriggeredRebuildUnsupervisedRequest()
# framework.process(stream=stream, clf=GaussianNB(), det=MD3())


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

ax.set_title(framework)

plt.tight_layout()
plt.savefig('foo.png')
