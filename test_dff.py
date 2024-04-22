import numpy as np
from sklearn.naive_bayes import GaussianNB
from strlearn.streams import StreamGenerator
from detectors.CDDD import CentroidDistanceDriftDetector
from dff import DetectionFeedbackFramework
import matplotlib.pyplot as plt

stream = StreamGenerator(
        n_chunks=500,
        chunk_size=250,
        random_state=233,
        n_drifts=10,
        n_features=10,
        n_redundant=0,
        n_informative=10)

clf = GaussianNB()
det = CentroidDistanceDriftDetector()

framework = DetectionFeedbackFramework(delta=55)

framework.process(stream=stream, det=det, clf=clf)

fig, ax = plt.subplots(1,1,figsize=(12,3))

minm = np.min(framework.scores)
maxm = np.max(framework.scores)

ax.plot(framework.scores, color='b', label='accuracy')
ax.vlines(framework.detections, minm, maxm, color='r', label='detections')
ax.vlines(framework.training_chunks, minm, maxm, color='g', label='training')

ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')
