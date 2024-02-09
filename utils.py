
from os import system
from scipy.io.arff import loadarff
import numpy as np

MOA_TEMPLATE = 'java -cp lib/moa.jar -javaagent:lib/sizeofag-1.0.4.jar moa.DoTask "%s"'

def generate_hyperplane(random_state = 1,               # Seed for random generation of instances.
                        n_features = 10,                # The number of attributes to generate.
                        n_classes = 2,                  # The number of classes to generate.
                        n_drifting_features = 2,        # The number of attributes with drift.
                        magnitude = .0,                 # Magnitude of the change for every example.
                        noise = 5,                      # Noise percentage (int).
                        sigma = 10,                     # Percentage of probability that the direction of change is reversed (int).
                        n_samples = 10000
                        ):
    path = ".tmp.arff"
    params = (
        n_classes,
        random_state, n_features, n_drifting_features, 
        magnitude, 
        noise, sigma, 
        path, 
        n_samples
    )
    cmd = "WriteStreamToARFFFile -s (generators.HyperplaneGenerator -c %i -i %i -a %i -k %i -t %f -n %i -s %i) -f %s -m %i" % params

    system(MOA_TEMPLATE % cmd)
    
    rawdata, meta = loadarff(path)

    X = np.array([row.tolist()[:-1] for row in rawdata])
    y = (np.array([row.tolist()[-1] for row in rawdata]) == b'class1').astype(int)

    return X, y
