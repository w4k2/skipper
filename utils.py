
from os import system
from scipy.io.arff import loadarff
import numpy as np
import pandas as pd

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

def generate_agrawal(random_state = 1,          # Seed for random generation of instances.
                     function = 1,              # Classification function used, as defined in the original paper. [1-10]
                     perturb_fraction = .05,    # The amount of peturbation (noise) introduced to numeric values.
                     n_samples = 10000
    ):
    path = ".tmp.arff"
    params = (
        function,
        random_state, 
        perturb_fraction,
        path,
        n_samples
    )
    cmd = "WriteStreamToARFFFile -s (generators.AgrawalGenerator -f %i -i %i -p %f -b) -f %s -m %i" % params

    system(MOA_TEMPLATE % cmd)
    
    rawdata, meta = loadarff(path)

    XX = pd.DataFrame(rawdata)
    Xnum = XX.values[:,[0,1,2,6,7,8]]
    Xcat = XX.apply(lambda x: pd.factorize(x)[0]).values[:,[3,4,5]]
    X = np.concatenate((Xnum, Xcat), axis=1).astype(np.float)
    
    print(X)
    y = (np.array([row.tolist()[-1] for row in rawdata]) == b'groupA').astype(int)

    print(y)
    return X, y


def generate_sea(random_state = 1,          # Seed for random generation of instances.
                     function = 1,              # Classification function used, as defined in the original paper. [1-10]
                     n_instances_concept = 0,
                     noise = 10,
                     n_samples = 10000
    ):
    path = ".tmp.arff"
    params = (
        function,
        random_state, 
        n_instances_concept,
        noise,
        path,
        n_samples
    )
    cmd = "WriteStreamToARFFFile -s (generators.SEAGenerator -f %i -i %i -n %i -p %i -b) -f %s -m %i" % params

    system(MOA_TEMPLATE % cmd)
    
    rawdata, meta = loadarff(path)

    X = np.array([row.tolist()[:-1] for row in rawdata])
    y = (np.array([row.tolist()[-1] for row in rawdata]) == b'groupA').astype(int)

    return X, y


def generate_led(random_state = 1,          # Seed for random generation of instances.
                     noise = 10,
                     n_samples = 10000
    ):
    path = ".tmp.arff"
    params = (
        random_state, 
        noise,
        path,
        n_samples
    )
    cmd = "WriteStreamToARFFFile -s (generators.LEDGenerator -i %i -n %i) -f %s -m %i" % params

    system(MOA_TEMPLATE % cmd)
    
    rawdata, meta = loadarff(path)
    
    X = np.array([row.tolist()[:-1] for row in rawdata]).astype(float)
    y = np.array([row.tolist()[-1] for row in rawdata]).astype(int)

    return X, y
