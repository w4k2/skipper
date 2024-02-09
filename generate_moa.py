import numpy as np
from utils import generate_hyperplane, generate_agrawal
from tqdm import tqdm

hyperplane_cc = {
    'n_features': 12,
    'n_drifting_features': 8,
    'magnitude': .7,
    'noise': 5,
    'sigma': 50,               
}
agrawal_cc = {
    'function': 1
}

n_samples = 100000

n_drifts_par = [5, 10, 15, 30]

print('Generate Hyperplane')
try:
    hyperplane_cube = np.load('moa_streams/raw/hyperplane.npy')
except:
    hyperplane_cube = []

    for random_state in tqdm(range(31)):
        X, y = generate_hyperplane(**hyperplane_cc,
                                   random_state = random_state,
                                   n_samples = n_samples)
        data = np.concatenate((X, y[:, None]), axis=1)
        hyperplane_cube.append(data)
        
    hyperplane_cube = np.array(hyperplane_cube)
    np.save('moa_streams/raw/hyperplane.npy', hyperplane_cube)

print(hyperplane_cube.shape)


print('Generate Agrawal')
try:
    agrawal_cube = np.load('moa_streams/raw/agrawal.npy')
except:
    agrawal_cube = []

    for random_state in tqdm(range(31)):
        X, y = generate_agrawal(**agrawal_cc,
                                   random_state = random_state,
                                   n_samples = n_samples)
        data = np.concatenate((X, y[:, None]), axis=1)
        agrawal_cube.append(data)
        
    agrawal_cube = np.array(agrawal_cube)
    np.save('moa_streams/raw/agrawal.npy', agrawal_cube)

print(agrawal_cube.shape)

print(agrawal_cube)

# Accumulate streams
for n_drifts in n_drifts_par:
    l = np.rint(np.linspace(0, n_drifts, n_samples)).astype(int)
    r = np.arange(hyperplane_cube.shape[0])
    
    # Hyperplane
    stream = np.sum(hyperplane_cube * (l[None,:] == r[:,None])[:,:,None], axis=0)
    print(n_drifts, stream, stream.shape)
    np.save('moa_streams/hyperplane_%i_drifts' % n_drifts, stream)
    
    # Agrawal
    stream = np.sum(agrawal_cube * (l[None,:] == r[:,None])[:,:,None], axis=0)
    print(n_drifts, stream, stream.shape)
    np.save('moa_streams/agrawal_%i_drifts' % n_drifts, stream)
    
