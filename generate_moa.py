import numpy as np
from utils import generate_hyperplane
from tqdm import tqdm

hyperplane_cc = {
    'n_features': 12,
    'n_drifting_features': 8,
    'magnitude': .7,
    'noise': 5,
    'sigma': 50,               
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

for n_drifts in n_drifts_par:
    l = np.rint(np.linspace(0, n_drifts, n_samples)).astype(int)
    r = np.arange(hyperplane_cube.shape[0])
    stream = np.sum(hyperplane_cube * (l[None,:] == r[:,None])[:,:,None], axis=0)
    
    print(n_drifts, stream, stream.shape)
    
    np.save('moa_streams/hyperplane_%i_drifts' % n_drifts, stream)
    
