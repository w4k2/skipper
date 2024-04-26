import numpy as np

class Oracle():
    def __init__(self, n_drifts, n_chunks):
        self.n_drifts = n_drifts
        self.n_chunks = n_chunks

        self.real_drifts = np.linspace(0,self.n_chunks,self.n_drifts+1)[:-1]
        self.real_drifts += ( self.real_drifts[1]/2)
        self.real_drifts = np.ceil(self.real_drifts).astype(int)
        
        self.chunk_count = 1
        self._is_drift = False
        
    def process(self, X=None, y=None, p=None):

        if self.chunk_count in self.real_drifts:
            self._is_drift = True
        else:
            self._is_drift = False
        
        self.chunk_count+=1

        return self

    def empty_process(self):
        # Used in case of Oracle detector to count chunks
        self.chunk_count+=1
        return self
