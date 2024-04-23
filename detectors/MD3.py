import numpy as np
from sklearn.svm import LinearSVC

class MD3:
    def __init__(self, sigma = 0.15):
        self.sigma = sigma
        self.first = True
        
    def process(self, X, y):
        if self.first:
            self.model = LinearSVC().fit(X, y)
            self.first = False
            self._is_drift = False
                    
            self.rho_min = np.inf
            self.rho_max = -np.inf
        
        else:
            # Merginal density calculated as in :
            # https://github.com/candice-fraisse/octo_workshop_drift/blob/main/drift_detector_multivariate_md3.py
            w = self.model.coef_
            w = w.T
            b = self.model.intercept_
            dp = abs(np.dot(X, w) + b)
            self.rho = np.array(dp < 1).sum() / X.shape[0]
            
            if self.rho < self.rho_min:
                self.rho_min = self.rho
                
            if self.rho > self.rho_max:
                self.rho_max = self.rho
                
            # print(self.rho_max - self.rho_min)
            if self.rho_max - self.rho_min > self.sigma:
                self._is_drift = True
                self.first = True
            else:
                self._is_drift = False