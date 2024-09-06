# Detection feedback framework
import numpy as np
from sklearn.metrics import balanced_accuracy_score

class ContinousRebuild:
    def __init__(self, score_metric=balanced_accuracy_score, delta=10, partial=True):
        self.score_metric = score_metric
        self.delta = delta # Number of chunks for the labels to arrive since explicit request 
        self.partial = partial
        
    def process(self, stream, clf):
        
        self.scores = []        
        self.label_request_chunks = []
        self.training_chunks = []

        pending_label_request_chunk_ids = []
            
        for chunk_id in range(stream.n_chunks):
            X, y = stream.get_chunk()
            
            if chunk_id == 0:
                # Train clf
                if self.partial == True:
                    clf.partial_fit(X, y, np.unique(y))
                else:
                    clf.fit(X, y)
                continue
            
            # Check if labels arrived 
            if chunk_id-self.delta in pending_label_request_chunk_ids:
                # Fit clf with data from request moment
                start, end = (
                stream.chunk_size * (chunk_id-self.delta),
                stream.chunk_size * (chunk_id-self.delta) + stream.chunk_size,
                )

                past_X = stream.X[start:end]
                past_y = stream.y[start:end]

                if self.partial == True:
                    clf.partial_fit(past_X, past_y, np.unique(past_y))
                else:
                    clf.fit(past_X, past_y)
                self.training_chunks.append(chunk_id)
                    
                # Remove from pending list
                pending_label_request_chunk_ids.remove(chunk_id-self.delta)
            
            
            # Always automatically request labels
            pending_label_request_chunk_ids.append(chunk_id)   
            self.label_request_chunks.append(chunk_id)
            
            preds = clf.predict(X)
            self.scores.append(self.score_metric(y, preds))
                