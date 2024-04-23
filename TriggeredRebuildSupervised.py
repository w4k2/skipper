# Detection feedback framework
import numpy as np
from sklearn.metrics import balanced_accuracy_score

class TriggeredRebuildSupervised:
    def __init__(self, score_metric=balanced_accuracy_score, delta=10):
        self.score_metric = score_metric
        self.delta = delta # Number of chunks for the labels to arrive since explicit request 
        
    def process(self, stream, det, clf):
        
        self.scores = []
        self.detections = []
        self.training_chunks = []
        self.past_training_chunks = []
        
        pending_label_request_chunk_ids = []
        past_preds = []
        
        for chunk_id in range(stream.n_chunks):
            X, y = stream.get_chunk()
            
            if chunk_id == 0:
                # Train clf
                clf.partial_fit(X, y, np.unique(y))
                continue
            
            # Check if labels arrived 
            if chunk_id-self.delta in pending_label_request_chunk_ids:
                # Fit clf with data from request moment
                start, end = (
                stream.chunk_size * chunk_id-self.delta,
                stream.chunk_size * chunk_id-self.delta + stream.chunk_size,
                )

                past_X = stream.X[start:end]
                past_y = stream.y[start:end]

                
                # check for drift
                pp = past_preds.pop(0)
                det.process(past_X, past_y, pp)
                
                if det._is_drift:
                    clf.partial_fit(past_X, past_y, np.unique(past_y))
                    self.detections.append(chunk_id)
                    self.training_chunks.append(chunk_id)
                    self.past_training_chunks.append(chunk_id-self.delta)
                    
                # Remove from pending list
                pending_label_request_chunk_ids.remove(chunk_id-self.delta)
            
            # Regardless of drift -- return predictions
            preds = clf.predict(X)
            self.scores.append(self.score_metric(y, preds))
            
            # Always automatically request labels
            pending_label_request_chunk_ids.append(chunk_id)
            # Save preds for later
            past_preds.append(preds)
            
            