# Detection feedback framework
import numpy as np
from sklearn.metrics import balanced_accuracy_score

class TriggeredRebuildUnsupervisedRequest:
    def __init__(self, score_metric=balanced_accuracy_score, delta=10, partial=True):
        self.score_metric = score_metric
        self.delta = delta # Number of chunks for the labels to arrive since explicit request 
        self.partial = partial

    def process(self, stream, det, clf):
                
        self.scores = []
        self.label_request_chunks = []
        self.training_chunks = []
        
        pending_label_request_chunk_ids = []
        
        for chunk_id in range(stream.n_chunks):
            # print(chunk_id == det.chunk_count, chunk_id, det.chunk_count)

            X, y = stream.get_chunk()

            if chunk_id == 0:
                # Just train clf
                if self.partial == True:
                    clf.partial_fit(X, y, np.unique(y))
                else:
                    clf.fit(X, y)
                # Request data for 1st chunk to fit the detector
                pending_label_request_chunk_ids.append(chunk_id)
                self.label_request_chunks.append(chunk_id)
                continue
                
            # Check if pending request exists
            if len(pending_label_request_chunk_ids)!=0:
                
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
                    
                    # Remove from pending list
                    pending_label_request_chunk_ids.remove(chunk_id-self.delta)
                    self.training_chunks.append(chunk_id)
                    
                    # Detection here (supervised mode)
                    det.process(past_X, past_y)
                    if det._is_drift:
                        # Request labels for current chunk
                        pending_label_request_chunk_ids.append(chunk_id)
                        self.label_request_chunks.append(chunk_id)
                        
                else:
                    try: 
                        # For Oracle detector
                        det.empty_process()
                    except:
                        pass
                        
            else:
                # Detection (unsupervised mode)
                    det.process(X, np.zeros(X.shape[0]))
                    if det._is_drift:
                        # Request labels for current chunk
                        pending_label_request_chunk_ids.append(chunk_id)
                        self.label_request_chunks.append(chunk_id)
                        
           
            # Regardless of drift -- return predictions
            preds = clf.predict(X)
            self.scores.append(self.score_metric(y, preds))
        