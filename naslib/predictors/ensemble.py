import numpy as np
import copy

from naslib.predictors.predictor import Predictor
from naslib.predictors.feedforward import FeedforwardPredictor

class Ensemble(Predictor):
    
    def __init__(self, 
                 encoding_type=None,
                 num_ensemble=3, 
                 predictor_type='feedforward',
                 ss_type='nasbench201'):
        self.num_ensemble = num_ensemble
        self.predictor_type = predictor_type
        self.encoding_type = encoding_type
        self.ss_type = ss_type

    def get_ensemble(self):
        # TODO: if encoding_type is not None, set the encoding type

        trainable_predictors = {
            'bananas': FeedforwardPredictor(ss_type=self.ss_type,
                                            encoding_type='path'),
            'feedforward': FeedforwardPredictor(ss_type=self.ss_type,
                                                encoding_type='adjacency_one_hot'),
        }

        return [copy.deepcopy(trainable_predictors[self.predictor_type]) for _ in range(self.num_ensemble)]

    def fit(self, xtrain, ytrain, train_info=None):
        
        self.ensemble = self.get_ensemble()
        train_errors = []
        for i in range(self.num_ensemble):
            train_error = self.ensemble[i].fit(xtrain, ytrain, train_info)
            train_errors.append(train_error)
        
        return train_errors

    def query(self, xtest, info=None):
        predictions = []
        for i in range(self.num_ensemble):
            prediction = self.ensemble[i].query(xtest, info)
            predictions.append(prediction)
            
        return np.array(predictions)
