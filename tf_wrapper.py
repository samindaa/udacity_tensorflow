""" Tf Wrapper to be used in ML tasks"""

import numpy as np

class TfWrapper(object):
    
    def placeholders(self):
        raise NotImplementedError('Implement TF placeholders');
    
    def variables(self):
        raise NotImplementedError('Implement TF Varables');
    
    def model(self):
        raise NotImplementedError('Implement the ML/AI model');
    
    def cost_op(self, y_hat):
        raise NotImplementedError('Implement TF cost operation');
    
    def train_op(self, y_hat):
        raise NotImplementedError('Implement TF training operation');
    
    def predict_op(self, y_hat):
        raise NotImplementedError('Implement TF predict operation');
    
    def feed_dict(self, np_inputs, np_outputs = None):
        raise NotImplementedError('Implement TF feed_dict from numpy inputs');
    
    def initialize_all_variables(self):
        self.placeholders();
        self.variables();
        self.y_hat = self.model();
        self.cost_fn = self.cost_op(self.y_hat);
        self.train_fn = self.train_op(self.cost_fn);
        self.predict_fn = self.predict_op(self.y_hat);
    
class Runner(object):
    
    def run(self):
        raise NotImplementedError;
    
    def error_rate(self, predictions, labels):
        """Return the error rate based on dense predictions and sparse labels."""
        return  100.0 - 100.0 * np.mean(predictions == np.argmax(labels, axis=1));