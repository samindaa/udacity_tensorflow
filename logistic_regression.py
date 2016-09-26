"""Logistc regression (this is a simple model to test)"""

import tensorflow as tf
from tf_wrapper import TfWrapper

class LogisticRegression(TfWrapper):
    
    def __init__(self, num_inputs, num_outputs, regularizer = 0.0):
        self.num_inputs = num_inputs;
        self.num_outputs = num_outputs;
        self.regularizer = regularizer;
        
    def placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.num_inputs), name='inputs');
        self.output_placeholer = tf.placeholder(tf.float32, shape=(None, self.num_outputs), name='outputs');
        
    def variables(self):
        self.w = tf.Variable(tf.truncated_normal([self.num_inputs, self.num_outputs], stddev=0.1), name='w');
        self.b = tf.Variable(tf.zeros([self.num_outputs]), name='b');    
            
        
    def model(self):
        y_hat = tf.matmul(self.input_placeholder, self.w) + self.b;
        return y_hat;
            
    def cost_op(self, y_hat):
        op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_hat, self.output_placeholer, name='cost'));
        op += self.regularizer * tf.nn.l2_loss(self.w);
        return op;
    
    def train_op(self, cost):
        op = tf.train.AdamOptimizer().minimize(cost);
        return op;
    
    def predict_op(self, y_hat):
        op = tf.argmax(y_hat, 1);
        return op;
    
    def feed_dict(self, np_inputs, np_outputs = None):
        feed_dict = {self.input_placeholder : np_inputs};
        if np_outputs is not None:
            feed_dict[self.output_placeholer] = np_outputs;
        return feed_dict;  