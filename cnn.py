"""cnn for noMNIST """

import tensorflow as tf
from tf_wrapper import TfWrapper

class Cnn(TfWrapper):
    
    def __init__(self, num_inputs, num_outputs, 
                 regularizer = 0.0, 
                 p_keep_conv_value = 0.8,
                 p_keep_hidden_value = 0.5):
        self.num_inputs = num_inputs;
        self.num_outputs = num_outputs;
        self.regularizer = regularizer;
        self.p_keep_conv_value = p_keep_conv_value;
        self.p_keep_hidden_value = p_keep_hidden_value;
        
        print('num_inputs: %dx%d num_outputs: %d p_keep_conv_value: %f p_keep_hidden_value: %f' 
              % (self.num_inputs, self.num_inputs, self.num_outputs, self.p_keep_conv_value, 
                 self.p_keep_hidden_value));
        
    def placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, 
                                                shape=(None, self.num_inputs , self.num_inputs, 1), 
                                                name='inputs');
        self.output_placeholer = tf.placeholder(tf.float32, 
                                                shape=(None, self.num_outputs), name='outputs');
        self.p_keep_conv       = tf.placeholder(tf.float32, name='p_keep_conv');
        self.p_keep_hidden     = tf.placeholder(tf.float32, name='p_keep_hidden');
        
    def variables(self):
        self.w_vec = {
            # 5x5 conv, 1 input sample, 32 output filters
            'wc1' : tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 32), stddev=0.1), name='wc1'),
            
            # 5x5 conv, 32 input filtes, 64 output filtes
            'wc2' : tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), stddev=0.1), name='wc2'),
            
            # 7*7*64 fc inputs, 1024 outputs to hidden
            'wfc1' : tf.Variable(tf.truncated_normal(shape=(7 * 7 * 64, 1024), stddev=0.1), name='wfc1'),
            
            # 1024 fc inputs, num_output predictions
            'wfc2' : tf.Variable(tf.truncated_normal(shape=(1024, self.num_outputs), stddev=0.1), name='wfc2')
            };
        
        self.b_vec = {
            'bc1' :  tf.Variable(tf.zeros([32]), name='bc1'),
            'bc2' :  tf.Variable(tf.zeros([64]), name='bc2'),
            'bfc1' : tf.Variable(tf.zeros([1024]), name='bfc1'),
            'bfc2' : tf.Variable(tf.zeros([self.num_outputs]), name='bfc2')
            };
        
    def _conv2d(self, x, w, b, stride = 1):
        x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME');
        x = tf.nn.bias_add(x, b);
        return tf.nn.relu(x);
    
    def _max_pool(self, x, k = 2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME');
    
    def model(self):
        
        conv1 = self._conv2d(self.input_placeholder, self.w_vec['wc1'], self.b_vec['bc1']);
        conv1 = self._max_pool(conv1);
        conv1 = tf.nn.dropout(conv1, self.p_keep_conv);
        
        conv2 = self._conv2d(conv1, self.w_vec['wc2'], self.b_vec['bc2']);
        conv2 = self._max_pool(conv2);
        conv2 = tf.nn.dropout(conv2, self.p_keep_conv);
        
        fc1   = tf.reshape(conv2, [-1, self.w_vec['wfc1'].get_shape().as_list()[0]]);
        fc1   = tf.add(tf.matmul(fc1, self.w_vec['wfc1']), self.b_vec['bfc1']);
        fc1   = tf.nn.relu(fc1);
        fc1   = tf.nn.dropout(fc1, self.p_keep_hidden);
        
        y_hat = tf.add(tf.matmul(fc1, self.w_vec['wfc2']), self.b_vec['bfc2']);
        return y_hat;
            
    def cost_op(self, y_hat):
        op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_hat, self.output_placeholer, name='cost'));
        
        for _, w in self.w_vec.iteritems():
            op += self.regularizer * tf.nn.l2_loss(w);
        
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
            feed_dict[self.p_keep_conv] = self.p_keep_conv_value;
            feed_dict[self.p_keep_hidden] = self.p_keep_hidden_value;
        else:
            feed_dict[self.p_keep_conv] = 1.0;
            feed_dict[self.p_keep_hidden] = 1.0;
            
        return feed_dict;  