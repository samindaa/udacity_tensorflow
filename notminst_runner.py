""" notMNIST runner """
import tensorflow as tf
from tf_wrapper import Runner
from access_data import AccessData
from logistic_regression import LogisticRegression 
from cnn import Cnn

class NotMnistRunner(Runner):
    
    def __init__(self, access_data, tf_wrapper, batch_size = 256, num_steps = 100):
        self.access_data = access_data;
        self.tf_wrapper = tf_wrapper;
        #
        self.num_inputs = self.access_data.num_inputs();
        self.num_outputs = self.access_data.num_outputs();
        self.batch_size= batch_size;
        self.num_steps = num_steps;
        print("num_inputs: %d, num_outputs: %d" % (self.num_inputs, self.num_outputs));
        
    def run(self):
        
        with tf.Session() as sess:
            # Wraper INIT
            self.tf_wrapper.initialize_all_variables();
            
            # Summary WRITER
            tf.scalar_summary("J", self.tf_wrapper.cost_fn);
            merged_summary_op = tf.merge_all_summaries();

            # TF INIT
            tf.initialize_all_variables().run();
            
            # Summary WRITER
            summary_writer = tf.train.SummaryWriter("/tmp/tensorflow/logis_reg", 
                                                    graph=tf.get_default_graph());
           
            
            
            for i in xrange(self.num_steps):
                
                batch_data, batch_labels = self.access_data.next_train_batch(self.batch_size);
                train_feed_dict = self.tf_wrapper.feed_dict(batch_data, batch_labels);
                
                
                _, l, summary = sess.run([self.tf_wrapper.train_fn, 
                                          self.tf_wrapper.cost_fn,
                                          merged_summary_op], train_feed_dict);
                                                    
                summary_writer.add_summary(summary, i);
                
                train_feed_dict_predictions = self.tf_wrapper.feed_dict(batch_data);
                                                    
                train_predictions = sess.run([self.tf_wrapper.predict_fn], train_feed_dict_predictions);                                    
                
                valid_err_rate = self.valid_error_rate(sess);
                
                print('Minibatch loss: %.3f' % (l))
                print('Minibatch error: %.1f%%' % self.error_rate(train_predictions, batch_labels))
                print('Validation error: %.1f%%' % valid_err_rate);
            
            test_err_rate = self.test_error_rate(sess);
            print('Test error: %.1f%%' % test_err_rate);                                                       
     
    def valid_error_rate(self, sess):
        
        counter   = 0;
        err_rate  = 0;
        fin_state = False;
        
        
        while not fin_state:
            batch_data, batch_labels = self.access_data.next_valid_batch(self.batch_size);
            
            if self.access_data.valid_batch_size_offset == 0:
                fin_state = True;
            
            if fin_state:
                break;
            
            valid_feed_dict = self.tf_wrapper.feed_dict(batch_data);
            valid_predictions = sess.run([self.tf_wrapper.predict_fn], valid_feed_dict);
            err_rate += self.error_rate(valid_predictions, batch_labels);
            counter += 1;
                
        return err_rate / counter;
    
    def test_error_rate(self, sess):
        
        counter   = 0;
        err_rate  = 0;
        fin_state = False;
        
        
        while not fin_state:
            batch_data, batch_labels = self.access_data.next_test_batch(self.batch_size);
            
            if self.access_data.test_batch_size_offset == 0:
                fin_state = True;
            
            if fin_state:
                break;
            
            test_feed_dict = self.tf_wrapper.feed_dict(batch_data);
            test_predictions = sess.run([self.tf_wrapper.predict_fn], test_feed_dict);
            err_rate += self.error_rate(test_predictions, batch_labels);
            counter += 1;
                
        return err_rate / counter;            
    
def testLogisticRegression():
    access_data = AccessData();
    access_data.reshap(process_dataset_channels=False);
    # LR ML
    lr = LogisticRegression(access_data.num_inputs(), access_data.num_outputs(), 1e-3);
    notMnistRunner = NotMnistRunner(access_data, lr);
    notMnistRunner.run();
    
def testCnn():
    access_data = AccessData();
    access_data.reshap();
    # LR ML
    lr = Cnn(access_data.num_inputs(), access_data.num_outputs(), 1e-3);
    notMnistRunner = NotMnistRunner(access_data, lr, 256, 1000);
    notMnistRunner.run();    


if __name__ == '__main__':
    #testLogisticRegression();
    testCnn();