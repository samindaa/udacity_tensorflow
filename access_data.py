""" Access data """

import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

class AccessData(object):
    
    def __init__(self):
        self.pickle_file = '/home/saminda/Data/notMNIST/notMNIST.pickle';
        
        with open(self.pickle_file, 'rb') as f:
            save = pickle.load(f);
            
            self.train_dataset = save['train_dataset'];
            self.train_labels  = save['train_labels'];
            self.valid_dataset = save['valid_dataset'];
            self.valid_labels  = save['valid_labels'];
            self.test_dataset  = save['test_dataset'];
            self.test_labels   = save['test_labels'];
            
            del save;
            
        self.image_size = self.train_dataset.shape[1];
        self.num_labels = 10;    
        self.num_train_examples = self.train_dataset.shape[0];
        self.num_valid_examples = self.valid_dataset.shape[0];
        self.num_test_examples  = self.test_dataset.shape[0];
        self.tarin_batch_size_offset = 0;
        self.valid_batch_size_offset = 0;
        self.test_batch_size_offset  = 0;
            
        print('Training set:', self.train_dataset.shape, self.train_labels.shape);
        print('Validation set:', self.valid_dataset.shape, self.valid_labels.shape);
        print('Test set:', self.test_dataset.shape, self.test_labels.shape);
        
    def _reshape(self, dataset, labels, process_dataset_channels):
        if process_dataset_channels:
            dataset = dataset.reshape((-1, self.image_size, self.image_size, 1)).astype(np.float32);
        else:
            dataset = dataset.reshape((-1, self.image_size * self.image_size)).astype(np.float32);
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(self.num_labels) == labels[:,None]).astype(np.float32);
        return dataset, labels;   
     
     
    def reshap(self, process_dataset_channels=True):
        self.train_dataset, self.train_labels = self._reshape(self.train_dataset, self.train_labels,
                                                              process_dataset_channels); 
        self.valid_dataset, self.valid_labels = self._reshape(self.valid_dataset, self.valid_labels,
                                                              process_dataset_channels); 
        self.test_dataset, self.test_labels = self._reshape(self.test_dataset, self.test_labels,
                                                            process_dataset_channels); 
        
        print('Training set (reshpe):', self.train_dataset.shape, self.train_labels.shape);
        print('Validation set (reshpe):', self.valid_dataset.shape, self.valid_labels.shape);
        print('Test set (reshpe):', self.test_dataset.shape, self.test_labels.shape);
    
    def num_inputs(self):
        return self.train_dataset.shape[1];
    
    def num_outputs(self):
        return self.train_labels.shape[1];
    
    
    def next_batch_bounds(self, batch_size, batch_size_offset, num_samples):
        start = batch_size_offset;
        end = None;
        batch_size_offset += batch_size;
        
        if batch_size_offset > num_samples:
            if num_samples - start > 0:
                end = num_samples;
            else:
                start = 0;
                end = batch_size;
            
            batch_size_offset = 0;
        else:
            end = batch_size_offset;   
        
        return start, end, batch_size_offset;
    
    def next_train_batch(self, batch_size):
        start, end, self.tarin_batch_size_offset = self.next_batch_bounds(
            batch_size, 
            self.tarin_batch_size_offset,
            self.num_train_examples);
        return self.train_dataset[start:end], self.train_labels[start:end];  
    
    def next_valid_batch(self, batch_size):
        start, end, self.valid_batch_size_offset = self.next_batch_bounds(
            batch_size, 
            self.valid_batch_size_offset,
            self.num_valid_examples);
        return self.valid_dataset[start:end], self.valid_labels[start:end]; 
    
    def next_test_batch(self, batch_size):
        start, end, self.test_batch_size_offset = self.next_batch_bounds(
            batch_size, 
            self.test_batch_size_offset,
            self.num_test_examples);
        return self.test_dataset[start:end], self.test_labels[start:end];
           
        
if __name__ == '__main__':
    ad = AccessData();       
    ad.reshap(); 
    
    batch_size = 256;
    
    start_exp = False;
    
    if True:
        
        s_vec = np.zeros((3,));
        e_vec = np.zeros((3,));
        o_vec = np.zeros((3,));
        t_vec = [ad.num_train_examples, ad.num_valid_examples, ad.num_test_examples];
        x_vec = ['train', 'valid', 'test'];
        
        for _ in xrange(ad.num_train_examples):
            for i in xrange(3):
                s_vec[i], e_vec[i], o_vec[i]  = ad.next_batch_bounds(batch_size, o_vec[i], t_vec[i]);
                print('type: %s start: %d end: %d batch_size_offset: %d' % (x_vec[i],
                                                                            s_vec[i], 
                                                                            e_vec[i],
                                                                            o_vec[i]));
        
            if start_exp and s_vec[0] == 0:
                break;
            if not start_exp:
                start_exp = True;
    
    if False:
        data, labels = ad.next_train_batch(batch_size);
        
        print(data.shape, labels.shape);
        print(data[0,:]);
        #print(labels[0,:]);
        print(type(data[0,0]));
        
        data0 = np.reshape(data[0,:], (28, 28));
        print('sum: ', np.sum(data0));
        plt.imshow(data0);
        plt.show();
        
    