""" 
https://www.udacity.com/course/deep-learning--ud730

Author: Sam Abeyruwan 

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile

from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


class NotMnist(object):
    
    def __init__(self):
        np.random.seed(133);
        
        self.url = 'http://commondatastorage.googleapis.com/books1000/';
        self.train_filename = None;
        self.test_filename = None;
        self.num_classes = 10;
        self.image_size = 28;
        self.pixel_depth = 255.0;
        
        self.train_folders = None;
        self.test_folders = None;
        
        self.train_datasets = None;
        self.test_datasets = None;
        
        
    def _maybe_download(self, filename, expected_bytes, force=False):
        """Download a file if not present, and make sure it's the right size."""
        if force or not os.path.exists(filename):
            print('Attempting to download:', filename); 
            filename, _ = urlretrieve(self.url + filename, filename);
            print('\nDownload Complete!');
        statinfo = os.stat(filename);
        if statinfo.st_size == expected_bytes:
            print('Found and verified', filename);
        else:
            raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?');
        return filename; 
    
    def _maybe_extract(self, filename, force=False):
        root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
        if os.path.isdir(root) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping extraction of %s.' % (root, filename))
        else:
            print('Extracting data for %s. This may take a while. Please wait.' % root)
            tar = tarfile.open(filename)
            sys.stdout.flush()
            tar.extractall()
            tar.close()
        data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
        if len(data_folders) != self.num_classes:
            raise Exception('Expected %d folders, one per class. Found %d instead.' % (self.num_classes, len(data_folders)));
        print(data_folders);
        return data_folders;
    
    
    
    def _problem_1(self, directories):
        """ peek to some images """
        skip_count = 0;
        for directory in directories:
            image_files = os.listdir(directory);
            for image in image_files:
                image_file = os.path.join(directory, image);
                print('image_file:', image_file);
                image_data = ndimage.imread(image_file).astype(float);
                print(image_data.shape);
                
                skip_count += 1;
                if skip_count == 10:
                    plt.imshow(image_data, cmap='gray');
                    plt.show();
                    # only one
                    return;
    
    
    def _load_letter(self, folder, min_num_images):
        """Load the data for a single letter label."""
        image_files = os.listdir(folder);
        dataset = np.ndarray(shape=(len(image_files), self.image_size, self.image_size), dtype=np.float32);
        print(folder);
        num_images = 0;
        for image in image_files:
            image_file = os.path.join(folder, image);
            try:
                #image_data = (ndimage.imread(image_file).astype(float) - self.pixel_depth / 2) / self.pixel_depth;
                image_data = ndimage.imread(image_file);
                image_data = image_data.astype(np.float32);
                image_data /= self.pixel_depth;
                              
                if image_data.shape != (self.image_size, self.image_size):
                    raise Exception('Unexpected image shape: %s' % str(image_data.shape));
                dataset[num_images, :, :] = image_data;
                num_images = num_images + 1;
            except IOError as e:
                print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.');
        
        dataset = dataset[0:num_images, :, :];
        if num_images < min_num_images:
            raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images));
    
        print('Full dataset tensor:', dataset.shape);
        print('Mean:', np.mean(dataset));
        print('Standard deviation:', np.std(dataset));
        return dataset;
    
    def _maybe_pickle(self, data_folders, min_num_images_per_class, force=False):
        dataset_names = [];
        for folder in data_folders:
            set_filename = folder + '.pickle';
            dataset_names.append(set_filename);
            if os.path.exists(set_filename) and not force:
                # You may override by setting force=True.
                print('%s already present - Skipping pickling.' % set_filename);
            else:
                print('Pickling %s.' % set_filename)
                dataset = self._load_letter(folder, min_num_images_per_class);
                print('dataset:', dataset.shape);
                try:
                    with open(set_filename, 'wb') as f:
                        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL);
                        print('Pickle wrote: %s' % (set_filename));
                except Exception as e:
                    print('Unable to save data to', set_filename, ':', e);
      
        return dataset_names;
    
    
    def _make_arrays(self, nb_rows, img_size):
        if nb_rows:
            dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32);
            labels = np.ndarray(nb_rows, dtype=np.int32);
        else:
            dataset, labels = None, None;
        return dataset, labels;

    def _merge_datasets(self, pickle_files, train_size, valid_size=0):
        num_classes = len(pickle_files)
        valid_dataset, valid_labels = self._make_arrays(valid_size, self.image_size);
        train_dataset, train_labels = self._make_arrays(train_size, self.image_size);
        vsize_per_class = valid_size // num_classes
        tsize_per_class = train_size // num_classes
    
        start_v, start_t = 0, 0
        end_v, end_t = vsize_per_class, tsize_per_class
        end_l = vsize_per_class+tsize_per_class
        for label, pickle_file in enumerate(pickle_files):       
            try:
                with open(pickle_file, 'rb') as f:
                    letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class
                    
                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
            except Exception as e:
                print('Unable to process data from', pickle_file, ':', e)
                raise
    
        return valid_dataset, valid_labels, train_dataset, train_labels
    
    def _randomize(self, dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation,:,:]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels
            
    
    def initialize(self):
        self.train_filename = self._maybe_download('/home/saminda/Data/notMNIST/notMNIST_large.tar.gz', 247336696);
        self.test_filename = self._maybe_download('/home/saminda/Data/notMNIST/notMNIST_small.tar.gz', 8458043);
        
        self.train_folders = self._maybe_extract(self.train_filename);
        self.test_folders  = self._maybe_extract(self.test_filename);
        
        #self._problem_1(self.train_folders);
        
        self.train_datasets = self._maybe_pickle(self.train_folders, 45000);
        self.test_datasets  = self._maybe_pickle(self.test_folders, 1800);
            
        train_size = 200000;
        valid_size = 10000;
        test_size = 10000;

        valid_dataset, valid_labels, train_dataset, train_labels = self._merge_datasets(self.train_datasets, train_size, valid_size);
        _, _, test_dataset, test_labels = self._merge_datasets(self.test_datasets, test_size);

        print('Training:', train_dataset.shape, train_labels.shape);
        print('Validation:', valid_dataset.shape, valid_labels.shape);
        print('Testing:', test_dataset.shape, test_labels.shape);
        
        train_dataset, train_labels = self._randomize(train_dataset, train_labels);
        test_dataset, test_labels = self._randomize(test_dataset, test_labels);
        valid_dataset, valid_labels = self._randomize(valid_dataset, valid_labels);
        
        
        pickle_file = '/home/saminda/Data/notMNIST/notMNIST.pickle'

        try:
            f = open(pickle_file, 'wb')
            save = {
                'train_dataset': train_dataset,
                'train_labels': train_labels,
                'valid_dataset': valid_dataset,
                'valid_labels': valid_labels,
                'test_dataset': test_dataset,
                'test_labels': test_labels,
            };
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise
        
        statinfo = os.stat(pickle_file)
        print('Compressed pickle size:', statinfo.st_size)

    
if __name__ == '__main__':
    notmnist = NotMnist();
    notmnist.initialize();    