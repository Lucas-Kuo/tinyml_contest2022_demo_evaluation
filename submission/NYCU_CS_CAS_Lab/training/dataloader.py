import csv
import os
import numpy as np
from tensorflow.compat.v1 import keras

def loadCSV(csvf):
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]

            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels

def txt_to_numpy(filename, row):
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1
    datamat = datamat.reshape(1, row, 1)
    return datamat

class IEGM_DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size, shuffle, size, n_classes=2):
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = size
        self.n_classes = n_classes
        self.on_epoch_end()
        
    
    def __len__(self):
        # Denotes the number of batches per epoch
         return int(np.floor(len(self.list_IDs) / self.batch_size))
     
    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(25)
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp, batchsz):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((batchsz, 1, self.size, 1))
        y = np.empty((batchsz), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            x[i] = txt_to_numpy(os.path.join('./dataset', ID), self.size)
            # Store class
            y[i] = self.labels[ID]

        return x, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    def __getitem__(self, index):
        
        list_IDs_temp = [self.list_IDs[k] for k in range(24588)]

        x, y = self.__data_generation(list_IDs_temp, 24588)

        return x, y  
    
class IEGM_DataGenerator_test(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size, shuffle, size, n_classes=2):
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = size
        self.n_classes = n_classes
        self.on_epoch_end()
        
    
    def __len__(self):
        # Denotes the number of batches per epoch
         return int(np.floor(len(self.list_IDs) / self.batch_size))
     
    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(25)
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp, batchsz):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        x = np.empty((batchsz, 1, self.size, 1))
        y = np.empty((batchsz), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            x[i] = txt_to_numpy(os.path.join('./dataset', ID), self.size)
            # Store class
            y[i] = self.labels[ID]

        return x, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    def __getitem__(self, index):

        list_IDs_temp = [self.list_IDs[k] for k in range(5625)]

        x, y = self.__data_generation(list_IDs_temp, 5625)
        return x, y





