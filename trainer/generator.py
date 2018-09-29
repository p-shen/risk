# Copyright 2018 Peter Shen. All Rights Reserved.
# MIT License

"""Simulate test and validation data"""

import numpy as np
import pandas as pd

def normalize(tpm):
  # perform quantile normalization
  # https://stackoverflow.com/questions/37935920/quantile-normalization-on-pandas-dataframe
  tpm /= np.max(np.abs(tpm),axis=0) # scale between [0,1]
  rank_mean = tpm.stack().groupby(tpm.rank(method='first').stack().astype(int)).mean()
  tpm = tpm.rank(method='min').stack().astype(int).map(rank_mean).unstack()
  return tpm

def splitDataLabel(data):
    data = data.iloc[:,:-2]
    labels = data.iloc[:,-2:]
    return data, labels

def generator_input(input_file, shuffle=True, batch_size=64):
    # Read in file
    data = pd.read_csv(input_file, sep="\t")
    data, labels = splitDataLabel(data) 
    
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1  

    data = normalize(data) # quantile normalization

    # Sorts the batches by survival time
    def data_generator():
        data_size = len(data)
        while True:
            # Sample from the dataset for each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X, y = shuffled_data[start_index: end_index], shuffled_labels[start_index: end_index]
                
                # Sort X and y by survival time in each batch
                idx = np.argsort(abs(y[:,0]))[::-1]
                X = X[idx, :]
                y = y[idx, 1].reshape(-1,1) # sort by survival time and take censored data

                # reshape for matmul
                y = y.reshape(-1,1) #reshape to [n, 1] for matmul
                
                yield X, y

    return num_batches_per_epoch, data_generator()

# TODO return labels based on training data
def generate_data():
    train = np.random.rand(100, 50)
    label = np.random.rand(100, 2)

    return(train,label)


if __name__=='__main__':
    BATCH_SIZE = 20
    train, label = generate_data()
    # train_steps, train_batches = batch_iter(train, label, 20)
    # print(next(train_batches))
