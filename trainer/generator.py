# Copyright 2018 Peter Shen. All Rights Reserved.
# MIT License

"""Creates a generator that can feed a stream of data from a file inpt"""

import numpy as np
import pandas as pd


def normalize(data):
    # perform quantile normalization
    # https://stackoverflow.com/questions/37935920/quantile-normalization-on-pandas-dataframe
    data /= np.max(np.abs(data), axis=0)  # scale between [0,1]
    rank_mean = data.stack().groupby(
        data.rank(method='first').stack().astype(int)).mean()
    data = data.rank(method='min').stack().astype(int).map(rank_mean).unstack()
    return data


def processDataLabels(input_file):
    # Read in file
    data = pd.read_csv(input_file, sep="\t")

    # split into data and features
    features = data.iloc[:, :-2]
    labels = data.iloc[:, -2:]

    # quantile normalization
    features = normalize(features)

    # process into a numpy array
    features = features.values
    labels = labels.values
    return features, labels


def generator_input(input_file, shuffle=True, batch_size=64):
    features, labels = processDataLabels(input_file)

    num_batches_per_epoch = int((len(features) - 1) / batch_size) + 1

    input_size = features.shape[1]

    # Sorts the batches by survival time
    def data_generator():
        data_size = len(features)
        while True:
            # Sample from the dataset for each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_features = features[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_features = features
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X, y = shuffled_features[start_index:
                                         end_index], shuffled_labels[start_index: end_index]

                # Sort X and y by survival time in each batch
                # This is required for the negative binomial log likelihood to work as a loss function
                idx = np.argsort(abs(y[:, 0]))[::-1]
                X = X[idx, :]
                # sort by survival time and take censored data
                y = y[idx, 1].reshape(-1, 1)

                # reshape for matmul
                y = y.reshape(-1, 1)  # reshape to [n, 1] for matmul

                yield X, y

    return num_batches_per_epoch, input_size, data_generator()

# TODO return labels as a function of training data


def generate_data():
    train = np.random.rand(100, 50)
    label = np.random.rand(100, 2)

    return(train, label)


if __name__ == '__main__':
    BATCH_SIZE = 20
    train, label = generate_data()
    # train_steps, train_batches = batch_iter(train, label, 20)
    # print(next(train_batches))
