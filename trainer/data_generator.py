# Copyright 2018 Peter Shen. All Rights Reserved.
# MIT License

"""Creates a generator that can feed a stream of data from a file inpt"""

import numpy as np
import pandas as pd

DEBUG = False


def normalize(data):
    # perform quantile normalization

    # force data into floats for np calculations
    data = data.astype('float64')

    # add a epsilon to the data to adjust for 0 values
    data += 0.001

    # https://stackoverflow.com/questions/37935920/quantile-normalization-on-pandas-dataframe
    data /= np.max(np.abs(data), axis=0)  # scale between [0,1]
    rank_mean = data.stack().groupby(
        data.rank(method='first').stack(dropna=False).astype(int)).mean()
    data = data.rank(method='min').stack(dropna=False).astype(
        int).map(rank_mean).unstack()
    return data


def processDataLabels(input_file, batch_by_type=False, normalize=True):
    # Read in file
    data = pd.read_csv(input_file, sep="\t")

    # split into data and features
    if batch_by_type:
        features = data.iloc[:, :-3]
        cancertype = data.iloc[:, -3]
        cancertype = cancertype.astype('category')
    else:
        features = data.iloc[:, :-2]

    labels = data.iloc[:, -2:]

    # quantile normalization
    if normalize:
        features = normalize(features)

    # process into a numpy array
    features = features.values
    labels = labels.values

    if batch_by_type:
        return features, labels, cancertype
    else:
        return features, labels, None


def generator_input(features, labels, cancertype=None, shuffle=True, batch_size=64, batch_by_type=True, normalize=True):
    """
    Parses the input file and creates a generator for the input file

    Returns:
    num_batches_per_epoch -- The number of batches per epoch based on data size
    input_size -- the dimensions of the input data
    data_generator() -- the generator function to yield the features and labels
    """

    # features, labels, cancertype = processDataLabels(
    #     input_file, batch_by_type=batch_by_type, normalize=normalize)

    if (batch_by_type):
        types = cancertype.dtype.categories

    num_batches_per_epoch = int((len(features) - 1) / batch_size) + 1

    input_size = features.shape[1]

    # Sorts the batches by survival time
    def data_generator():
        while True:
            data_size = len(features)
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_features = features[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
                if batch_by_type:
                    shuffled_type = cancertype[shuffle_indices]
            else:
                shuffled_features = features
                shuffled_labels = labels
                if batch_by_type:
                    shuffled_type = cancertype

            num_batches_per_epoch = int(
                (len(shuffled_labels) - 1) / batch_size) + 1

            # Sample from the dataset for each epoch
            if batch_by_type:
                random_type = np.random.choice(types, 1)[0]
                shuffled_features = features[shuffled_type == random_type]
                shuffled_labels = labels[shuffled_type == random_type]
                data_size = len(shuffled_features)
                num_batches_per_epoch = int(
                    (len(shuffled_labels) - 1) / batch_size) + 1

                if DEBUG:
                    print(random_type)
                    print(len(shuffled_features))
                    print(num_batches_per_epoch)

            for batch_num in range(num_batches_per_epoch):
                if DEBUG:
                    print("batch num {}".format(batch_num))

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

                yield X, y

    return num_batches_per_epoch, input_size, data_generator()


def generate_validation_data(features, labels, batch_size=64):
    """
      Takes features and labels and returns a generator for the features and survival time

      Returns:
      data_generator() -- the generator function to yield the features and survival time
      """

    # Sorts the batches by survival time
    def data_generator():
        while True:
            data_size = len(features)

            shuffle_indices = np.random.permutation(np.arange(data_size))
            end_index = min(batch_size, len(shuffle_indices))
            shuffled_features = features[shuffle_indices[0:end_index, ], :]
            shuffled_labels = labels[shuffle_indices[0:end_index, ], :]

            yield shuffled_features, shuffled_labels

    return data_generator()


def generate_data():
    train = np.random.rand(100, 50)
    label = np.random.rand(100, 2)

    return(train, label)


def test_generator():
    """
    Only for testing purposes
    """
    eval_files = "data/tcga/EvalData.txt"
    BATCH_SIZE = 20

    # generator model loss calculation
    feature, labels, _ = processDataLabels(
        eval_files, batch_by_type=False, normalize=False)
    generator = generate_validation_data(
        feature, labels, BATCH_SIZE)

    return generator


if __name__ == '__main__':
    DEBUG = True

    BATCH_SIZE = 20
    BATCH_BY_TYPE = False
    NORMALIZE = False
    shuffle = True
    eval_files = "data/tcga/EvalData.txt"

    # generator model loss calculation
    eval_features_censor, eval_labels_censor, eval_cancertypes = processDataLabels(
        eval_files, batch_by_type=BATCH_BY_TYPE, normalize=NORMALIZE)
    eval_steps, eval_input_size, eval_generator_censor = generator_input(
        eval_features_censor, eval_labels_censor, shuffle=shuffle, batch_size=BATCH_SIZE, batch_by_type=BATCH_BY_TYPE, normalize=NORMALIZE)

    # generator for CI index evaluation
    eval_features_surv, eval_labels_surv, eval_cancertypes = processDataLabels(
        eval_files, batch_by_type=BATCH_BY_TYPE, normalize=NORMALIZE)
    eval_generator_surv = generate_validation_data(
        eval_features_surv, eval_labels_surv[:, 0], batch_size=BATCH_SIZE)

    # testing censor generator
    index = 0
    for _ in eval_generator_censor:
        print("index is {}".format(index))
        index += 1
        if index < 5:
            pass
        else:
            break

    # testing surv time generator
    index = 0
    for _ in eval_generator_surv:
        print("index is {}".format(index))
        index += 1
        if index < 5:
            pass
        else:
            break

    # testing data processing
    hazard_features, surv_labels = next(eval_generator_surv)
    print(hazard_features.shape)
    print(surv_labels.shape)
