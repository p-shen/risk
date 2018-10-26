import numpy as np
import pandas as pd

import data_generator as gn


def test_generator():
    """
    Only for testing purposes
    """
    eval_files = "data/tcga/EvalData.txt"
    BATCH_SIZE = 20

    # generator model loss calculation
    feature, labels, _ = gn.processDataLabels(
        eval_files, batch_by_type=False, normalize=False)
    generator = gn.generator_simple(
        feature, labels, BATCH_SIZE)

    return generator


def test_input_gen_surv():
    BATCH_SIZE = 20
    BATCH_BY_TYPE = False
    NORMALIZE = False
    shuffle = True
    eval_files = "data/tcga/EvalData.txt"

    eval_features, eval_labels, eval_cancertypes = gn.processDataLabels(
        eval_files, batch_by_type=BATCH_BY_TYPE, normalize=NORMALIZE)

    # generator model loss calculation
    eval_steps, eval_input_size, eval_generator_censor = gn.generator_survival(
        eval_features, eval_labels, shuffle=shuffle, batch_size=BATCH_SIZE, batch_by_type=BATCH_BY_TYPE, normalize=NORMALIZE)

    # generator for CI index evaluation
    eval_generator_surv = gn.generator_simple(
        eval_features, eval_labels, batch_size=BATCH_SIZE)

    # testing censor generator
    index = 0
    for X, y in eval_generator_censor:
        print("index is {}".format(index))
        print(X)
        print(y)
        index += 1
        if index < 3:
            pass
        else:
            break

    # testing surv time generator
    index = 0
    for X, y in eval_generator_surv:
        print("index is {}".format(index))
        print(X)
        print(y)
        index += 1
        if index < 3:
            pass
        else:
            break

    # testing data processing
    hazard_features, surv_labels = next(eval_generator_surv)
    print(hazard_features.shape)
    print(surv_labels.shape)


if __name__ == '__main__':
    DEBUG = True

    # test_input_gen_surv()

    BATCH_SIZE = 20
    shuffle = True

    file_input = "/Users/Peter/Documents/GitHub/capstonedata/all_immune/EvalData.txt"
    label_input = "/Users/Peter/Documents/GitHub/capstonedata/all_immune/immune.genes.eval.txt"
    train_features = gn.readFile(file_input)
    train_labels = gn.readFile(label_input)

    train_steps_gen, train_input_size, train_generator = gn.generator_simple(
        train_features, train_labels, shuffle=shuffle, batch_size=BATCH_SIZE)

    index = 0
    for X, y in train_generator:
        print("index is {}".format(index))
        print(X)
        print(y)
        index += 1
        if index < 3:
            pass
        else:
            break
