# Copyright 2018 Peter Shen. All Rights Reserved.
# MIT License


"""Pre-processing script for processing training and evaluation data."""

import pandas as pd
import numpy as np
import csv
import argparse

DEBUG = True


def select_features(data, features):
    # select unique features from dataset

    data = data.reindex(np.unique(features), axis=1)
    data = data.dropna(axis=1)
    return data


def process_files(expression_file, survival_file, features_file):
    print(expression_file)
    print(survival_file)
    print(features_file)
    # TODO: check TCGA data format before continuing with parsing
    data = pd.read_csv(expression_file, sep="\t")
    labels = pd.read_csv(survival_file, sep="\t")

    # select features based on features input
    with open(features_file) as f:
        features = f.read().splitlines()

    data = select_features(data, features)  # select the features from dataset

    # join the data to the features set
    data = pd.concat([data.reset_index(drop=True),
                      labels.reset_index(drop=True)], axis=1)

    sample_index = int(data.shape[0] * 0.80)
    train_index = int(sample_index * 0.75)
    # TODO: Need to randomly split training and evaluation files
    data.iloc[0:train_index, :].to_csv("train.tsv", sep="\t")
    data.iloc[(train_index+1):sample_index, :].to_csv("valid.tsv", sep="\t")
    data.iloc[(sample_index+1):, :].to_csv("eval.tsv", sep="\t")


if __name__ == "__main__":

    if DEBUG:
        process_files("experiment/data/tcga_sample/expression.tsv",
                      "experiment/data/tcga_sample/survival.tsv",
                      "experiment/data/genes.tide.txt")
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--expression-file',
                            required=True,
                            type=str,
                            help='Expression matrix')
        parser.add_argument('--survival-file',
                            required=True,
                            type=str,
                            help='Survival data for patients')
        parser.add_argument('--features-file',
                            required=True,
                            type=str,
                            help='Features to select')
        parse_args, unknown = parser.parse_known_args()
        process_files(**parse_args.__dict__)
