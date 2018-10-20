# Copyright 2018 Peter Shen. All Rights Reserved.
# MIT License

"""Implements a deep learning model to model survival risks"""

import keras
import tensorflow as tf
from keras import backend as K
from keras import models, layers, regularizers
from keras.utils import np_utils
from keras.backend import relu, softmax
from keras.models import load_model

from .data_generator import generator_input, processDataLabels, test_generator

import numpy as np
from lifelines.utils import concordance_index

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

DEBUG = False


def negative_log_partial_likelihood(censor, risk):
    """Return the negative log-partial likelihood of the prediction
    y_true contains the survival time
    risk is the risk output from the neural network
    censor is the vector of inputs that are censored
    regularization is the regularization constant (not used currently in model)

    Uses the Keras backend to perform calculations

    Sorts the surv_time by sorted reverse time
    """

    # calculate negative log likelihood from estimated risk
    epsilon = 0.001
    risk = K.reshape(risk, [-1])  # flatten
    hazard_ratio = K.exp(risk)

    # cumsum on sorted surv time accounts for concordance
    log_risk = K.log(tf.cumsum(hazard_ratio)+epsilon)
    log_risk = K.reshape(log_risk, [-1])
    uncensored_likelihood = risk - log_risk

    # apply censor mask: 1 - dead, 0 - censor
    censored_likelihood = uncensored_likelihood * censor
    num_observed_events = K.sum(censor)
    neg_likelihood = - K.sum(censored_likelihood) / \
        tf.cast(num_observed_events, tf.float32)

    return neg_likelihood


def concordance_metric(survival_time, predicted_risk, censor):
    # calculate the concordance index
    epsilon = 0.001
    partial_hazard = np.exp(-(predicted_risk+epsilon)).flatten()
    censor = censor.astype(int)
    ci = concordance_index(survival_time, partial_hazard, censor)
    return ci


def model_fn(input_dim,
             labels_dim,
             loss_fn,
             hidden_units=[100, 70, 50, 20],
             learning_rate=0.001):

    # TODO support parameters to build the network
    # examples of layers
    # model.add(layers.LeakyReLU(alpha=0.1))
    # model.add(layers.Dropout(0.5))

    model = models.Sequential()
    model.add(layers.Dense(256, input_dim=input_dim,
                           kernel_regularizer=regularizers.l2(0.01),
                           activity_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(labels_dim, activation='relu'))

    compile_model(model, learning_rate, loss_fn)
    return model


def compile_model(model, learning_rate, loss_fn):
    model.compile(loss=loss_fn,
                  optimizer=keras.optimizers.Adam(
                      lr=learning_rate, clipvalue=0.5, clipnorm=1.0),
                  metrics=['accuracy'])
    return model


def to_savedmodel(model, export_path):
    """Convert the Keras HDF5 model into TensorFlow SavedModel."""

    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                      outputs={'income': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
        )
        builder.save()


if __name__ == '__main__':
    DEBUG = True

    survival_time = np.array(
        [361., 1919., 989., 2329., 3622., 871., 1126., 1431., 669., 791.])
    partial_hazard = np.array([0.36544856, 0.3840349, 0.36263838, 0.36480516,
                               0.36792752, 0.3759846, 0.34084716, 0.35269484, 0.329434, 0.3364767])
    censor = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 1])

    ci = concordance_metric(survival_time, partial_hazard, censor)
    print(ci)

    surv_model = load_model(filepath="/Users/Peter/Documents/GitHub/risk/tmp/models_test/type_norm_10_19__15_40/surv.hdf5", custom_objects={
        'negative_log_partial_likelihood': negative_log_partial_likelihood}, compile=True)
    surv_model = compile_model(
        surv_model, 0.003, negative_log_partial_likelihood)

    filename = "/Users/Peter/Documents/GitHub/risk/data/tcga/EvalData.txt"

    features, labels, _ = processDataLabels(
        filename, batch_by_type=False, normalize=False)
    _, _, gen = generator_input(features, labels)
    loss, acc = surv_model.evaluate_generator(
        gen,
        steps=10)

    # evaluate CI index for evaluation set
    gen = test_generator()
    hazard_features, surv_labels = next(gen)

    hazard_predict = surv_model.predict(hazard_features)
    ci = concordance_metric(
        surv_labels[:, 0], hazard_predict, surv_labels[:, 1])

    print(ci)

    neg_likelihood = negative_log_partial_likelihood(
        surv_labels[:, 1], hazard_predict)

    with K.get_session() as sess:
        print(neg_likelihood.eval())
