# Copyright 2018 Peter Shen. All Rights Reserved.
# MIT License

"""Implements a deep learning model to model survival risks"""

import keras
import tensorflow as tf
from keras import backend as K
from keras import models, layers
from keras.utils import np_utils
from keras.backend import relu, softmax

import numpy as np
from lifelines.utils import concordance_index

# Python2/3 compatibility imports
from six.moves.urllib import parse as urlparse
from builtins import range

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


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
    hazard_ratio = K.exp(risk)
    # cumsum on sorted surv time accounts for concordance
    log_risk = K.log(tf.cumsum(hazard_ratio))
    uncensored_likelihood = risk - log_risk
    censored_likelihood = uncensored_likelihood * censor
    num_observed_events = K.sum(censor)
    neg_likelihood = - K.sum(censored_likelihood) / \
        tf.cast(num_observed_events, tf.float32)
    return neg_likelihood


def concordance_metric(survival_time, predicted_risk, censor):
    # calculate the concordance index
    partial_hazard = np.exp(-predicted_risk)
    ci = concordance_index(survival_time, partial_hazard, censor)
    return ci


def model_fn(input_dim,
             labels_dim,
             loss_fn,
             hidden_units=[100, 70, 50, 20],
             learning_rate=0.001):

    # TODO support parameters to build the network
    model = models.Sequential()
    model.add(layers.Dense(128, input_dim=input_dim))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(labels_dim, activation='linear'))

    compile_model(model, learning_rate, loss_fn)
    return model


def compile_model(model, learning_rate, loss_fn):
    model.compile(loss=loss_fn,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
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