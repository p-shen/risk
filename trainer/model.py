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
    partial_hazard = np.exp(-(predicted_risk+epsilon))
    censor = censor.astype(int)
    ci = concordance_index(survival_time, partial_hazard, censor)
    return ci


def model_fn(input_dim,
             labels_dim,
             loss_fn,
             activation_fn,
             learning_rate=0.001):

    # TODO support parameters to build the network
    # examples of layers
    # model.add(layers.LeakyReLU(alpha=0.1))
    # model.add(layers.Dropout(0.5))

    model = models.Sequential()
    model.add(layers.Dense(128, input_dim=input_dim,
                           kernel_regularizer=regularizers.l2(0.01),
                           activity_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    # model.add(layers.Dropout(0.5))

    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(labels_dim, activation=activation_fn))

    compile_model(model, learning_rate, loss_fn)
    return model


def model_fn_xfer(model,
                  class_size,
                  loss_fn,
                  activation_fn,
                  learning_rate=0.001,
                  freeze=False,
                  freeze_layers=-2):
    '''Remove the last layer and add in another layer for training
    freeze -- if any layers should be frozen
    freeze_layers -- how many layers from the last layer to freeze
    '''

    print("""Performing transfer learning on previous model with 
    loss fn {} and activation fn {}""".format(loss_fn, activation_fn))

    model.pop()
    model.add(layers.Dense(class_size, activation=activation_fn,
                           name="xfer_dense_output"))

    if freeze:
        for layer in model.layers[0:freeze_layers]:
            layer.trainable = False

    compile_model(model, learning_rate, loss_fn)
    return model


def compile_model(model, learning_rate, loss_fn, print_summary=True):
    print("Compiling model with loss fn {}".format(loss_fn))
    model.compile(loss=loss_fn,
                  optimizer=keras.optimizers.Adam(
                      lr=learning_rate, clipvalue=0.5, clipnorm=1.0),
                  metrics=['accuracy'])

    if print_summary:
        print(model.summary())

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


def load_savedmodel(file_path, custom_objects):
    if file_path == None:
        raise Exception("No model specified for loading.")

    model = load_model(file_path, custom_objects)
    return model
