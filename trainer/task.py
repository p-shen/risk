# Copyright 2017 Peter Shen. All Rights Reserved.
# MIT License

"""Orchestrates training and evaluation of a neural network"""

import argparse
import glob
import json
import os

from keras.callbacks import Callback, ModelCheckpoint, TensorBoard, EarlyStopping
from keras.models import load_model
from tensorflow.python.lib.io import file_io

import trainer.data_generator as gn
import trainer.model as model

FILE_PATH = 'checkpoint.{epoch:04d}.hdf5'
SURV_MODEL = 'surv.hdf5'

CLASS_SIZE = 1

BATCH_BY_TYPE = False
NORMALIZE = False


class ContinuousEval(Callback):
    """Continuous eval callback to evaluate the checkpoint once
       every so many epochs.
    """

    def __init__(self,
                 loss_fn,
                 eval_batch_size,
                 eval_frequency,
                 eval_files,
                 learning_rate,
                 job_dir,
                 eval_generator,
                 eval_ci_generator,
                 training_ci_generator,
                 eval_steps,
                 steps=1000):
        self.loss_fn = loss_fn
        self.eval_batch_size = eval_batch_size
        self.eval_files = eval_files
        self.eval_frequency = eval_frequency
        self.learning_rate = learning_rate
        self.job_dir = job_dir
        self.steps = steps
        self.eval_generator = eval_generator
        self.eval_ci_generator = eval_ci_generator
        self.training_ci_generator = training_ci_generator
        self.eval_steps = eval_steps

    def on_epoch_begin(self, epoch, logs={}):
        if epoch > 0 and epoch % self.eval_frequency == 0:

            # Unhappy hack to work around h5py not being able to write to GCS.
            # Force snapshots and saves to local filesystem, then copy them over to GCS.
            model_path_glob = 'checkpoint.*'
            if not self.job_dir.startswith("gs://"):
                model_path_glob = os.path.join(self.job_dir, model_path_glob)
            checkpoints = glob.glob(model_path_glob)
            if len(checkpoints) > 0:
                checkpoints.sort()
                surv_model = load_model(checkpoints[-1], custom_objects={
                                        'negative_log_partial_likelihood': model.negative_log_partial_likelihood})
                surv_model = model.compile_model(
                    surv_model, self.learning_rate, self.loss_fn)
                loss, _ = surv_model.evaluate_generator(
                    self.eval_generator,
                    steps=self.eval_steps)

                # evaluate CI index for evaluation set
                hazard_features, surv_labels = next(self.eval_ci_generator)

                hazard_predict = surv_model.predict(hazard_features)
                ci = model.concordance_metric(
                    surv_labels[:, 0], hazard_predict, surv_labels[:, 1])

                # evaluate CI index for training set
                training_hazard_features, training_surv_labels = next(
                    self.training_ci_generator)

                training_hazard_predict = surv_model.predict(
                    training_hazard_features)
                ci_training = model.concordance_metric(
                    training_surv_labels[:, 0], training_hazard_predict, training_surv_labels[:, 1])

                print('\nEvaluation epoch[{}] metrics[Loss:{:.2f}, Concordance Index Training: {:.2f}, Concordance Index Evaluation:{:.2f}]'.format(
                    epoch, loss, ci_training, ci))

                # write out concordance index to a file for graphing later
                with open(os.path.join(self.job_dir, "concordance.tsv"), "a") as myfile:
                    myfile.write('{}\t{:.2f}\t{:.2f}\n'.format(
                        epoch, ci_training, ci))

                if self.job_dir.startswith("gs://"):
                    copy_file_to_gcs(self.job_dir, checkpoints[-1])
            else:
                print(
                    '\nEvaluation epoch[{}] (no checkpoints found)'.format(epoch))


def dispatch(train_files,
             validation_files,
             eval_files,
             job_dir,
             train_steps,
             eval_steps,
             train_batch_size,
             eval_batch_size,
             learning_rate,
             eval_frequency,
             early_stop,
             first_layer_size,
             scale_factor,
             eval_num_epochs,
             num_epochs,
             checkpoint_epochs):

    try:
        os.makedirs(job_dir)
    except:
        pass

    print("Creating data generators...")

    # parse training files and create training data generators
    train_features, train_labels, _ = gn.processDataLabels(
        train_files, batch_by_type=BATCH_BY_TYPE, normalize=NORMALIZE)
    train_steps_gen, train_input_size, train_generator = gn.generator_input(
        train_features, train_labels, shuffle=True, batch_size=train_batch_size,
        batch_by_type=BATCH_BY_TYPE, normalize=NORMALIZE)

    # parse validation files and create training data generators
    valid_features, valid_labels, _ = gn.processDataLabels(
        validation_files, batch_by_type=BATCH_BY_TYPE, normalize=NORMALIZE)
    valid_steps_gen, valid_input_size, val_generator = gn.generator_input(
        valid_features, valid_labels, shuffle=True, batch_size=500,
        batch_by_type=BATCH_BY_TYPE, normalize=NORMALIZE)

    valid_feature_data, valid_labels_data = next(val_generator)

    if train_input_size != valid_input_size:
        raise ValueError(
            "Training input size is not the same as validation input size")

    surv_model = model.model_fn(
        train_input_size, CLASS_SIZE, model.negative_log_partial_likelihood, learning_rate=learning_rate)

    print("Done creating data generators.")

    print("Creating model checkpoints...")

    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    checkpoint_path = FILE_PATH
    if not job_dir.startswith("gs://"):
        checkpoint_path = os.path.join(job_dir, checkpoint_path)

    # Model checkpoint callback
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        period=checkpoint_epochs,
        mode='max')

    # Stop training early when the loss is not decreasing
    early_stop = EarlyStopping(monitor='loss',
                               min_delta=0,
                               patience=early_stop,
                               verbose=0,
                               mode='auto')

    # process evaluation files
    eval_features, eval_labels, _ = gn.processDataLabels(
        eval_files, batch_by_type=BATCH_BY_TYPE, normalize=NORMALIZE)

    # generator model loss calculation
    eval_steps, _, eval_generator = gn.generator_input(
        eval_features, eval_labels, shuffle=False, batch_size=train_batch_size, batch_by_type=BATCH_BY_TYPE, normalize=NORMALIZE)

    # generators for the evaluation and training datasets for calculating
    # concordance index
    eval_ci_generator = gn.generate_validation_data(
        eval_features, eval_labels, batch_size=train_batch_size)
    training_ci_generator = gn.generate_validation_data(
        train_features, train_labels, batch_size=train_batch_size)

    # Continuous eval callback
    evaluation = ContinuousEval(model.negative_log_partial_likelihood,
                                eval_batch_size,
                                eval_frequency,
                                eval_files,
                                learning_rate,
                                job_dir,
                                eval_generator=eval_generator,
                                eval_ci_generator=eval_ci_generator,
                                training_ci_generator=training_ci_generator,
                                eval_steps=eval_steps)

    # Tensorboard logs callback
    tblog = TensorBoard(
        log_dir=os.path.join(job_dir, 'logs'),
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        write_grads=True,
        embeddings_freq=0
    )

    cb = [checkpoint, evaluation, early_stop, tblog]

    print("Done creating model checkpoints.")

    print("Started training.")

    surv_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_steps_gen,
        epochs=num_epochs,
        validation_data=(valid_feature_data, valid_labels_data),
        validation_steps=10,
        verbose=1,  # for tensorboard visualization
        callbacks=cb)

    print("Saving final model as {}".format(SURV_MODEL))

    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    if job_dir.startswith("gs://"):
        surv_model.save(SURV_MODEL)
        copy_file_to_gcs(job_dir, SURV_MODEL)
    else:
        surv_model.save(os.path.join(job_dir, SURV_MODEL))

    # Convert the Keras model to TensorFlow SavedModel
    model.to_savedmodel(surv_model, os.path.join(job_dir, 'export'))

# h5py workaround: copy local models over to GCS if the job_dir is GCS.


def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-files',
                        required=True,
                        type=str,
                        help='Training files local or GCS as a tab seperated file (.tsv)')
    parser.add_argument('--validation-files',
                        required=True,
                        type=str,
                        help='Validation files local or GCS as a tab seperated file (.tsv)')
    parser.add_argument('--eval-files',
                        required=True,
                        type=str,
                        help='Evaluation files local or GCS as a tab seperated file (.tsv)')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')
    parser.add_argument('--train-steps',
                        type=int,
                        default=100,
                        help="""\
                       Maximum number of training steps to perform
                       Training steps are in the units of training-batch-size.
                       So if train-steps is 500 and train-batch-size if 100 then
                       at most 500 * 100 training instances will be used to train.
                      """)
    parser.add_argument('--eval-steps',
                        help='Number of steps to run evalution for at each checkpoint',
                        default=100,
                        type=int)
    parser.add_argument('--train-batch-size',
                        type=int,
                        default=40,
                        help='Batch size for training steps')
    parser.add_argument('--eval-batch-size',
                        type=int,
                        default=40,
                        help='Batch size for evaluation steps')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.003,
                        help='Learning rate for SGD')
    parser.add_argument('--eval-frequency',
                        type=int,
                        default=5,
                        help='Perform one evaluation per n epochs')
    parser.add_argument('--early-stop',
                        type=int,
                        default=5,
                        help='Stop training after loss is not decreasing for n epochs')
    parser.add_argument('--first-layer-size',
                        type=int,
                        default=256,
                        help='Number of nodes in the first layer of DNN')
    parser.add_argument('--scale-factor',
                        type=float,
                        default=0.25,
                        help="""\
                      Rate of decay size of layer for Deep Neural Net.
                      max(2, int(first_layer_size * scale_factor**i)) \
                      """)
    parser.add_argument('--eval-num-epochs',
                        type=int,
                        default=1,
                        help='Number of epochs during evaluation')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=20,
                        help='Maximum number of epochs on which to train')
    parser.add_argument('--checkpoint-epochs',
                        type=int,
                        default=5,
                        help='Checkpoint per n training epochs')
    parse_args, unknown = parser.parse_known_args()

    dispatch(**parse_args.__dict__)
