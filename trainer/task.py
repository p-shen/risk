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
                 learning_rate,
                 job_dir,
                 eval_ci,
                 eval_generator,
                 eval_ci_generator,
                 training_ci_generator,
                 eval_steps):
        self.loss_fn = loss_fn
        self.eval_batch_size = eval_batch_size
        self.eval_frequency = eval_frequency
        self.learning_rate = learning_rate
        self.job_dir = job_dir
        self.eval_ci = eval_ci
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
                kmodel = model.load_savedmodel(checkpoints[-1], custom_objects={
                    'negative_log_partial_likelihood': model.negative_log_partial_likelihood})
                kmodel = model.compile_model(
                    kmodel, self.learning_rate, self.loss_fn)
                loss, acc = kmodel.evaluate_generator(
                    self.eval_generator,
                    steps=self.eval_steps)

                if self.eval_ci:
                    # evaluate CI index for evaluation set
                    hazard_features, surv_labels = next(self.eval_ci_generator)

                    hazard_predict = kmodel.predict(hazard_features)
                    ci = model.concordance_metric(
                        surv_labels[:, 0], hazard_predict, surv_labels[:, 1])

                    # evaluate CI index for training set
                    training_hazard_features, training_surv_labels = next(
                        self.training_ci_generator)

                    training_hazard_predict = kmodel.predict(
                        training_hazard_features)
                    ci_training = model.concordance_metric(
                        training_surv_labels[:, 0], training_hazard_predict, training_surv_labels[:, 1])

                if self.eval_ci:
                    print('\nEvaluation epoch[{}] metrics[Loss:{:.2f}, Accuracy:{:.2f}, Concordance Index Training: {:.2f}, Concordance Index Evaluation:{:.2f}]'.format(
                        epoch, loss, acc, ci_training, ci))
                    # write out concordance index to a file for graphing later
                    with open(os.path.join(self.job_dir, "concordance.tsv"), "a") as myfile:
                        myfile.write('{}\t{:.2f}\t{:.2f}\n'.format(
                            epoch, ci_training, ci))
                else:
                    print('\nEvaluation epoch[{}] metrics[Loss:{:.2f}, Accuracy:{:.2f}]'.format(
                        epoch, loss, acc))

                if self.job_dir.startswith("gs://"):
                    copy_file_to_gcs(self.job_dir, checkpoints[-1])
            else:
                print(
                    '\nEvaluation epoch[{}] (no checkpoints found)'.format(epoch))


def create_callbacks(job_dir,
                     eval_steps,
                     eval_features,
                     train_features,
                     train_labels,
                     eval_labels,
                     eval_generator,
                     train_batch_size,
                     eval_batch_size,
                     loss_fn,
                     learning_rate,
                     eval_frequency,
                     early_stop,
                     checkpoint_epochs,
                     eval_ci):
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

    # generators for the evaluation and training datasets for calculating
    # concordance index
    _, _, eval_ci_generator = gn.generator_simple(
        eval_features, eval_labels, batch_size=train_batch_size)
    _, _, training_ci_generator = gn.generator_simple(
        train_features, train_labels, batch_size=train_batch_size)

    # Continuous eval callback
    evaluation = ContinuousEval(loss_fn,
                                eval_batch_size,
                                eval_frequency,
                                learning_rate,
                                job_dir,
                                eval_ci,
                                eval_generator,
                                eval_ci_generator,
                                training_ci_generator,
                                eval_steps)

    # Tensorboard logs callback
    tblog = TensorBoard(
        log_dir=os.path.join(job_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        embeddings_freq=0
    )

    print("Done creating model checkpoints.")
    cb = [checkpoint, evaluation, early_stop, tblog]

    return cb


def dispatch(train_files,
             validation_files,
             eval_files,
             is_transfer,
             prev_model,
             loss_fn,
             activation_fn,
             class_size,
             job_dir,
             model_file_name,
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

    # check input lengths
    if (len(train_files) != len(validation_files) or len(train_files) != len(validation_files) or len(validation_files) != len(eval_files)):
        raise ValueError("Input file lengths do not match")

    # Read feature files
    train_features = gn.readFile(train_files[0])
    valid_features = gn.readFile(validation_files[0])
    eval_features = gn.readFile(eval_files[0])

    # Read labels
    train_labels = gn.readFile(train_files[1])
    valid_labels = gn.readFile(validation_files[1])
    eval_labels = gn.readFile(eval_files[1])

    if is_transfer:
        # Load model from specified path
        kmodel = model.load_savedmodel(prev_model,
                                       {"negative_log_partial_likelihood": model.negative_log_partial_likelihood})

    print("Creating data generators...")
    # Create data generators based on each file
    if (loss_fn == "negative_log_partial_likelihood"):
        # generate survival generators
        train_steps_gen, train_input_size, train_generator = gn.generator_survival(
            train_features, train_labels, shuffle=True, batch_size=train_batch_size,
            batch_by_type=BATCH_BY_TYPE, normalize=NORMALIZE)
        valid_steps_gen, valid_input_size, val_generator = gn.generator_survival(
            valid_features, valid_labels, shuffle=True, batch_size=train_batch_size,
            batch_by_type=BATCH_BY_TYPE, normalize=NORMALIZE)
        eval_steps_gen, eval_input_size, eval_generator = gn.generator_survival(
            eval_features, eval_labels, shuffle=False, batch_size=train_batch_size,
            batch_by_type=BATCH_BY_TYPE, normalize=NORMALIZE)

        # survival custom loss function
        loss_fn = model.negative_log_partial_likelihood
        eval_ci = True
    else:
        train_steps_gen, train_input_size, train_generator = gn.generator_simple(
            train_features, train_labels, shuffle=True, batch_size=train_batch_size)
        valid_steps_gen, valid_input_size, val_generator = gn.generator_simple(
            valid_features, valid_labels, shuffle=True, batch_size=train_batch_size)
        eval_steps_gen, eval_input_size, eval_generator = gn.generator_simple(
            eval_features, eval_labels, shuffle=False, batch_size=train_batch_size)
        eval_ci = False

    if train_input_size != valid_input_size:
        raise ValueError(
            "Training input size is not the same as validation input size")

    print("Done creating data generators -- {}.".format(loss_fn))

    # TODO: Create callbacks based on these generators
    cb = create_callbacks(job_dir,
                          eval_steps,
                          eval_features,
                          train_features,
                          train_labels,
                          eval_labels,
                          eval_generator,
                          train_batch_size,
                          eval_batch_size,
                          loss_fn,
                          learning_rate,
                          eval_frequency,
                          early_stop,
                          checkpoint_epochs,
                          eval_ci)

    if is_transfer:
        kmodel = model.model_fn_xfer(
            kmodel, class_size, loss_fn, activation_fn, learning_rate=learning_rate)
    else:
        # compile model
        kmodel = model.model_fn(
            train_input_size, class_size, loss_fn, activation_fn, learning_rate=learning_rate)

    kmodel = dispatch_model_training(kmodel,
                                     job_dir,
                                     model_file_name,
                                     train_generator,
                                     train_steps,
                                     num_epochs,
                                     val_generator,
                                     cb)


def dispatch_model_training(kmodel,
                            job_dir,
                            model_file_name,
                            train_generator,
                            train_steps,
                            num_epochs,
                            val_generator,
                            cb):

    print("Started training.")

    kmodel.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_steps,
        epochs=num_epochs,
        validation_data=val_generator,
        validation_steps=10,
        verbose=1,  # for tensorboard visualization
        callbacks=cb)

    print("Saving final model as {}".format(model_file_name))

    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    if job_dir.startswith("gs://"):
        kmodel.save(model_file_name)
        copy_file_to_gcs(job_dir, model_file_name)
    else:
        kmodel.save(os.path.join(job_dir, model_file_name))

    # Convert the Keras model to TensorFlow SavedModel
    model.to_savedmodel(kmodel, os.path.join(job_dir, 'export'))

    return model


# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-files',
                        nargs='*',
                        required=True,
                        type=str,
                        help='''Training files local or GCS as a tab seperated file (.tsv).
                            The first file should be the features file followed by response files''')
    parser.add_argument('--validation-files',
                        nargs='*',
                        required=True,
                        type=str,
                        help='''Validation files local or GCS as a tab seperated file (.tsv).
                            The first file should be the features file followed by response files''')
    parser.add_argument('--eval-files',
                        nargs='*',
                        required=True,
                        type=str,
                        help='''Evaluation files local or GCS as a tab seperated file (.tsv).
                            The first file should be the features file followed by response files.
                            These will be used to evaluate the model every checkpoint''')
    parser.add_argument('--is-transfer',
                        action="store_true",
                        default=False,
                        help='''Training on a new model or on previous model. If previous,
                        need to specifiy previous model path''')
    parser.add_argument('--prev-model',
                        type=str,
                        default=None,
                        help='''Previous model file path''')
    parser.add_argument('--loss-fn',
                        required=True,
                        type=str,
                        help='Loss function to minimize')
    parser.add_argument('--activation-fn',
                        required=True,
                        type=str,
                        help='Activation functions the last layer')
    parser.add_argument('--class-size',
                        required=True,
                        type=int,
                        help='Class size for prediction')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')
    parser.add_argument('--model-file-name',
                        required=True,
                        type=str,
                        help='Name of the file to save the model as. (*.hdf5)')
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
