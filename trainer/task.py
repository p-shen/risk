# Copyright 2017 Peter Shen. All Rights Reserved.
# MIT License

"""Orchestrates training and evaluation of a neural network"""

import argparse
import glob
import json
import os

from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from keras.models import load_model
from tensorflow.python.lib.io import file_io

import trainer.generator as gn
import trainer.model as model

FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
SURV_MODEL = 'surv.hdf5'

CLASS_SIZE = 1

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
               steps=1000):
    self.loss_fn = loss_fn
    self.eval_batch_size = eval_batch_size
    self.eval_files = eval_files
    self.eval_frequency = eval_frequency
    self.learning_rate = learning_rate
    self.job_dir = job_dir
    self.steps = steps

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
        surv_model = load_model(checkpoints[-1], custom_objects={'negative_log_partial_likelihood': model.negative_log_partial_likelihood})
        surv_model = model.compile_model(surv_model, self.learning_rate, self.loss_fn)
        eval_steps, input_size, eval_generator = gn.generator_input(self.eval_files, shuffle=False, batch_size=self.eval_batch_size)
        loss, acc = surv_model.evaluate_generator(
            eval_generator,
            steps=eval_steps)
        
        # calculate concordance index
        features, labels = gn.processDataLabels(self.eval_files)
        features = features.values
        labels = labels.values
        hazard_predict = surv_model.predict(features)
        ci = model.concordance_metric(labels[:,0], hazard_predict, labels[:,1])
        
        print('\nEvaluation epoch[{}] metrics[Loss:{:.2f}, Concordance Index:{:.2f}]'.format(
            epoch, loss, ci))
        if self.job_dir.startswith("gs://"):
          copy_file_to_gcs(self.job_dir, checkpoints[-1])
      else:
        print('\nEvaluation epoch[{}] (no checkpoints found)'.format(epoch))

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
             first_layer_size,
             scale_factor,
             eval_num_epochs,
             num_epochs,
             checkpoint_epochs):

  try:
    os.makedirs(job_dir)
  except:
    pass

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

  # Continuous eval callback
  evaluation = ContinuousEval(model.negative_log_partial_likelihood,
                              eval_batch_size,
                              eval_frequency,
                              eval_files,
                              learning_rate,
                              job_dir)

  # Tensorboard logs callback
  tblog = TensorBoard(
      log_dir=os.path.join(job_dir, 'logs'),
      histogram_freq=0,
      write_graph=True,
      embeddings_freq=0)

  cb=[checkpoint, evaluation, tblog]

  train_steps, input_size, generator = gn.generator_input(train_files, shuffle=True, batch_size=train_batch_size)
  valid_steps, input_size, val_generator = gn.generator_input(validation_files, shuffle=True, batch_size=train_batch_size)
  surv_model = model.model_fn(input_size, CLASS_SIZE, model.negative_log_partial_likelihood)

  surv_model.fit_generator(
      generator,
      steps_per_epoch=train_steps,
      epochs=num_epochs,
      validation_data=val_generator,
      validation_steps=valid_steps,
      callbacks=cb)

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
                      default=5,
                      help='Perform one evaluation per n epochs')
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
