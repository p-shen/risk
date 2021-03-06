# Predicting cancer surival using Keras

## Data Sources

The [PreCog data set](https://precog.stanford.edu/download.php) used
for training is hosted by Stanford.

To host the file on Google Cloud Storage:

```{bash}
TRAIN_FILE=train.csv
EVAL_FILE=eval.csv

GCS_TRAIN_FILE=gs://gs_bucket/train.csv
GCS_EVAL_FILE=gs://gs_bucket/eval.csv

gsutil cp $GCS_TRAIN_FILE $TRAIN_FILE
gsutil cp $GCS_EVAL_FILE $EVAL_FILE
```

## Data Format

Training file with dimensions $N$ samples by $p*genes+survival+censor$ features:

* [N, 0:p] Expression matrix
* [N, p+1] Survival information
* [N, p+2] Censor information

### Preprocessing Files

Selects features from TCGA expression from a features file, and creates training and evaluation files.

```{bash}
python preprocess.py --expression-file "experiment/data/tcga_sample/expression.tsv" --survival-file "experiment/data/tcga_sample/survival.tsv" --features-file "experiment/data/genes.tide.txt"
```

The files should be in a `.tsv` format (tab seperated values).

## Virtual environment

Virtual environments are strongly suggested, but not required. Installing this
sample's dependencies in a new virtual environment allows you to run the sample
without changing global python packages on your system.

There are two options for the virtual environments:

* Install [Virtual](https://virtualenv.pypa.io/en/stable/) env
  * Create virtual environment `virtualenv census_keras`
  * Activate env `source census_keras/bin/activate`
* Install [Miniconda](https://conda.io/miniconda.html)
  * Create conda environment `conda create --name census_keras python=2.7`
  * Activate env `source activate census_keras`

## Install dependencies

* Install [gcloud](https://cloud.google.com/sdk/gcloud/)
* Install the python dependencies. `pip install --upgrade -r requirements.txt`

## Using local python

You can run the Keras code locally.

A sample local run can be run as:

```{bash}
TRAIN_STEPS=35
BATCH_SIZE=256
TRAIN_FILE="$DATA_DIR/TrainingData.txt"
EVAL_FILE="$DATA_DIR/EvalData.txt"
VALIDATION_FILE="$DATA_DIR/TestData.txt"
LEARNING_RATE=0.0003
python -m trainer.task --train-files $TRAIN_FILE \
                       --eval-files $EVAL_FILE \
                       --validation-files $VALIDATION_FILE \
                       --job-dir $JOB_DIR \
                       --train-steps $TRAIN_STEPS \
                       --learning-rate $LEARNING_RATE \
                       --num-epochs 100 \
                       --early-stop 10 \
                       --train-batch-size $BATCH_SIZE
```


## Training using gcloud local

You can run Keras training using gcloud locally

```{bash}
JOB_DIR=surv_keras
TRAIN_STEPS=200
gcloud ml-engine local train --package-path trainer \
                             --module-name trainer.task \
                             -- \
                             --train-files $TRAIN_FILE \
                             --eval-files $EVAL_FILE \
                             --job-dir $JOB_DIR \
                             --train-steps $TRAIN_STEPS
```

## Prediction using gcloud local

You can run prediction on the SavedModel created from Keras HDF5 model

```{bash}
python preprocess.py sample.json
```

```{bash}
gcloud ml-engine local predict --model-dir=$JOB_DIR/export \
                               --json-instances sample.json
```

## Training using Cloud ML Engine

You can train the model on Cloud ML Engine

```{bash}
gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.4 \
                                    --job-dir $JOB_DIR \
                                    --package-path trainer \
                                    --module-name trainer.task \
                                    --region us-central1 \
                                    -- \
                                    --train-files $GCS_TRAIN_FILE \
                                    --eval-files $GCS_EVAL_FILE \
                                    --train-steps $TRAIN_STEPS
```

## Prediction using Cloud ML Engine

You can perform prediction on Cloud ML Engine by following the steps below.
Create a model on Cloud ML Engine

```{bash}
gcloud ml-engine models create keras_model --regions us-central1
```

Export the model binaries

```{bash}
MODEL_BINARIES=$JOB_DIR/export
```

Deploy the model to the prediction service

```{bash}
gcloud ml-engine versions create v1 --model keras_model --origin $MODEL_BINARIES --runtime-version 1.2
```

Create a processed sample from the data

```{bash}
python preprocess.py sample.json

```

Run the online prediction

```{bash}
gcloud ml-engine predict --model keras_model --version v1 --json-instances sample.json
```

## Visualize training with TensorBoard

```{bash}
tensorboard --logdir=path/to/log-directory --host=127.0.0.1
```