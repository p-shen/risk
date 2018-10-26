#!/bin/bash

today=`date '+%m_%d__%H_%M'`;

JOB_DIR="/Users/Peter/Documents/GitHub/risk/tmp/models_test/xfer_$today"
DATA_DIR="/Users/Peter/Documents/GitHub/capstonedata/all_immune"

# Training survival model
JOB_DIR_MODEL_1="$JOB_DIR/surv"

TRAIN_STEPS=10
BATCH_SIZE=128
LEARNING_RATE=0.003

TRAIN_FILE="$DATA_DIR/TrainingData.txt"
EVAL_FILE="$DATA_DIR/EvalData.txt"
VALIDATION_FILE="$DATA_DIR/TestData.txt"
TRAIN_LABEL="$DATA_DIR/surv.train.txt"
EVAL_LABEL="$DATA_DIR/surv.eval.txt"
VALIDATION_LABEL="$DATA_DIR/surv.test.txt"
LOSS_FN="negative_log_partial_likelihood"
ACT_FN="linear"
CLASS_SIZE=1
MODEL1_FILE_NAME="surv.hdf5"
                       

python -m trainer.task --train-files $TRAIN_FILE $TRAIN_LABEL \
                       --validation-files $VALIDATION_FILE $VALIDATION_LABEL \
                       --eval-files $EVAL_FILE $EVAL_LABEL \
                       --loss-fn $LOSS_FN \
                       --activation-fn $ACT_FN \
                       --class-size $CLASS_SIZE \
                       --job-dir $JOB_DIR_MODEL_1 \
                       --model-file-name $MODEL1_FILE_NAME \
                       --train-steps $TRAIN_STEPS \
                       --learning-rate $LEARNING_RATE \
                       --num-epochs 10 \
                       --early-stop 5 \
                       --train-batch-size $BATCH_SIZE \
                       --eval-frequency 3 \
                       --checkpoint-epochs 3
                       


# Training on immune gene expression
JOB_DIR_MODEL_1 = "$JOB_DIR/immune"

TRAIN_STEPS=10
BATCH_SIZE=128
LEARNING_RATE=0.003

TRAIN_FILE="$DATA_DIR/TrainingData.txt"
EVAL_FILE="$DATA_DIR/EvalData.txt"
VALIDATION_FILE="$DATA_DIR/TestData.txt"
TRAIN_LABEL="$DATA_DIR/immune.genes.train.txt"
EVAL_LABEL="$DATA_DIR/immune.genes.eval.txt"
VALIDATION_LABEL="$DATA_DIR/immune.genes.test.txt"
LOSS_FN="mean_squared_error"
ACT_FN="relu"
CLASS_SIZE=14
MODEL2_FILE_NAME="genes.hdf5"

PREV_MODEL="$JOB_DIR_MODEL_1/$MODEL1_FILE_NAME"

python -m trainer.task --train-files $TRAIN_FILE $TRAIN_LABEL \
                       --validation-files $VALIDATION_FILE $VALIDATION_LABEL \
                       --eval-files $EVAL_FILE $EVAL_LABEL \
                       --is-transfer \
                       --prev-model $PREV_MODEL \
                       --loss-fn $LOSS_FN \
                       --activation-fn $ACT_FN \
                       --class-size $CLASS_SIZE \
                       --job-dir $JOB_DIR \
                       --model-file-name $MODEL2_FILE_NAME \
                       --train-steps $TRAIN_STEPS \
                       --learning-rate $LEARNING_RATE \
                       --num-epochs 10 \
                       --early-stop 5 \
                       --train-batch-size $BATCH_SIZE \
                       --eval-frequency 3 \
                       --checkpoint-epochs 3
                       