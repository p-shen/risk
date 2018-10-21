#!/bin/bash

today=`date '+%m_%d__%H_%M'`;

JOB_DIR="/Users/Peter/Documents/GitHub/risk/tmp/models_test/type_norm_$today"
DATA_DIR="/Users/Peter/Documents/GitHub/risk/data/tcga"

TRAIN_STEPS=80
BATCH_SIZE=128
TRAIN_FILE="$DATA_DIR/TrainingData.txt"
EVAL_FILE="$DATA_DIR/EvalData.txt"
VALIDATION_FILE="$DATA_DIR/TestData.txt"
LEARNING_RATE=0.003
python -m trainer.task --train-files $TRAIN_FILE \
                       --eval-files $EVAL_FILE \
                       --validation-files $VALIDATION_FILE \
                       --job-dir $JOB_DIR \
                       --train-steps $TRAIN_STEPS \
                       --learning-rate $LEARNING_RATE \
                       --num-epochs 6 \
                       --early-stop 10 \
                       --train-batch-size $BATCH_SIZE