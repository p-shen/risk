#!/bin/bash

#SBATCH -p short   # queue name
#SBATCH -t 0-08:00       # hours:minutes runlimit after which job will be killed.
#SBATCH -n 16      # number of cores requested
#SBATCH --mem=32G # memory requested
#SBATCH -J xfer_chain         # Job name
#SBATCH -o %j.out       # File to which standard out will be written
#SBATCH -e %j.err       # File to which standard err will be written
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Peter_Shen@hms.harvard.edu

# make dependencies
source /home/pzs2/keras/bin/activate

today=`date '+%m_%d__%H_%M'`;

JOB_DIR="/home/pzs2/capstone/models/surv_immune_xfer_$today"
DATA_DIR="/home/pzs2/capstone/proj/TCGA_processed/pancancer_all_immune"

# Training survival model on TCGA survival information
JOB_DIR_MODEL_1="$JOB_DIR/surv"
MODEL1_FILE_NAME="surv.hdf5"

TRAIN_STEPS=80
BATCH_SIZE=128
LEARNING_RATE=0.05

TRAIN_FILE="$DATA_DIR/TrainingData.txt"
EVAL_FILE="$DATA_DIR/EvalData.txt"
VALIDATION_FILE="$DATA_DIR/TestData.txt"

TRAIN_LABEL="$DATA_DIR/surv.train.txt"
EVAL_LABEL="$DATA_DIR/surv.eval.txt"
VALIDATION_LABEL="$DATA_DIR/surv.test.txt"

LOSS_FN="negative_log_partial_likelihood"
ACT_FN="linear"
CLASS_SIZE=1
                       

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
                       --num-epochs 400 \
                       --early-stop 40 \
                       --train-batch-size $BATCH_SIZE \
                       --eval-frequency 10 \
                       --checkpoint-epochs 5
                       


# Training on immune gene expression
JOB_DIR_MODEL_2="$JOB_DIR/immune"
MODEL2_FILE_NAME="genes.hdf5"
PREV_MODEL="$JOB_DIR_MODEL_1/$MODEL1_FILE_NAME"

TRAIN_STEPS=80
BATCH_SIZE=128
LEARNING_RATE=0.0005

TRAIN_FILE="$DATA_DIR/TrainingData.txt"
EVAL_FILE="$DATA_DIR/EvalData.txt"
VALIDATION_FILE="$DATA_DIR/TestData.txt"

TRAIN_LABEL="$DATA_DIR/immune.genes.train.txt"
EVAL_LABEL="$DATA_DIR/immune.genes.eval.txt"
VALIDATION_LABEL="$DATA_DIR/immune.genes.test.txt"

LOSS_FN="mean_squared_error"
ACT_FN="relu"
CLASS_SIZE=2

python -m trainer.task --train-files $TRAIN_FILE $TRAIN_LABEL \
                       --validation-files $VALIDATION_FILE $VALIDATION_LABEL \
                       --eval-files $EVAL_FILE $EVAL_LABEL \
                       --is-transfer \
                       --prev-model $PREV_MODEL \
                       --loss-fn $LOSS_FN \
                       --activation-fn $ACT_FN \
                       --class-size $CLASS_SIZE \
                       --job-dir $JOB_DIR_MODEL_2 \
                       --model-file-name $MODEL2_FILE_NAME \
                       --train-steps $TRAIN_STEPS \
                       --learning-rate $LEARNING_RATE \
                       --num-epochs 400 \
                       --early-stop 40 \
                       --train-batch-size $BATCH_SIZE \
                       --eval-frequency 10 \
                       --checkpoint-epochs 5
                       
