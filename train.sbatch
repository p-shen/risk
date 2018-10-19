#!/bin/bash

#SBATCH -p short   # queue name
#SBATCH -t 0-4:00       # hours:minutes runlimit after which job will be killed.
#SBATCH -n 8      # number of cores requested
#SBATCH --mem=20G # memory requested
#SBATCH -J ctype_norm         # Job name
#SBATCH -o %j.out       # File to which standard out will be written
#SBATCH -e %j.err       # File to which standard err will be written
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Peter_Shen@hms.harvard.edu

# make dependencies
source /home/pzs2/keras/bin/activate

today=`date '+%m_%d__%H_%M'`;

JOB_DIR="/home/pzs2/capstone/models/batch_by_type_normalized_$today"
DATA_DIR="/home/pzs2/capstone/proj/TCGA_LiuLab/pancancer_normalize_by_type"

TRAIN_STEPS=35
BATCH_SIZE=256
TRAIN_FILE="$DATA_DIR/TrainingData.txt"
EVAL_FILE="$DATA_DIR/EvalData.txt"
VALIDATION_FILE="$DATA_DIR/TestData.txt"
LEARNING_RATE=0.0001
python -m trainer.task --train-files $TRAIN_FILE \
                       --eval-files $EVAL_FILE \
                       --validation-files $VALIDATION_FILE \
                       --job-dir $JOB_DIR \
                       --train-steps $TRAIN_STEPS \
                       --learning-rate $LEARNING_RATE \
                       --num-epochs 500 \
                       --early-stop 75 \
                       --train-batch-size $BATCH_SIZE
