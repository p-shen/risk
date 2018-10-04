#!/bin/bash

#SBATCH -p short   # queue name
#SBATCH -t 0-23:59       # hours:minutes runlimit after which job will be killed.
#SBATCH -n 12      # number of cores requested
#SBATCH --mem=64G # memory requested
#SBATCH -J TCGA_surv         # Job name
#SBATCH -o %j.out       # File to which standard out will be written
#SBATCH -e %j.err       # File to which standard err will be written
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Peter_Shen@hms.harvard.edu

# make dependencies
source /home/pzs2/keras/bin/activate


JOB_DIR=/home/pzs2/capstone/models/tcga_linear_1

TRAIN_FILE=/n/groups/liu/peter/TCGA/TrainingData.txt
EVAL_FILE=/n/groups/liu/peter/TCGA/RetestData.txt
VALIDATION_FILE=/n/groups/liu/peter/TCGA/TestData.txt

LEARNING_RATE=0.003
TRAIN_BATCH_SIZE=500
EVAL_BATCH_SIZE=500
EVAL_FREQUENCY=10

python -m trainer.task --train-files $TRAIN_FILE \
                       --eval-files $EVAL_FILE \
                       --validation-files $VALIDATION_FILE \
                       --job-dir $JOB_DIR \
                       --train-batch-size $TRAIN_BATCH_SIZE \
                       --eval-batch-size $EVAL_BATCH_SIZE \
                       --learning-rate $LEARNING_RATE \
                       --eval-frequency $EVAL_FREQUENCY \
                       --num-epochs 5000
