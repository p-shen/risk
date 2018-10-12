#!/bin/bash

#SBATCH -p short   # queue name
#SBATCH -t 0-4:00       # hours:minutes runlimit after which job will be killed.
#SBATCH -n 8      # number of cores requested
#SBATCH --mem=32G # memory requested
#SBATCH -J TCGA_surv         # Job name
#SBATCH -o %j.out       # File to which standard out will be written
#SBATCH -e %j.err       # File to which standard err will be written
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Peter_Shen@hms.harvard.edu

# make dependencies
source /home/pzs2/keras/bin/activate


# model = models.Sequential()
# model.add(layers.Dense(256, input_dim=input_dim))
# model.add(layers.Dense(256, activation='sigmoid'))
# model.add(layers.Dense(128, activation='sigmoid'))
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dropout(0.25))
# model.add(layers.Dense(labels_dim, activation='linear'))

JOB_DIR=/home/pzs2/capstone/models/batch_by_type

TRAIN_STEPS=3
BATCH_SIZE=128
TRAIN_FILE=/home/pzs2/capstone/data/TrainingData.txt
EVAL_FILE=/home/pzs2/capstone/data/EvalData.txt
VALIDATION_FILE=/home/pzs2/capstone/data/TestData.txt
LEARNING_RATE=0.003
python -m trainer.task --train-files $TRAIN_FILE \
                       --eval-files $EVAL_FILE \
                       --validation-files $VALIDATION_FILE \
                       --job-dir $JOB_DIR \
                       --train-steps $TRAIN_STEPS \
                       --learning-rate $LEARNING_RATE \
                       --num-epochs 500 \
                       --early-stop 100 \
                      --train-batch-size $BATCH_SIZE
