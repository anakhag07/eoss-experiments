#!/bin/bash

for LR in 0.001 0.005 0.01 0.05; do
  for BATCH in 8 32 128; do

    if [[ "$LR" == "0.01" || "$LR" == "0.05" ]]; then
      STEPS=20000
    else
      STEPS=100000
    fi

    NUM_DATA=8192

    echo "Submitting LR=${LR}, NUM_DATA=${NUM_DATA}, BATCH=${BATCH}, STEPS=${STEPS}"

    sbatch \
      --export=LR=${LR},NUM_DATA=${NUM_DATA},BATCH=${BATCH},STEPS=${STEPS} \
      train_eoss_sgd_imagenet.slurm

  done
done