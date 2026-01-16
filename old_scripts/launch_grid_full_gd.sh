#!/bin/bash

for MODEL in mlp cnn resnet; do
  for LR in 0.0001 0.001 0.005 0.01; do
    for LMAX_DECAY in 0 1; do

      STEPS=$(awk -v lr="$LR" 'BEGIN { printf "%d", 100 / lr }')
      NUM_DATA=10000

      if [ "$MODEL" = "mlp" ]; then
        TRACK_FEATURE_PROTOTYPES_FROM="20260108_1246_57_lr0.01000_b8"
      elif [ "$MODEL" = "cnn" ]; then
        TRACK_FEATURE_PROTOTYPES_FROM="20260115_1632_56_lr0.01000_b64"
      elif [ "$MODEL" = "resnet" ]; then
        TRACK_FEATURE_PROTOTYPES_FROM="20260115_1547_45_lr0.01000_b128"
      fi
    
      echo "Submitting MODEL=${MODEL}, LR=${LR}, NUM_DATA=${NUM_DATA}, LMAX_DECAY=${LMAX_DECAY}, STEPS=${STEPS}, TRACK_FEATURE_PROTOTYPES_FROM=${TRACK_FEATURE_PROTOTYPES_FROM}"

      sbatch \
        --job-name="${MODEL}-cifar10-decay${LMAX_DECAY}" \
        --export=MODEL=${MODEL},LR=${LR},NUM_DATA=${NUM_DATA},LMAX_DECAY=${LMAX_DECAY},STEPS=${STEPS},TRACK_FEATURE_PROTOTYPES_FROM=${TRACK_FEATURE_PROTOTYPES_FROM} \
        train_eoss_full_gd.slurm

    done
  done
done
