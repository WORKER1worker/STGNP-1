#!/usr/bin/env bash
set -euo pipefail

python train_spfs.py \
  --h 24 \
  --K 5 \
  --T 48 \
  --d 64 \
  --lr 0.001 \
  --epochs 100 \
  --patience 10 \
  --data_file data/dataset/SM_NQ/SM_NQ-30-minutes_05cm.csv \
  --station_file data/dataset/SM_NQ/Stations_information_NAQU.csv \
  --test_file dataset/SM_NQ/test_nodes.npy \
  --dataset_name SM \
  --exp_name SM_NQ
