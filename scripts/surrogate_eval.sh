#!/bin/bash
export BUILDINGS_BENCH=/data/local/projects/foundation/v1.1.0/BuildingsBench
export WORLD_SIZE=1
NUM_GPUS=1

torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --rdzv-backend=c10d --rdzv-endpoint=localhost:0 scripts/surrogate_train.py --ignore_scoring_rules --config MLP --use-weather temperature humidity wind_speed wind_direction global_horizontal_radiation direct_normal_radiation diffuse_horizontal_radiation --train_idx_filename train_weekly_weather.idx --val_idx_filename val_weekly_weather.idx --disable_slurm --aggregate True