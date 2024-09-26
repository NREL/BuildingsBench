#!/bin/bash

# THIS IS AN EXAMPLE SCRIPT. 
# PLEASE CONFIGURE FOR YOUR SETUP.

export WORLD_SIZE=1
export WANDB_PROJECT="buildingsbenchv2.0.0"
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=1 torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    scripts/pretrain.py \
    --model TransformerWithGaussian-weather-M \
    --disable_slurm \
    --num_workers 20 \
    --train_idx_filename train_weekly_weather.idx \
    --val_idx_filename val_weekly_weather.idx \
    --disable_wandb
