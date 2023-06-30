#!/bin/bash

# THIS IS AN EXAMPLE SCRIPT. 
# PLEASE CONFIGURE FOR YOUR SETUP.
NUM_GPUS=8

torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    scripts/pretrain.py --config your_model.toml
