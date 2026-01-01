#!/bin/bash

echo "starting..."

python run.py \
    --split_dir ./data_split \
    --model WaveletDenoisingNet \
    --batch_size 64 \
    --epochs 10 \
    --lr 1e-2 \
    --noise_type emb \
    --snr_db -4 \
    --gpu_id 0 \
    --checkpoint_dir ./checkpoints \
    --mode train \

echo "训练任务已完成!"