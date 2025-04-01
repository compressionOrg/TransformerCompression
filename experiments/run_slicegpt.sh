#!/bin/bash
# conda activate slicegpt

export CUDA_VISIBLE_DEVICES=1

python run_slicegpt.py \
        --model meta-llama/Llama-2-7b-hf \
        --save-dir pruned/Llama-2-7b-hf \
        --sparsity 0.25 \
        --device cuda:0 \
        --eval-baseline \
        --no-wandb