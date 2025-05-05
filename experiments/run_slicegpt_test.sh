#!/bin/bash
# conda activate slicegpt
export CUDA_VISIBLE_DEVICES=1
MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B"
SPARSITY=0.2
SAVE_DIR=pruned/${MODEL_NAME_OR_PATH//\//-}_sparsity_${SPARSITY}
LOG_PATH=logs/${MODEL_NAME_OR_PATH//\//-}_sparsity_${SPARSITY}_pruned.log
mkdir -p $SAVE_DIR
mkdir -p logs

python run_slicegpt.py \
        --model ${MODEL_NAME_OR_PATH} \
        --save-dir ${SAVE_DIR} \
        --sparsity ${SPARSITY} \
        --device cuda:0 \
        --eval-baseline \
        --no-wandb 
