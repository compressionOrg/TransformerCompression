#!/bin/bash
# conda activate slicegpt

export CUDA_VISIBLE_DEVICES=1
python run_lm_eval.py \
    --model meta-llama/Llama-2-7b-hf \
    --sliced-model-path pruned/Llama-2-7b-hf \
    --sparsity 0.25 \
    --tasks  "hellaswag" "arc_challenge" "winogrande" "arc_easy" \
    --no-wandb > logs/llama2_7b_eval.log 