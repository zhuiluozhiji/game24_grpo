#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${QWEN_3B:-Qwen/Qwen2.5-3B-Instruct}}"
GPU_ID="${2:-0}"

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

SFT_DIR="./output/game24-sft-3b-curriculum"
GRPO_DIR="./output/game24-grpo-3b-curriculum-short"

python game24/sft_warmup.py \
  --model_name "$MODEL_PATH" \
  --output_dir "$SFT_DIR" \
  --epochs 2 \
  --learning_rate 8e-5 \
  --synthetic \
  --unsolvable_limit 300

python game24/quick_eval.py \
  --base_model "$MODEL_PATH" \
  --model_path "$SFT_DIR/final_model" \
  --output_file "$SFT_DIR/quick_eval_60.json" \
  --n 60 \
  --max_new_tokens 96

python game24/bestof_eval.py \
  --base_model "$MODEL_PATH" \
  --model_path "$SFT_DIR/final_model" \
  --output_file "$SFT_DIR/bestof8_eval_30.json" \
  --n 30 \
  --best_of 8 \
  --max_new_tokens 96 \
  --temperature 0.9

python game24/grpo_warm.py \
  --base_model "$MODEL_PATH" \
  --init_adapter "$SFT_DIR/final_model" \
  --output_dir "$GRPO_DIR" \
  --epochs 1 \
  --max_train_samples 50 \
  --learning_rate 8e-7 \
  --num_generations 4 \
  --max_new_tokens 96 \
  --temperature 0.9

python game24/quick_eval.py \
  --base_model "$MODEL_PATH" \
  --model_path "$GRPO_DIR/final_model" \
  --output_file "$GRPO_DIR/quick_eval_30.json" \
  --n 30 \
  --max_new_tokens 96

python game24/bestof_eval.py \
  --base_model "$MODEL_PATH" \
  --model_path "$GRPO_DIR/final_model" \
  --output_file "$GRPO_DIR/bestof8_eval_30.json" \
  --n 30 \
  --best_of 8 \
  --max_new_tokens 96 \
  --temperature 0.9
