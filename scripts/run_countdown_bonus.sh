#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${QWEN_15B:-Qwen/Qwen2.5-1.5B-Instruct}}"
GPU_ID="${2:-0}"

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

COUNTDOWN_TRAIN="${COUNTDOWN_TRAIN:-3000}"
COUNTDOWN_EVAL="${COUNTDOWN_EVAL:-200}"
BEST_OF="${BEST_OF:-8}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
SFT_LR="${SFT_LR:-8e-5}"
GRPO_SAMPLES="${GRPO_SAMPLES:-300}"
GRPO_EPOCHS="${GRPO_EPOCHS:-1}"
GRPO_LR="${GRPO_LR:-8e-7}"
GRPO_GENERATIONS="${GRPO_GENERATIONS:-8}"

SFT_DIR="./output/countdown-sft-15b"
GRPO_DIR="./output/countdown-grpo-15b"

python game24/sft_warmup.py \
  --model_name "$MODEL_PATH" \
  --task countdown \
  --output_dir "$SFT_DIR" \
  --epochs "$SFT_EPOCHS" \
  --max_train_samples "$COUNTDOWN_TRAIN" \
  --learning_rate "$SFT_LR"

python game24/quick_eval.py \
  --base_model "$MODEL_PATH" \
  --model_path "$SFT_DIR/final_model" \
  --task countdown \
  --output_file "$SFT_DIR/quick_eval_${COUNTDOWN_EVAL}.json" \
  --n "$COUNTDOWN_EVAL" \
  --max_new_tokens 96

python game24/bestof_eval.py \
  --base_model "$MODEL_PATH" \
  --model_path "$SFT_DIR/final_model" \
  --task countdown \
  --output_file "$SFT_DIR/bestof${BEST_OF}_eval_${COUNTDOWN_EVAL}.json" \
  --n "$COUNTDOWN_EVAL" \
  --best_of "$BEST_OF" \
  --max_new_tokens 96 \
  --temperature 0.9

python game24/grpo_warm.py \
  --base_model "$MODEL_PATH" \
  --init_adapter "$SFT_DIR/final_model" \
  --task countdown \
  --output_dir "$GRPO_DIR" \
  --epochs "$GRPO_EPOCHS" \
  --max_train_samples "$GRPO_SAMPLES" \
  --learning_rate "$GRPO_LR" \
  --num_generations "$GRPO_GENERATIONS" \
  --max_new_tokens 96 \
  --temperature 0.9

python game24/quick_eval.py \
  --base_model "$MODEL_PATH" \
  --model_path "$GRPO_DIR/final_model" \
  --task countdown \
  --output_file "$GRPO_DIR/quick_eval_${COUNTDOWN_EVAL}.json" \
  --n "$COUNTDOWN_EVAL" \
  --max_new_tokens 96

python game24/bestof_eval.py \
  --base_model "$MODEL_PATH" \
  --model_path "$GRPO_DIR/final_model" \
  --task countdown \
  --output_file "$GRPO_DIR/bestof${BEST_OF}_eval_${COUNTDOWN_EVAL}.json" \
  --n "$COUNTDOWN_EVAL" \
  --best_of "$BEST_OF" \
  --max_new_tokens 96 \
  --temperature 0.9

python game24/plot_metrics.py "$SFT_DIR"
python game24/plot_metrics.py "$GRPO_DIR"
