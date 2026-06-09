#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${QWEN_15B:-Qwen/Qwen2.5-1.5B-Instruct}}"
GPU_ID="${2:-0}"

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

N_EVAL="${N_EVAL:-200}"
N_HARD="${N_HARD:-100}"
BEST_OF="${BEST_OF:-8}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
SFT_LR="${SFT_LR:-8e-5}"
GRPO_SAMPLES="${GRPO_SAMPLES:-300}"
GRPO_EPOCHS="${GRPO_EPOCHS:-1}"
GRPO_LR="${GRPO_LR:-8e-7}"
GRPO_GENERATIONS="${GRPO_GENERATIONS:-8}"

BASE_DIR="./output/game24-15b-base"
SFT_DIR="./output/game24-sft-15b-curriculum"
GRPO_DIR="./output/game24-grpo-15b-curriculum"

mkdir -p "$BASE_DIR"

python game24/quick_eval.py \
  --base_model "$MODEL_PATH" \
  --output_file "$BASE_DIR/quick_eval_${N_EVAL}.json" \
  --n "$N_EVAL" \
  --n_hard "$N_HARD" \
  --max_new_tokens 96

python game24/bestof_eval.py \
  --base_model "$MODEL_PATH" \
  --output_file "$BASE_DIR/bestof${BEST_OF}_eval_${N_EVAL}.json" \
  --n "$N_EVAL" \
  --n_hard "$N_HARD" \
  --best_of "$BEST_OF" \
  --max_new_tokens 96 \
  --temperature 0.9

python game24/sft_warmup.py \
  --model_name "$MODEL_PATH" \
  --task game24 \
  --output_dir "$SFT_DIR" \
  --epochs "$SFT_EPOCHS" \
  --learning_rate "$SFT_LR" \
  --synthetic \
  --unsolvable_limit 300

python game24/quick_eval.py \
  --base_model "$MODEL_PATH" \
  --model_path "$SFT_DIR/final_model" \
  --output_file "$SFT_DIR/quick_eval_${N_EVAL}.json" \
  --n "$N_EVAL" \
  --n_hard "$N_HARD" \
  --max_new_tokens 96

python game24/bestof_eval.py \
  --base_model "$MODEL_PATH" \
  --model_path "$SFT_DIR/final_model" \
  --output_file "$SFT_DIR/bestof${BEST_OF}_eval_${N_EVAL}.json" \
  --n "$N_EVAL" \
  --n_hard "$N_HARD" \
  --best_of "$BEST_OF" \
  --max_new_tokens 96 \
  --temperature 0.9

python game24/grpo_warm.py \
  --base_model "$MODEL_PATH" \
  --init_adapter "$SFT_DIR/final_model" \
  --task game24 \
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
  --output_file "$GRPO_DIR/quick_eval_${N_EVAL}.json" \
  --n "$N_EVAL" \
  --n_hard "$N_HARD" \
  --max_new_tokens 96

python game24/bestof_eval.py \
  --base_model "$MODEL_PATH" \
  --model_path "$GRPO_DIR/final_model" \
  --output_file "$GRPO_DIR/bestof${BEST_OF}_eval_${N_EVAL}.json" \
  --n "$N_EVAL" \
  --n_hard "$N_HARD" \
  --best_of "$BEST_OF" \
  --max_new_tokens 96 \
  --temperature 0.9

python game24/plot_metrics.py "$SFT_DIR"
python game24/plot_metrics.py "$GRPO_DIR"
