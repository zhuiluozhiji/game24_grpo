#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

MODEL_PATH="${1:-${QWEN_15B:-Qwen/Qwen2.5-1.5B-Instruct}}"
GPU_ID="${2:-0}"

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

N_EVAL="${N_EVAL:-200}"
N_HARD="${N_HARD:-100}"
BEST_OF="${BEST_OF:-8}"
GRPO_EPOCHS="${GRPO_EPOCHS:-1}"
GRPO_GENERATIONS="${GRPO_GENERATIONS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
TEMPERATURE="${TEMPERATURE:-0.9}"

SFT_DIR="${SFT_DIR:-./output/game24-sft-15b-curriculum}"
BASELINE_GRPO_DIR="${BASELINE_GRPO_DIR:-./output/game24-grpo-15b-curriculum}"
ABLATION_ROOT="${ABLATION_ROOT:-./output}"
GRPO_ABLATIONS="${GRPO_ABLATIONS:-lr3e-7-s300:3e-7:300 lr3e-7-s600:3e-7:600}"

if [[ ! -d "$SFT_DIR/final_model" ]]; then
  echo "Missing SFT adapter: $SFT_DIR/final_model"
  echo "Run scripts/run_15b_mainline.sh first, or set SFT_DIR to an existing SFT output."
  exit 1
fi

run_dirs=()

for spec in $GRPO_ABLATIONS; do
  IFS=":" read -r run_name learning_rate train_samples <<< "$spec"
  if [[ -z "$run_name" || -z "$learning_rate" || -z "$train_samples" ]]; then
    echo "Invalid GRPO_ABLATIONS item: $spec"
    echo "Expected format: name:learning_rate:train_samples"
    exit 1
  fi

  out_dir="$ABLATION_ROOT/game24-grpo-15b-$run_name"
  run_dirs+=("$out_dir")

  python game24/grpo_warm.py \
    --base_model "$MODEL_PATH" \
    --init_adapter "$SFT_DIR/final_model" \
    --task game24 \
    --output_dir "$out_dir" \
    --epochs "$GRPO_EPOCHS" \
    --max_train_samples "$train_samples" \
    --learning_rate "$learning_rate" \
    --num_generations "$GRPO_GENERATIONS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE"

  python game24/quick_eval.py \
    --base_model "$MODEL_PATH" \
    --model_path "$out_dir/final_model" \
    --output_file "$out_dir/quick_eval_${N_EVAL}.json" \
    --n "$N_EVAL" \
    --n_hard "$N_HARD" \
    --max_new_tokens "$MAX_NEW_TOKENS"

  python game24/bestof_eval.py \
    --base_model "$MODEL_PATH" \
    --model_path "$out_dir/final_model" \
    --output_file "$out_dir/bestof${BEST_OF}_eval_${N_EVAL}.json" \
    --n "$N_EVAL" \
    --n_hard "$N_HARD" \
    --best_of "$BEST_OF" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE"

  python game24/plot_metrics.py "$out_dir"
done

summary_files=()
for dir in "$BASELINE_GRPO_DIR" "${run_dirs[@]}"; do
  for file in "$dir"/quick_eval_"$N_EVAL".json "$dir"/bestof"$BEST_OF"_eval_"$N_EVAL".json; do
    if [[ -f "$file" ]]; then
      summary_files+=("$file")
    fi
  done
done

if [[ "${#summary_files[@]}" -gt 0 ]]; then
  python game24/summarize_eval.py "${summary_files[@]}"
fi
