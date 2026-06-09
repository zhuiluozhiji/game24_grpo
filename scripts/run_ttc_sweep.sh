#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

MODEL_PATH="${1:-${QWEN_15B:-Qwen/Qwen2.5-1.5B-Instruct}}"
GPU_ID="${2:-0}"

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

N_EVAL="${N_EVAL:-200}"
N_HARD="${N_HARD:-100}"
BEST_OF_LIST="${BEST_OF_LIST:-1 4 8 16}"

BASE_DIR="./output/game24-15b-base"
SFT_DIR="./output/game24-sft-15b-curriculum"
GRPO_DIR="./output/game24-grpo-15b-curriculum"

run_bestof() {
  local label="$1"
  local out_dir="$2"
  local adapter_arg=()

  if [[ "$label" != "base" ]]; then
    if [[ ! -d "$out_dir/final_model" ]]; then
      echo "Skip $label: missing adapter $out_dir/final_model"
      return
    fi
    adapter_arg=(--model_path "$out_dir/final_model")
  fi

  mkdir -p "$out_dir"
  for best_of in $BEST_OF_LIST; do
    python game24/bestof_eval.py \
      --base_model "$MODEL_PATH" \
      "${adapter_arg[@]}" \
      --output_file "$out_dir/bestof${best_of}_eval_${N_EVAL}.json" \
      --n "$N_EVAL" \
      --n_hard "$N_HARD" \
      --best_of "$best_of" \
      --max_new_tokens 96 \
      --temperature 0.9
  done
}

run_bestof base "$BASE_DIR"
run_bestof sft "$SFT_DIR"
run_bestof grpo "$GRPO_DIR"

summary_files=(
  "$BASE_DIR"/bestof*_eval_"$N_EVAL".json
  "$SFT_DIR"/bestof*_eval_"$N_EVAL".json
  "$GRPO_DIR"/bestof*_eval_"$N_EVAL".json
)

if [[ "${#summary_files[@]}" -gt 0 ]]; then
  python game24/summarize_eval.py "${summary_files[@]}"
fi
