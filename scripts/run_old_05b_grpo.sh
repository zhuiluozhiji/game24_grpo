#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${QWEN_05B:-Qwen/Qwen2.5-0.5B-Instruct}}"
GPU_ID="${2:-0}"

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "Before running, make sure game24/config.py ModelConfig.model_name points to:"
echo "  $MODEL_PATH"
echo "The legacy training entry reads that dataclass field during training."

python game24/run_experiment.py \
  --epochs 3 \
  --learning_rate 2e-5 \
  --beta 0.04 \
  --num_generations 4 \
  --output_dir ./output/game24-grpo-full-rerun \
  --base_model "$MODEL_PATH"
