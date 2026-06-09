# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

NLP course project — **Topic 3: 24-Point Game Solving with GRPO Reinforcement Learning**.

Trains Qwen2.5-0.5B-Instruct to solve the 24-point game using GRPO (Group Relative Policy Optimization) with verifiable rewards (RLVR), following the DeepSeek-R1 paradigm.

## Project Structure

```
game24/
├── config.py            # All configuration dataclasses (Model, LoRA, Data, GRPO, Eval)
├── utils.py             # Safe expression evaluator (AST-based), solution validator
├── data.py              # Dataset loading, prompt formatting for GRPOTrainer
├── rewards.py           # Format reward + accuracy reward functions
├── train.py             # GRPO training with TRL GRPOTrainer, 4-bit/LoRA support
├── evaluate.py          # Multi-dataset evaluation (ID, OOD, hallucination check)
├── run_experiment.py    # One-click experiment pipeline (train + eval)
└── requirements.txt     # Python dependencies
```

## Commands

```bash
# Quick test (50 samples, 1 epoch) — always run this first
python game24/run_experiment.py --quick

# Full training + evaluation
python game24/run_experiment.py --epochs 3 --output_dir ./output/game24

# Train only
python game24/train.py --epochs 3 --output_dir ./output/game24

# Evaluate only (after training)
python game24/evaluate.py --model_path ./output/game24/final_model

# Test expression evaluator
python -c "from game24.utils import validate_solution; print(validate_solution([1,2,3,4], '(1+2+3)*4'))"
```

## Key Design Decisions

- **CPU-only**: No vLLM, no bf16/fp16. 4-bit quantization preferred with float32 fallback.
- **Model**: Qwen2.5-0.5B-Instruct (small enough for CPU). LoRA rank=16, ~2% trainable params.
- **Rewards**: Format (think/answer tags, weight 0.3) + Accuracy (expression evaluates to 24, weight 0.7).
- **Data**: Train on `nlile/24-game` solvable=True (~1009 examples). Test on held-out + OOD + unsolvable (hallucination check).
- **GRPO params**: num_generations=4, beta=0.04, temperature=0.9.

## Key References

- TinyZero: https://github.com/Jiayi-Pan/TinyZero
- TRL GRPO Trainer: https://huggingface.co/docs/trl/main/grpo_trainer
- Open-R1: https://github.com/huggingface/open-r1
- DeepSeek-R1 (RLVR): arXiv:2501.12948
- DeepSeekMath (GRPO): arXiv:2402.03300
