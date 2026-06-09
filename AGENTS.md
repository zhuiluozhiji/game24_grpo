# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

NLP course project — **Topic 3: 24-Point Game Solving with GRPO Reinforcement Learning**.

Trains Qwen2.5-1.5B-Instruct to solve the 24-point game using SFT warm-up plus GRPO (Group Relative Policy Optimization) with verifiable rewards (RLVR), following the DeepSeek-R1 paradigm. Countdown 3-4 number arbitrary-target solving is included as the bonus extension.

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
# Main 1.5B experiment pipeline
bash scripts/run_15b_mainline.sh "$QWEN_15B" 0

# Countdown bonus pipeline
bash scripts/run_countdown_bonus.sh "$QWEN_15B" 0

# Quick legacy smoke test
python game24/run_experiment.py --quick --base_model "$QWEN_15B"

# Evaluate only (after training)
python game24/evaluate.py --model_path ./output/game24/final_model

# Test expression evaluator
python -c "from game24.utils import validate_solution; print(validate_solution([1,2,3,4], '(1+2+3)*4'))"
```

## Key Design Decisions

- **Main model**: Qwen2.5-1.5B-Instruct. LoRA rank=16.
- **Rewards**: Format (think/answer tags, weight 0.3) + Accuracy (expression evaluates to 24, weight 0.7).
- **Data**: Train on `nlile/24-game` solvable=True. Test on held-out ID, official `test-time-compute/game-of-24`, ToT hard 900-1000, and unsolvable hallucination check.
- **Bonus**: `Jiayi-Pan/Countdown-Tasks-3to4` uses per-example arbitrary `target`.
- **GRPO params**: num_generations=4, beta=0.04, temperature=0.9.
- **Archived results**: Existing 0.5B and 3B outputs are preserved as historical pre-experiments, not formal model-size comparisons.

## Key References

- TinyZero: https://github.com/Jiayi-Pan/TinyZero
- TRL GRPO Trainer: https://huggingface.co/docs/trl/main/grpo_trainer
- Open-R1: https://github.com/huggingface/open-r1
- DeepSeek-R1 (RLVR): arXiv:2501.12948
- DeepSeekMath (GRPO): arXiv:2402.03300
