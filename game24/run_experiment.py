"""Complete experiment pipeline for 24-Point Game GRPO."""

# CRITICAL: Set CPU thread count BEFORE any torch import (prevents segfault on Windows)
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Usage:
#   python run_experiment.py --quick
#   python run_experiment.py --epochs 3 --output_dir ./output/game24
#   python run_experiment.py --eval_only --model_path ./output/game24/final_model

import sys
import json
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Run full 24-Point Game GRPO experiment"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run with minimal samples/epochs")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="./output/game24-grpo",
                        help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, only evaluate")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model for evaluation")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")

    args = parser.parse_args()

    print("=" * 70)
    print("  24-Point Game GRPO Experiment Pipeline")
    print("=" * 70)
    print(f"  Start time: {datetime.now().isoformat()}")
    print(f"  Mode: {'Quick test' if args.quick else 'Full training'}")
    print(f"  CPU-only: True")
    print("=" * 70)

    # Step 1: Verify environment
    print("\n[Step 1/4] Checking environment...")
    _check_environment()

    # Step 2: Training (unless eval_only)
    model_path = args.model_path

    if not args.eval_only:
        print("\n[Step 2/4] Starting training...")

        from game24.train import run_training
        from game24.config import ModelConfig, LoRAConfig as LoRACfg, DataConfig, GRPOConfig

        if args.quick:
            data_cfg = DataConfig(max_train_samples=10, num_unsolvable_test=5, load_official_eval=False)
            grpo_cfg = GRPOConfig(
                output_dir=args.output_dir,
                num_train_epochs=1,
                learning_rate=args.learning_rate,
                beta=args.beta,
                num_generations=2,      # at least 2 for proper GRPO
                max_new_tokens=128,
                save_steps=1000,
                logging_steps=2,
                warmup_ratio=0.0,
            )
        else:
            data_cfg = DataConfig(load_official_eval=False)
            grpo_cfg = GRPOConfig(
                output_dir=args.output_dir,
                num_train_epochs=args.epochs,
                learning_rate=args.learning_rate,
                beta=args.beta,
                num_generations=args.num_generations,
            )

        model_cfg = ModelConfig()
        model_cfg.model_name = args.base_model
        lora_cfg = LoRACfg()

        model, tokenizer, datasets = run_training(
            model_cfg, lora_cfg, data_cfg, grpo_cfg
        )

        model_path = os.path.join(args.output_dir, "final_model")
    else:
        if model_path is None:
            print("Error: --model_path required with --eval_only")
            sys.exit(1)

    # Step 3: Evaluation
    print("\n[Step 3/4] Running evaluation...")

    from game24.evaluate import run_full_evaluation
    from game24.config import EvalConfig
    if args.quick:
        EvalConfig.num_test_samples = 5   # minimal evaluation

    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    results = run_full_evaluation(
        model_path=model_path,
        base_model=args.base_model,
        output_file=results_file,
    )

    # Step 4: Summary
    print("\n[Step 4/4] Experiment Summary")
    print("=" * 60)

    summary_lines = []
    for key, res in results.items():
        solve_rate = res.get('solve_rate', 0) * 100
        fmt_rate = res.get('format_rate', 0) * 100
        summary_lines.append(
            f"  {key}: solve={solve_rate:.1f}%, format={fmt_rate:.1f}%"
        )
        if 'hallucination_rate' in res:
            summary_lines[-1] += f", halluc={res['hallucination_rate']*100:.1f}%"

    for line in summary_lines:
        print(line)

    # Save overall summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "base_model": args.base_model,
        "results": {
            key: {
                k: v for k, v in res.items() if k != 'details'
            }
            for key, res in results.items()
        },
    }
    summary_file = os.path.join(args.output_dir, "experiment_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nFull summary saved to: {summary_file}")
    print("=" * 60)
    print("Experiment complete!")


def _check_environment():
    """Check and report Python environment."""
    import platform
    print(f"  Python: {platform.python_version()}")
    print(f"  OS: {platform.system()} {platform.release()}")

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("  PyTorch: NOT INSTALLED")

    try:
        import transformers
        print(f"  Transformers: {transformers.__version__}")
    except ImportError:
        print("  Transformers: NOT INSTALLED")

    try:
        import trl
        print(f"  TRL: {trl.__version__}")
    except ImportError:
        print("  TRL: NOT INSTALLED (required for GRPO training)")

    try:
        import bitsandbytes
        print(f"  bitsandbytes: {bitsandbytes.__version__}")
    except ImportError:
        print("  bitsandbytes: NOT INSTALLED (will use float32 fallback)")

    try:
        import peft
        print(f"  PEFT: {peft.__version__}")
    except ImportError:
        print("  PEFT: NOT INSTALLED (required for LoRA)")

    try:
        import datasets
        print(f"  Datasets: {datasets.__version__}")
    except ImportError:
        print("  Datasets: NOT INSTALLED")


if __name__ == "__main__":
    main()
