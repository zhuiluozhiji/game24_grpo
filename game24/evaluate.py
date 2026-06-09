"""Evaluation script for trained 24-Point Game GRPO model.

Evaluates:
1. Solve rate on in-distribution (held-out solvable) test set
2. Solve rate on out-of-distribution test set
3. Hallucination rate on unsolvable problems
4. Format compliance rate
5. Detailed qualitative analysis
"""

import os
import sys
import json
import re
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from game24.config import ModelConfig, DataConfig, EvalConfig
from game24.data import load_24game_dataset, get_number_list, SYSTEM_PROMPT
from game24.utils import (
    validate_solution,
    extract_answer_from_completion,
    safe_eval,
    format_prompt,
    ExpressionError,
)
from game24.rewards import format_reward


def load_trained_model(
    model_path: str,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> Tuple:
    """Load a trained (LoRA) model and tokenizer."""
    print(f"Loading base model: {base_model}")
    print(f"Loading adapter from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path if os.path.exists(os.path.join(model_path, "tokenizer_config.json")) else base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try loading with quantization, fall back to float32
    model = None
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="cpu",
            trust_remote_code=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    # Try to load LoRA adapter
    try:
        model = PeftModel.from_pretrained(model, model_path)
        print("  LoRA adapter loaded successfully.")
    except Exception as e:
        print(f"  Warning: Could not load LoRA adapter: {e}")
        print("  Using base model for evaluation.")

    model.eval()
    return model, tokenizer


def generate_solution(
    model,
    tokenizer,
    numbers: List[int],
    temperature: float = 0.6,
    max_new_tokens: int = 512,
    num_beams: int = 1,
) -> str:
    """Generate a solution for given numbers.

    Returns the raw model completion (text after the prompt).
    """
    prompt_text = format_prompt(numbers)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        formatted_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt_text}\nAssistant:"

    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    # Move to model device
    try:
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except StopIteration:
        pass

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=(temperature > 0),
            top_p=0.95,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    input_len = inputs["input_ids"].shape[1]
    completion = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    return completion


def evaluate_on_dataset(
    model,
    tokenizer,
    dataset: Dataset,
    eval_cfg: EvalConfig,
    dataset_name: str = "test",
    is_solvable: bool = True,
) -> Dict:
    """Evaluate model on a dataset of 24-point problems.

    Args:
        model: The trained model.
        tokenizer: The tokenizer.
        dataset: Dataset with 'numbers' column.
        eval_cfg: Evaluation configuration.
        dataset_name: Name for reporting.
        is_solvable: Whether the problems are known to be solvable.

    Returns:
        Dict with evaluation metrics.
    """
    print(f"\n{'=' * 50}")
    print(f"Evaluating on {dataset_name}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Solvable: {is_solvable}")
    print(f"{'=' * 50}")

    results = {
        "dataset_name": dataset_name,
        "total": 0,
        "solved": 0,
        "format_correct": 0,
        "expression_valid": 0,
        "hallucinated": 0,  # Claims solution for unsolvable problem
        "errors": 0,
        "details": [],
        "solve_times": [],
    }

    n_eval = min(eval_cfg.num_test_samples, len(dataset))

    for i in tqdm(range(n_eval), desc=f"Eval: {dataset_name}"):
        example = dataset[i]
        numbers = get_number_list(example)

        start_time = time.time()
        try:
            completion = generate_solution(
                model, tokenizer, numbers,
                temperature=eval_cfg.temperature,
                max_new_tokens=eval_cfg.max_new_tokens,
                num_beams=eval_cfg.num_beams,
            )
        except Exception as e:
            results["errors"] += 1
            results["details"].append({
                "numbers": numbers,
                "error": str(e),
            })
            continue

        solve_time = time.time() - start_time
        results["solve_times"].append(solve_time)

        # Check format
        fmt_score = format_reward([completion])[0]
        is_format_correct = fmt_score >= 0.9
        if is_format_correct:
            results["format_correct"] += 1

        # Extract and validate expression
        expression = extract_answer_from_completion(completion)
        is_solved = False
        is_hallucination = False

        if expression is not None:
            is_valid, reason = validate_solution(numbers, expression, target=24)
            if is_valid:
                results["solved"] += 1
                results["expression_valid"] += 1
                is_solved = True
            elif not is_solvable and is_valid:
                # This shouldn't happen for unsolvable problems, but just in case
                results["solved"] += 1
                is_solved = True
            elif not is_solvable:
                # Model produced an expression for an unsolvable problem
                results["hallucinated"] += 1
                is_hallucination = True
        else:
            if not is_solvable:
                # Model correctly didn't produce an answer - this is good
                pass
            else:
                # Model failed to produce an answer for a solvable problem
                pass

        results["details"].append({
            "numbers": numbers,
            "completion": completion,
            "expression": expression,
            "is_solved": is_solved,
            "is_format_correct": is_format_correct,
            "is_hallucination": is_hallucination,
            "solve_time": solve_time,
        })

    # Compute summary metrics
    total = n_eval
    results["total"] = total
    results["solve_rate"] = results["solved"] / total if total > 0 else 0
    results["format_rate"] = results["format_correct"] / total if total > 0 else 0
    results["avg_solve_time"] = np.mean(results["solve_times"]) if results["solve_times"] else 0

    if not is_solvable:
        results["hallucination_rate"] = results["hallucinated"] / total if total > 0 else 0

    return results


def print_eval_results(results: Dict):
    """Pretty-print evaluation results."""
    print(f"\n--- Results: {results['dataset_name']} ---")
    print(f"  Total samples:       {results['total']}")
    print(f"  Solved:              {results['solved']} ({results.get('solve_rate', 0)*100:.1f}%)")
    print(f"  Format correct:      {results['format_correct']} ({results.get('format_rate', 0)*100:.1f}%)")

    if 'hallucination_rate' in results:
        print(f"  Hallucination rate:  {results['hallucinated']}/{results['total']} ({results['hallucination_rate']*100:.1f}%)")

    print(f"  Avg solve time:      {results.get('avg_solve_time', 0):.2f}s")
    print(f"  Errors:              {results['errors']}")

    # Show some examples
    print("\n  Sample results:")
    for i, detail in enumerate(results['details'][:5]):
        status = "SOLVED" if detail['is_solved'] else ("HALLUC" if detail.get('is_hallucination') else "FAIL")
        print(f"  [{status}] nums={detail['numbers']} -> expr={detail['expression']}")


def run_full_evaluation(
    model_path: str,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_file: Optional[str] = None,
):
    """Run full evaluation suite."""
    print("=" * 60)
    print("24-Point Game Model Evaluation")
    print("=" * 60)

    # Load model
    model, tokenizer = load_trained_model(model_path, base_model)

    # Load datasets
    data_cfg = DataConfig()
    datasets = load_24game_dataset(data_cfg)
    eval_cfg = EvalConfig()

    all_results = {}

    # 1. Evaluate on in-distribution solvable test set
    if datasets['test_solvable'] and len(datasets['test_solvable']) > 0:
        results = evaluate_on_dataset(
            model, tokenizer, datasets['test_solvable'],
            eval_cfg, "In-Distribution Solvable Test", is_solvable=True,
        )
        print_eval_results(results)
        all_results['in_distribution'] = results

    # 2. Evaluate on OOD test set
    if datasets['test_ood'] and len(datasets['test_ood']) > 0:
        results = evaluate_on_dataset(
            model, tokenizer, datasets['test_ood'],
            eval_cfg, "Out-of-Distribution Test", is_solvable=True,
        )
        print_eval_results(results)
        all_results['out_of_distribution'] = results

    # 3. Evaluate on unsolvable set (hallucination check)
    if datasets['test_unsolvable'] and len(datasets['test_unsolvable']) > 0:
        results = evaluate_on_dataset(
            model, tokenizer, datasets['test_unsolvable'],
            eval_cfg, "Unsolvable Hallucination Check", is_solvable=False,
        )
        print_eval_results(results)
        all_results['unsolvable'] = results

    # Save results
    if output_file is None:
        output_file = os.path.join(model_path, "..", "evaluation_results.json")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Convert to serializable format
    serializable_results = {}
    for key, results in all_results.items():
        serializable_results[key] = {
            k: v for k, v in results.items()
            if k != 'details'
        }
        # Keep only summary of details
        serializable_results[key]['sample_details'] = results['details'][:10]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\nFull evaluation results saved to: {output_file}")
    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate 24-Point Game model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (LoRA adapter)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Base model name")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for results JSON")
    parser.add_argument("--quick", action="store_true",
                        help="Quick evaluation with fewer samples")

    args = parser.parse_args()

    if args.quick:
        import config
        config.EvalConfig.num_test_samples = 20

    run_full_evaluation(args.model_path, args.base_model, args.output_file)


if __name__ == "__main__":
    main()
