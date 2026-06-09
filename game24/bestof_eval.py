"""Verifier-based best-of-N evaluation for 24-point models."""

import argparse
import json
import os

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from game24.config import DataConfig
from game24.data import (
    SYSTEM_PROMPT,
    get_number_list,
    get_target_value,
    load_24game_dataset,
    load_countdown_dataset,
)
from game24.eval_analysis import DifficultyTracker, example_metadata
from game24.rewards import format_reward
from game24.utils import extract_answer_from_completion, format_prompt, validate_solution


def error_type(expr, reason, is_fmt, solvable):
    if not is_fmt:
        return "format_error"
    if expr is None:
        return "refusal_or_no_answer" if solvable else "refused"
    if "Numbers mismatch" in reason:
        return "number_mismatch"
    if "Result is" in reason:
        return "wrong_value"
    if "Expression error" in reason:
        return "invalid_expression"
    return "hallucination" if not solvable else "other_error"


def completions(model, tokenizer, numbers, target, n, max_new_tokens, temperature):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_prompt(numbers, target=target)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            top_k=50,
            num_return_sequences=n,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    start = enc["input_ids"].shape[1]
    return [tokenizer.decode(row[start:], skip_special_tokens=True) for row in out]


def eval_split(model, tokenizer, dataset, name, n_items, best_of, solvable, max_new_tokens, temperature):
    solved = fmt = hallucinated = errors = 0
    error_counts = {}
    difficulty = DifficultyTracker()
    details = []
    for i in tqdm(range(min(n_items, len(dataset))), desc=name):
        row = dataset[i]
        nums = get_number_list(row)
        target = get_target_value(row)
        metadata = example_metadata(row)
        try:
            cands = completions(model, tokenizer, nums, target, best_of, max_new_tokens, temperature)
            selected = cands[0]
            selected_expr = extract_answer_from_completion(selected)
            selected_reason = "not checked"
            is_solved = False
            has_refusal = False
            refusal = None
            for cand in cands:
                expr = extract_answer_from_completion(cand)
                if expr is None:
                    has_refusal = True
                    if refusal is None:
                        refusal = cand
                    continue
                valid, reason = validate_solution(nums, expr, target=target)
                if valid:
                    selected = cand
                    selected_expr = expr
                    selected_reason = reason
                    is_solved = True
                    break
                selected_reason = reason
            if (not solvable) and (not is_solved) and refusal is not None:
                selected = refusal
                selected_expr = None
                selected_reason = "refused"
            is_fmt = format_reward([selected])[0] >= 0.9
            is_hallucination = (not solvable) and (not has_refusal) and (not is_solved)
            err_type = None if is_solved else error_type(selected_expr, selected_reason, is_fmt, solvable)
            if err_type:
                error_counts[err_type] = error_counts.get(err_type, 0) + 1
            solved += int(is_solved)
            fmt += int(is_fmt)
            hallucinated += int(is_hallucination)
            difficulty.add(metadata, is_solved)
            details.append({
                "numbers": nums,
                "target": target,
                **metadata,
                "expression": selected_expr,
                "completion": selected,
                "is_solved": is_solved,
                "is_format_correct": is_fmt,
                "is_hallucination": is_hallucination,
                "had_refusal": has_refusal,
                "reason": selected_reason,
                "error_type": err_type,
            })
        except Exception as e:
            errors += 1
            difficulty.add(metadata, False)
            details.append({"numbers": nums, "target": target, **metadata, "error": str(e)})
    total = min(n_items, len(dataset))
    return {
        "dataset_name": name,
        "total": total,
        "best_of": best_of,
        "solved": solved,
        "solve_rate": solved / total if total else 0,
        "format_correct": fmt,
        "format_rate": fmt / total if total else 0,
        "hallucinated": hallucinated,
        "hallucination_rate": hallucinated / total if (total and not solvable) else None,
        "error_counts": error_counts,
        "difficulty": difficulty.summary(),
        "errors": errors,
        "sample_details": details[:10],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--base_model", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--n_hard", type=int, default=None)
    p.add_argument("--best_of", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--task", choices=["game24", "countdown"], default="game24")
    args = p.parse_args()

    tok_source = (
        args.model_path
        if args.model_path and os.path.exists(os.path.join(args.model_path, "tokenizer_config.json"))
        else args.base_model
    )
    tok = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    if args.model_path:
        model = PeftModel.from_pretrained(model, args.model_path)
    model.eval()
    data = load_countdown_dataset(DataConfig()) if args.task == "countdown" else load_24game_dataset(DataConfig())
    if args.task == "countdown":
        results = {
            "countdown": eval_split(model, tok, data["test_countdown"], "Countdown", args.n, args.best_of, True, args.max_new_tokens, args.temperature),
        }
    else:
        results = {
            "in_distribution": eval_split(model, tok, data["test_solvable"], "ID", args.n, args.best_of, True, args.max_new_tokens, args.temperature),
            "out_of_distribution": eval_split(model, tok, data["test_ood"], "OOD", args.n, args.best_of, True, args.max_new_tokens, args.temperature),
            "unsolvable": eval_split(model, tok, data["test_unsolvable"], "Unsolvable", args.n, args.best_of, False, args.max_new_tokens, args.temperature),
        }
        if data.get("test_hard") is not None and len(data["test_hard"]) > 0:
            results["tot_hard_900_1000"] = eval_split(
                model,
                tok,
                data["test_hard"],
                "ToT Hard 900-1000",
                args.n_hard or args.n,
                args.best_of,
                True,
                args.max_new_tokens,
                args.temperature,
            )
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: {kk: v.get(kk) for kk in ["total", "best_of", "solved", "solve_rate", "format_rate", "hallucination_rate", "errors"]} for k, v in results.items()}, indent=2))
    for k, v in results.items():
        print("SAMPLES", k)
        for s in v["sample_details"][:3]:
            print(s.get("numbers"), s.get("target"), s.get("expression"), s.get("is_solved"), s.get("had_refusal"), (s.get("completion") or "")[:160].replace("\n", " "))


if __name__ == "__main__":
    main()
