"""Fast evaluation for iteration during 24-point experiments."""

import argparse
import json
import os
import time

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from game24.data import SYSTEM_PROMPT, get_number_list, load_24game_dataset
from game24.config import DataConfig
from game24.rewards import format_reward
from game24.utils import extract_answer_from_completion, format_prompt, validate_solution


def generate(model, tokenizer, numbers, max_new_tokens):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_prompt(numbers)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def eval_split(model, tokenizer, dataset, name, n, solvable, max_new_tokens):
    details = []
    solved = fmt = hallucinated = errors = 0
    for i in tqdm(range(min(n, len(dataset))), desc=name):
        nums = get_number_list(dataset[i])
        try:
            t0 = time.time()
            completion = generate(model, tokenizer, nums, max_new_tokens)
            elapsed = time.time() - t0
            expr = extract_answer_from_completion(completion)
            is_fmt = format_reward([completion])[0] >= 0.9
            is_solved = False
            is_hallucination = False
            if expr is not None:
                valid, reason = validate_solution(nums, expr)
                is_solved = bool(valid)
                if (not solvable) and not valid:
                    is_hallucination = True
            else:
                reason = "no expression"
            fmt += int(is_fmt)
            solved += int(is_solved)
            hallucinated += int(is_hallucination)
            details.append({
                "numbers": nums,
                "completion": completion,
                "expression": expr,
                "is_solved": is_solved,
                "is_format_correct": is_fmt,
                "is_hallucination": is_hallucination,
                "reason": reason,
                "time": elapsed,
            })
        except Exception as e:
            errors += 1
            details.append({"numbers": nums, "error": str(e)})
    total = min(n, len(dataset))
    return {
        "dataset_name": name,
        "total": total,
        "solved": solved,
        "solve_rate": solved / total if total else 0,
        "format_correct": fmt,
        "format_rate": fmt / total if total else 0,
        "hallucinated": hallucinated,
        "hallucination_rate": hallucinated / total if (total and not solvable) else None,
        "errors": errors,
        "sample_details": details[:10],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(
        args.model_path if os.path.exists(os.path.join(args.model_path, "tokenizer_config.json")) else args.base_model,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.model_path)
    model.eval()

    data = load_24game_dataset(DataConfig())
    results = {}
    results["in_distribution"] = eval_split(model, tok, data["test_solvable"], "ID", args.n, True, args.max_new_tokens)
    if data.get("test_ood") is not None:
        results["out_of_distribution"] = eval_split(model, tok, data["test_ood"], "OOD", args.n, True, args.max_new_tokens)
    if data.get("test_unsolvable") is not None:
        results["unsolvable"] = eval_split(model, tok, data["test_unsolvable"], "Unsolvable", args.n, False, args.max_new_tokens)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: {kk: v.get(kk) for kk in ["total", "solved", "solve_rate", "format_rate", "hallucination_rate", "errors"]} for k, v in results.items()}, indent=2))
    for k, v in results.items():
        print("SAMPLES", k)
        for s in v["sample_details"][:3]:
            print(s.get("numbers"), s.get("expression"), s.get("is_solved"), (s.get("completion") or "")[:160].replace("\n", " "))


if __name__ == "__main__":
    main()
