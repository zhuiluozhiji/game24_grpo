"""SFT warm-up for 24-point GRPO experiments.

This trains a LoRA adapter on verified dataset expressions before RL. The goal is
to prevent the policy from learning the easy format-only shortcut.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from game24.config import DataConfig, LoRAConfig as LoRACfg, ModelConfig
from game24.data import SYSTEM_PROMPT, get_number_list, load_24game_dataset
from game24.solver import curriculum_examples
from game24.utils import format_prompt, validate_solution


def normalize_expr(expr: str) -> str:
    expr = expr.replace("×", "*").replace("÷", "/").replace("−", "-")
    expr = re.sub(r"\s+", "", expr)
    return expr


def pick_solution(example) -> str | None:
    numbers = get_number_list(example)
    for raw in example.get("solutions") or []:
        expr = normalize_expr(str(raw))
        valid, _ = validate_solution(numbers, expr)
        if valid:
            return expr
    return None


def format_completion(expr: str | None) -> str:
    if expr is None:
        return (
            "<think>I checked the possible arithmetic combinations and did not "
            "find a valid way to make 24 using each number exactly once.</think>\n"
            "<answer>NO_SOLUTION</answer>"
        )
    return (
        f"<think>Use the expression {expr}. It uses each given number exactly "
        f"once and evaluates to 24.</think>\n<answer>{expr}</answer>"
    )


def build_example(tokenizer, example):
    numbers = get_number_list(example)
    if "solution" in example:
        expr = example["solution"]
    else:
        expr = pick_solution(example)
    if expr is None and example.get("solvable", True):
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_prompt(numbers)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    completion = format_completion(expr)
    return prompt, completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./output/game24-sft-3b")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic_low", type=int, default=1)
    parser.add_argument("--synthetic_high", type=int, default=13)
    parser.add_argument("--synthetic_ood_low", type=int, default=10)
    parser.add_argument("--synthetic_ood_high", type=int, default=20)
    parser.add_argument("--unsolvable_limit", type=int, default=300)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = ModelConfig()
    if args.model_name:
        model_cfg.model_name = args.model_name
    lora_cfg = LoRACfg()

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name, trust_remote_code=model_cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=model_cfg.trust_remote_code,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.lora_dropout,
            bias=lora_cfg.bias,
            task_type=lora_cfg.task_type,
        ),
    )
    model.print_trainable_parameters()
    model.train()

    data_cfg = DataConfig(max_train_samples=args.max_train_samples)
    train_ds = load_24game_dataset(data_cfg)["train"]
    raw_examples = list(train_ds)
    if args.synthetic:
        raw_examples.extend(curriculum_examples(args.synthetic_low, args.synthetic_high, True))
        raw_examples.extend(curriculum_examples(args.synthetic_ood_low, args.synthetic_ood_high, True))
        raw_examples.extend(
            curriculum_examples(args.synthetic_low, args.synthetic_high, False, args.unsolvable_limit)
        )

    examples = []
    seen = set()
    for ex in raw_examples:
        key = (tuple(get_number_list(ex)), ex.get("solution"), ex.get("solvable", True))
        if key in seen:
            continue
        seen.add(key)
        built = build_example(tokenizer, ex)
        if built is not None:
            examples.append(built)

    os.makedirs(args.output_dir, exist_ok=True)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

    metrics = []
    start = time.time()
    global_step = 0
    for epoch in range(args.epochs):
        order = torch.randperm(len(examples)).tolist()
        pbar = tqdm(order, desc=f"SFT epoch {epoch + 1}")
        for idx in pbar:
            prompt, completion = examples[idx]
            full = prompt + completion + tokenizer.eos_token
            enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=args.max_length)
            prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)["input_ids"]
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            labels = input_ids.clone()
            prompt_len = min(prompt_ids.shape[1], labels.shape[1])
            labels[:, :prompt_len] = -100

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            metrics.append({"step": global_step, "epoch": epoch + 1, "loss": float(loss.detach().cpu())})
            pbar.set_postfix(loss=f"{metrics[-1]['loss']:.4f}")
            if global_step % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    summary = {
        "model": model_cfg.model_name,
        "train_examples": len(examples),
        "epochs": args.epochs,
        "total_steps": global_step,
        "learning_rate": args.learning_rate,
        "training_time_seconds": time.time() - start,
        "final_loss": metrics[-1]["loss"] if metrics else None,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "sft_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "sft_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
