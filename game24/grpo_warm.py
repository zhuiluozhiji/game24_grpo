"""GRPO continuation from an SFT adapter."""

import argparse
import json
import os
import time
from datetime import datetime

import torch
from peft import PeftModel
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from game24.config import DataConfig, GRPOConfig
from game24.data import (
    get_number_list,
    get_target_value,
    load_24game_dataset,
    load_countdown_dataset,
)
from game24.rewards import accuracy_reward, format_reward
from game24.train import compute_advantages, generate_completions, get_chat_messages


def grpo_step(model, optimizer, tokenizer, numbers, target, cfg, data_cfg, device):
    messages = get_chat_messages(numbers, target=target)
    model.eval()
    completions, _ = generate_completions(
        model,
        tokenizer,
        messages,
        num_generations=cfg.num_generations,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        device=device,
    )
    fmt = format_reward(completions)
    acc = accuracy_reward(
        completions,
        numbers=[numbers] * len(completions),
        targets=[target] * len(completions),
    )
    rewards = [0.05 * f + 0.95 * a for f, a in zip(fmt, acc)]
    advantages = compute_advantages(rewards)

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_len = tokenizer.encode(prompt, return_tensors="pt").shape[1]
    model.train()
    total_loss = 0.0
    n_valid = 0
    for completion, adv in zip(completions, advantages):
        if abs(adv) < 1e-8:
            continue
        enc = tokenizer(
            prompt + completion,
            return_tensors="pt",
            truncation=True,
            max_length=data_cfg.max_prompt_length + cfg.max_new_tokens,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        labels_mask[0, min(prompt_len, input_ids.shape[1]):] = True

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = labels_mask[:, 1:].contiguous()
        logp = torch.log_softmax(shift_logits, dim=-1).gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        selected = logp[shift_mask]
        if selected.numel() == 0:
            continue
        loss = -(selected * torch.tensor(float(adv), device=device)).mean()
        loss.backward()
        total_loss += float(loss.detach().cpu())
        n_valid += 1

    if n_valid:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "loss": total_loss / n_valid if n_valid else 0.0,
        "avg_reward": sum(rewards) / len(rewards),
        "avg_format": sum(fmt) / len(fmt),
        "avg_accuracy": sum(acc) / len(acc),
        "solved": sum(1 for x in acc if x >= 0.99),
        "num_generations": len(completions),
        "samples": completions[:2],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--init_adapter", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_train_samples", type=int, default=300)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--task", choices=["game24", "countdown"], default="game24")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.init_adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
    model.train()

    data_cfg = DataConfig(
        max_train_samples=args.max_train_samples,
        countdown_train_samples=args.max_train_samples,
        load_official_eval=False,
    )
    if args.task == "countdown":
        data = load_countdown_dataset(data_cfg)["train"]
    else:
        data = load_24game_dataset(data_cfg)["train"]
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        warmup_ratio=0.0,
    )
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = []
    start = time.time()
    step = 0
    for epoch in range(args.epochs):
        order = torch.randperm(len(data)).tolist()
        for idx in tqdm(order, desc=f"GRPO epoch {epoch + 1}"):
            nums = get_number_list(data[idx])
            target = get_target_value(data[idx])
            m = grpo_step(model, optimizer, tokenizer, nums, target, cfg, data_cfg, device)
            step += 1
            m.update({"step": step, "epoch": epoch + 1, "numbers": nums, "target": target})
            metrics.append(m)
            if step % 10 == 0:
                recent = metrics[-10:]
                print(
                    f"step={step} reward={sum(x['avg_reward'] for x in recent)/len(recent):.3f} "
                    f"acc={sum(x['avg_accuracy'] for x in recent)/len(recent):.3f} "
                    f"solved={sum(x['solved'] for x in recent)}/{sum(x['num_generations'] for x in recent)}",
                    flush=True,
                )

    final_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    summary = {
        "base_model": args.base_model,
        "init_adapter": args.init_adapter,
        "task": args.task,
        "train_samples": len(data),
        "epochs": args.epochs,
        "total_steps": step,
        "learning_rate": args.learning_rate,
        "num_generations": args.num_generations,
        "max_new_tokens": args.max_new_tokens,
        "training_time_seconds": time.time() - start,
        "final_avg_reward": sum(x["avg_reward"] for x in metrics[-50:]) / min(50, len(metrics)),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "grpo_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "grpo_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
