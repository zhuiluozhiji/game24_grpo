"""GRPO Training for 24-Point Game — Manual Implementation.

Implements Group Relative Policy Optimization (GRPO) from scratch,
compatible with PyTorch 2.5+ and CPU-only environments.

Reference: DeepSeek-R1 (arXiv:2501.12948), DeepSeekMath (arXiv:2402.03300)
"""

import os
import json
import re
import math
import time
import gc
import warnings

# CPU thread safety on Windows — MUST be set before any torch import
os.environ.setdefault("OMP_NUM_THREADS", "1")

from typing import List, Dict, Optional, Tuple
from datetime import datetime

import torch
torch.set_num_threads(1)
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from game24.config import ModelConfig, LoRAConfig as LoRACfg, DataConfig, GRPOConfig
from game24.data import load_24game_dataset, format_dataset_for_grpo, get_number_list, SYSTEM_PROMPT
from game24.rewards import format_reward, accuracy_reward
from game24.utils import extract_answer_from_completion, format_prompt

warnings.filterwarnings("ignore", category=FutureWarning)


def build_prompt(numbers: List[int]) -> str:
    """Build a prompt string for the 24-point game."""
    return format_prompt(numbers)


def get_chat_messages(numbers: List[int]) -> List[Dict]:
    """Build Qwen chat messages."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(numbers)},
    ]


def compute_log_probs(model, input_ids, attention_mask, labels_mask):
    """Compute token-level log probabilities for a sequence.

    Args:
        model: The policy model.
        input_ids: Full input sequence [1, seq_len].
        attention_mask: Attention mask [1, seq_len].
        labels_mask: Boolean mask marking completion (non-prompt) tokens [1, seq_len].

    Returns:
        log_probs: Log probs for each completion token [num_completion_tokens].
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :].contiguous()  # [1, seq_len-1, vocab]
    shift_labels = input_ids[:, 1:].contiguous()    # [1, seq_len-1]
    shift_mask = labels_mask[:, 1:].contiguous()     # [1, seq_len-1]

    # Log softmax
    log_probs_all = F.log_softmax(shift_logits, dim=-1)  # [1, seq_len-1, vocab]

    # Gather log probs of actual tokens
    token_log_probs = log_probs_all.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # [1, seq_len-1]

    # Mask to only completion tokens
    masked_log_probs = token_log_probs[shift_mask]  # [num_completion_tokens]
    return masked_log_probs


def generate_completions(
    model,
    tokenizer,
    messages: List[Dict],
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[List[str], List[torch.Tensor]]:
    """Generate multiple completions for a single prompt.

    Returns:
        completions: List of decoded completion strings.
        completion_ids: List of token ID tensors for each completion.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    prompt_len = prompt_ids.shape[1]

    completions = []
    completion_ids = []

    for _ in range(num_generations):
        with torch.no_grad():
            outputs = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Extract only the generated part
        full_ids = outputs[0]
        generated_ids = full_ids[prompt_len:]  # tokens after prompt
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

        completions.append(completion)
        completion_ids.append(full_ids)  # store full sequence for log prob computation

    return completions, completion_ids


def compute_advantages(rewards: List[float], epsilon: float = 1e-8) -> List[float]:
    """Compute advantages from rewards.

    For G >= 2: Group-relative advantage A_i = (r_i - mean(r)) / (std(r) + ε)
    For G == 1: A = 2 * r - 1  (maps reward [0,1] to advantage [-1,1])
    """
    if len(rewards) <= 1:
        # Single sample: use signed reward as advantage
        return [2.0 * r - 1.0 for r in rewards]

    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    mean_r = rewards_t.mean()
    std_r = rewards_t.std()

    if std_r < epsilon:
        return [2.0 * r - 1.0 for r in rewards]

    advantages = (rewards_t - mean_r) / (std_r + epsilon)
    return advantages.tolist()


def train_step(
    model,
    optimizer,
    tokenizer,
    messages: List[Dict],
    numbers: List[int],
    grpo_cfg: GRPOConfig,
    data_cfg: DataConfig,
    global_step: int,
) -> Dict[str, float]:
    """Single GRPO training step for one prompt.

    1. Generate G completions
    2. Compute rewards
    3. Compute group-relative advantages
    4. Compute policy loss with clipping
    5. Update model
    """
    G = grpo_cfg.num_generations

    # === Phase 1: Generate completions (no grad) ===
    model.eval()
    completions, full_ids = generate_completions(
        model, tokenizer, messages,
        num_generations=G,
        max_new_tokens=grpo_cfg.max_new_tokens,
        temperature=grpo_cfg.temperature,
    )

    # === Phase 2: Compute rewards ===
    fmt_rewards = format_reward(completions)
    acc_rewards = accuracy_reward(completions, numbers=[numbers] * G)

    combined_rewards = [
        0.3 * f + 0.7 * a for f, a in zip(fmt_rewards, acc_rewards)
    ]

    # === Phase 3: Compute group-relative advantages ===
    advantages = compute_advantages(combined_rewards)

    # === Phase 4: Compute GRPO loss and update ===
    model.train()

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    prompt_len = prompt_ids.shape[1]

    total_loss = 0.0
    total_clip_frac = 0.0
    n_valid = 0

    for i, (completion, advantage) in enumerate(zip(completions, advantages)):
        if abs(advantage) < 1e-8:
            continue  # skip near-zero advantage samples

        # Tokenize full sequence (prompt + completion)
        full_text = prompt_text + completion
        full_enc = tokenizer(full_text, return_tensors="pt", truncation=True,
                            max_length=data_cfg.max_prompt_length + grpo_cfg.max_new_tokens)

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        # Create labels mask: True for completion tokens only
        labels_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        comp_start = min(prompt_len, input_ids.shape[1])
        labels_mask[0, comp_start:] = True

        # Forward pass to get logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, seq_len, vocab]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = labels_mask[:, 1:].contiguous()

        # Log probs under current policy
        log_probs_curr = F.log_softmax(shift_logits, dim=-1)
        token_log_probs_curr = log_probs_curr.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # [1, seq_len-1]

        # Compute old log probs (detach from current graph)
        with torch.no_grad():
            token_log_probs_old = token_log_probs_curr.detach().clone()

        # Only keep completion tokens
        logp_curr = token_log_probs_curr[shift_mask]  # [n_comp_tokens]
        logp_old = token_log_probs_old[shift_mask]    # [n_comp_tokens]

        if logp_curr.numel() == 0:
            continue

        # Probability ratio
        ratio = torch.exp(logp_curr - logp_old)  # [n_comp_tokens]

        # PPO-style clipping
        eps_clip = 0.2
        advantage_t = torch.tensor(advantage, dtype=torch.float32)

        surr1 = ratio * advantage_t
        surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantage_t
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL penalty (approximate)
        kl_approx = ((logp_old - logp_curr).mean()).detach()

        # Total loss
        loss = policy_loss + grpo_cfg.beta * kl_approx

        # Backward
        loss.backward()

        total_loss += loss.item()
        total_clip_frac += ((torch.abs(ratio - 1.0) > eps_clip).float().mean()).item()
        n_valid += 1

    # Gradient step
    if n_valid > 0:
        total_loss /= n_valid
        total_clip_frac /= n_valid

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()

    # Metrics
    avg_reward = sum(combined_rewards) / len(combined_rewards)
    avg_fmt = sum(fmt_rewards) / len(fmt_rewards)
    avg_acc = sum(acc_rewards) / len(acc_rewards)
    max_reward = max(combined_rewards)
    solved = sum(1 for r in acc_rewards if r >= 0.99)

    return {
        "loss": total_loss if n_valid > 0 else 0.0,
        "avg_reward": avg_reward,
        "avg_fmt_reward": avg_fmt,
        "avg_acc_reward": avg_acc,
        "max_reward": max_reward,
        "solved": solved,
        "G": G,
        "n_valid": n_valid,
        "clip_frac": total_clip_frac,
    }


def run_training(
    model_cfg: ModelConfig,
    lora_cfg: LoRACfg,
    data_cfg: DataConfig,
    grpo_cfg: GRPOConfig,
):
    """Run the full GRPO training loop."""
    print("=" * 60)
    print("24-Point Game GRPO Training (Manual Implementation)")
    print("=" * 60)

    # === Load model ===
    print(f"\nLoading model: {model_cfg.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name, trust_remote_code=model_cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  Loading model in float32 on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=model_cfg.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    print(f"  Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    # === Apply LoRA ===
    print("\nApplying LoRA...")
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.train()

    # === Load data ===
    print("\nLoading datasets...")
    datasets = load_24game_dataset(data_cfg)
    train_ds = datasets['train']
    print(f"  Training samples: {len(train_ds)}")

    # === Setup optimizer ===
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=grpo_cfg.learning_rate)

    # Learning rate scheduler (cosine)
    total_steps = len(train_ds) * grpo_cfg.num_train_epochs
    warmup_steps = int(total_steps * grpo_cfg.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # === Training loop ===
    os.makedirs(grpo_cfg.output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Starting GRPO Training")
    print(f"  Epochs: {grpo_cfg.num_train_epochs}")
    print(f"  Training samples: {len(train_ds)}")
    print(f"  Generations per prompt: {grpo_cfg.num_generations}")
    print(f"  Learning rate: {grpo_cfg.learning_rate}")
    print(f"  KL beta: {grpo_cfg.beta}")
    print(f"  Max new tokens: {grpo_cfg.max_new_tokens}")
    print(f"  Temperature: {grpo_cfg.temperature}")
    print(f"{'=' * 60}\n")

    all_metrics = []
    global_step = 0
    start_time = time.time()

    for epoch in range(grpo_cfg.num_train_epochs):
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch + 1}/{grpo_cfg.num_train_epochs} ---")

        # Shuffle training data each epoch
        indices = torch.randperm(len(train_ds)).tolist()

        pbar = tqdm(indices, desc=f"Epoch {epoch+1}", unit="sample")
        epoch_metrics = []

        for idx in pbar:
            example = train_ds[idx]
            numbers = get_number_list(example)

            # Build chat messages
            messages = get_chat_messages(numbers)

            # Current learning rate
            current_lr = scheduler.get_last_lr()[0]

            # Train step
            metrics = train_step(
                model, optimizer, tokenizer,
                messages, numbers, grpo_cfg, data_cfg, global_step
            )

            # Update LR
            scheduler.step()
            global_step += 1

            metrics["step"] = global_step
            metrics["lr"] = current_lr
            epoch_metrics.append(metrics)
            all_metrics.append(metrics)

            # Memory cleanup
            gc.collect()

            # Update progress bar
            pbar.set_postfix(
                reward=f"{metrics['avg_reward']:.2f}",
                solved=f"{metrics['solved']}/{metrics['G']}",
                loss=f"{metrics['loss']:.4f}",
                lr=f"{current_lr:.2e}",
            )

            # Logging
            if global_step % grpo_cfg.logging_steps == 0:
                recent = all_metrics[-grpo_cfg.logging_steps:]
                avg_r = sum(m["avg_reward"] for m in recent) / len(recent)
                avg_acc = sum(m["avg_acc_reward"] for m in recent) / len(recent)
                avg_fmt = sum(m["avg_fmt_reward"] for m in recent) / len(recent)
                avg_solved = sum(m["solved"] for m in recent) / len(recent)
                tqdm.write(
                    f"  Step {global_step}: "
                    f"reward={avg_r:.3f}, "
                    f"fmt={avg_fmt:.3f}, "
                    f"acc={avg_acc:.3f}, "
                    f"solved_avg={avg_solved:.1f}/{metrics['G']}, "
                    f"lr={current_lr:.2e}"
                )

            # Save checkpoint
            if global_step % grpo_cfg.save_steps == 0:
                ckpt_dir = os.path.join(grpo_cfg.output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                tqdm.write(f"  Checkpoint saved to {ckpt_dir}")

        epoch_time = time.time() - epoch_start
        # Epoch summary
        avg_reward = sum(m["avg_reward"] for m in epoch_metrics) / len(epoch_metrics)
        avg_acc = sum(m["avg_acc_reward"] for m in epoch_metrics) / len(epoch_metrics)
        avg_solved = sum(m["solved"] for m in epoch_metrics) / len(epoch_metrics)
        print(f"  Epoch {epoch+1} done ({epoch_time:.0f}s): "
              f"avg_reward={avg_reward:.3f}, "
              f"avg_acc={avg_acc:.3f}, "
              f"solved_avg={avg_solved:.1f}/{metrics['G']}")

    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    # === Save final model ===
    output_dir = os.path.join(grpo_cfg.output_dir, "final_model")
    print(f"Saving final model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # === Save training metrics ===
    metrics_file = os.path.join(grpo_cfg.output_dir, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

    summary = {
        "model": model_cfg.model_name,
        "lora_r": lora_cfg.r,
        "train_samples": len(train_ds),
        "epochs": grpo_cfg.num_train_epochs,
        "total_steps": global_step,
        "learning_rate": grpo_cfg.learning_rate,
        "beta": grpo_cfg.beta,
        "num_generations": grpo_cfg.num_generations,
        "training_time_seconds": total_time,
        "final_avg_reward": sum(m["avg_reward"] for m in all_metrics[-50:]) / min(50, len(all_metrics)) if all_metrics else 0,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(grpo_cfg.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return model, tokenizer, datasets


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train 24-Point Game solver with GRPO")
    parser.add_argument("--output_dir", type=str, default="./output/game24-grpo")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model_cfg = ModelConfig()
    lora_cfg = LoRACfg()
    data_cfg = DataConfig(max_train_samples=args.max_train_samples)
    grpo_cfg = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_generations=args.num_generations,
        seed=args.seed,
    )

    run_training(model_cfg, lora_cfg, data_cfg, grpo_cfg)
    print("\nTraining script complete!")


if __name__ == "__main__":
    main()
