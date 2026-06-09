"""Configuration for 24-Point Game GRPO Training — 3B Model GPU."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration — Qwen2.5-3B-Instruct on GPU."""
    model_name: str = "/data/ysf/.cache/modelscope/Qwen/Qwen2___5-3B-Instruct"
    use_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = "sdpa"


@dataclass
class LoRAConfig:
    """LoRA configuration for efficient fine-tuning of 3B model."""
    r: int = 16
    lora_alpha: int = 32
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DataConfig:
    """Data configuration."""
    train_dataset: str = "nlile/24-game"
    test_dataset: str = "test-time-compute/game-of-24"
    max_train_samples: Optional[int] = None  # None = all ~1089 solvable
    max_test_samples: Optional[int] = None
    max_prompt_length: int = 256
    max_completion_length: int = 512
    num_unsolvable_test: int = 100


@dataclass
class GRPOConfig:
    """GRPO training configuration — 3B GPU optimized."""
    output_dir: str = "./output/game24-grpo-3b"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    logging_steps: int = 5
    save_steps: int = 200
    eval_steps: int = 200

    # GRPO specific — reduced for 3B VRAM
    num_generations: int = 2        # 2 generations to save VRAM
    temperature: float = 0.9
    max_new_tokens: int = 512
    beta: float = 0.04

    # GPU settings
    use_vllm: bool = False
    bf16: bool = True
    fp16: bool = False

    # Misc
    seed: int = 42
    report_to: str = "none"
    run_name: str = "game24-grpo-3b-gpu"


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    num_test_samples: int = 200
    temperature: float = 0.6
    max_new_tokens: int = 512
    num_beams: int = 1
