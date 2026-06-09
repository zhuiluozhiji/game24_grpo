"""Configuration for 24-Point Game GRPO Training."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    use_4bit: bool = True           # 4-bit quantization for CPU
    bnb_4bit_compute_dtype: str = "float32"
    bnb_4bit_quant_type: str = "nf4"
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None  # "sdpa" for faster CPU inference


@dataclass
class LoRAConfig:
    """LoRA configuration for efficient fine-tuning."""
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
    countdown_dataset: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    max_train_samples: Optional[int] = None  # None = all 1262 solvable
    max_test_samples: Optional[int] = None
    max_prompt_length: int = 256
    max_completion_length: int = 512
    num_unsolvable_test: int = 100
    target: int = 24
    official_hard_start: int = 900
    official_hard_end: int = 1000
    exclude_hard_from_train: bool = False
    load_official_eval: bool = True
    countdown_train_samples: Optional[int] = None
    countdown_test_samples: int = 200


@dataclass
class GRPOConfig:
    """GRPO training configuration (CPU-friendly)."""
    output_dir: str = "./output/game24-grpo"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 50

    # GRPO specific
    num_generations: int = 4       # candidates per prompt (keep low for CPU)
    temperature: float = 0.9
    max_new_tokens: int = 512
    beta: float = 0.04             # KL penalty coefficient

    # CPU-specific
    use_vllm: bool = False         # cannot use vLLM on CPU
    bf16: bool = False             # no bf16 on CPU
    fp16: bool = False             # no fp16 on CPU

    # Misc
    seed: int = 42
    report_to: str = "none"        # "wandb" if available
    run_name: str = "game24-grpo-cpu"


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    num_test_samples: int = 200
    temperature: float = 0.6
    max_new_tokens: int = 512
    num_beams: int = 1
