"""Data loading and preprocessing for 24-Point Game GRPO training."""

from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from itertools import combinations_with_replacement
import random
import json
import os

from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizer

from game24.config import DataConfig
from game24.utils import format_prompt


SYSTEM_PROMPT = (
    "You are a math puzzle solver. Given 4 numbers, find an expression "
    "using each number exactly once with +, -, *, / and parentheses to make 24. "
    "Think step by step inside <think>...</think> tags, then give the final "
    "expression inside <answer>...</answer> tags."
)


def load_24game_dataset(config: DataConfig) -> Dict[str, Dataset]:
    """Load and prepare 24-game datasets for training and evaluation.

    Returns a dict with keys:
        - 'train': solvable training set
        - 'test_solvable': held-out solvable test set
        - 'test_unsolvable': unsolvable set (for hallucination check)
        - 'test_ood': out-of-distribution test set
    """
    print("Loading nlile/24-game dataset...")
    cache_file = os.path.join(
        os.path.dirname(__file__), "..", "dataset_cache", "train.json"
    )
    if os.path.exists(cache_file):
        print(f"  Loading cached dataset: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            ds = Dataset.from_list(json.load(f))
    else:
        ds = load_dataset(config.train_dataset)

    # The dataset structure: usually has 'train' split
    if isinstance(ds, dict):
        ds = ds.get('train', list(ds.values())[0])

    solvable = ds.filter(lambda x: x['solvable'] is True)
    unsolvable = ds.filter(lambda x: x['solvable'] is False)

    print(f"  Solvable examples: {len(solvable)}")
    print(f"  Unsolvable examples: {len(unsolvable)}")

    # Shuffle and split solvable into train/test
    solvable = solvable.shuffle(seed=42)

    if config.max_train_samples:
        n_train = min(config.max_train_samples, len(solvable))
    else:
        n_train = max(int(len(solvable) * 0.8), 1)

    train_solvable = solvable.select(range(n_train))
    test_solvable = solvable.select(range(n_train, len(solvable)))

    # Limit unsolvable test set
    test_unsolvable = unsolvable.shuffle(seed=42)
    if config.num_unsolvable_test and len(test_unsolvable) > config.num_unsolvable_test:
        test_unsolvable = test_unsolvable.select(range(config.num_unsolvable_test))
    if len(test_unsolvable) == 0 and config.num_unsolvable_test:
        test_unsolvable = _make_synthetic_unsolvable(config.num_unsolvable_test)

    print(f"  Train (solvable): {len(train_solvable)}")
    print(f"  Test (solvable): {len(test_solvable)}")
    print(f"  Test (unsolvable): {len(test_unsolvable)}")

    # Try to load OOD test dataset
    test_ood = None
    try:
        print(f"Loading OOD test dataset: {config.test_dataset}...")
        ood_ds = load_dataset(config.test_dataset)
        if isinstance(ood_ds, dict):
            ood_ds = ood_ds.get('train', list(ood_ds.values())[0])
        if config.max_test_samples and len(ood_ds) > config.max_test_samples:
            ood_ds = ood_ds.shuffle(seed=42).select(range(config.max_test_samples))
        test_ood = ood_ds
        print(f"  OOD test: {len(test_ood)}")
    except Exception as e:
        print(f"  Could not load OOD dataset: {e}")
        print("  Falling back to synthetic OOD solvable puzzles.")
        test_ood = _make_synthetic_ood(config.max_test_samples or 200)

    return {
        'train': train_solvable,
        'test_solvable': test_solvable,
        'test_unsolvable': test_unsolvable,
        'test_ood': test_ood,
    }


def _make_synthetic_unsolvable(n_samples: int) -> Dataset:
    """Create deterministic unsolvable examples when the source has none."""
    examples = []
    for nums in combinations_with_replacement(range(1, 14), 4):
        nums = list(nums)
        if not _is_24_solvable(tuple(nums)):
            examples.append({
                "numbers": nums,
                "solutions": [],
                "solvable": False,
                "source": "synthetic_unsolvable",
            })
            if len(examples) >= n_samples:
                break
    return Dataset.from_list(examples)


def _make_synthetic_ood(n_samples: int) -> Dataset:
    """Create deterministic solvable puzzles with numbers outside train range."""
    examples = []
    for nums in combinations_with_replacement(range(10, 21), 4):
        nums = list(nums)
        if _is_24_solvable(tuple(nums)):
            examples.append({
                "numbers": nums,
                "solutions": [],
                "solvable": True,
                "source": "synthetic_ood",
            })
            if len(examples) >= n_samples:
                break
    return Dataset.from_list(examples)


@lru_cache(maxsize=None)
def _is_24_solvable(nums: Tuple[int, ...]) -> bool:
    """Brute-force 24-point solvability check for small synthetic datasets."""
    values = tuple(float(n) for n in nums)
    return any(abs(v - 24.0) < 1e-6 for v in _reachable_values(values))


@lru_cache(maxsize=None)
def _reachable_values(values: Tuple[float, ...]) -> Tuple[float, ...]:
    if len(values) == 1:
        return values

    out = set()
    length = len(values)
    indices = range(length)
    for mask in range(1, (1 << length) - 1):
        left_idx = tuple(i for i in indices if mask & (1 << i))
        right_idx = tuple(i for i in indices if not mask & (1 << i))
        if left_idx[0] != 0:
            continue

        left_vals = tuple(values[i] for i in left_idx)
        right_vals = tuple(values[i] for i in right_idx)
        for a in _reachable_values(tuple(sorted(left_vals))):
            for b in _reachable_values(tuple(sorted(right_vals))):
                out.add(a + b)
                out.add(a - b)
                out.add(b - a)
                out.add(a * b)
                if abs(b) > 1e-12:
                    out.add(a / b)
                if abs(a) > 1e-12:
                    out.add(b / a)
    return tuple(out)


def format_dataset_for_grpo(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
) -> Dataset:
    """Format dataset for GRPO training.

    GRPOTrainer expects a dataset with a 'prompt' column containing
    tokenized (or text) prompts. Each prompt should be formatted as
    a chat template or direct text.

    Args:
        dataset: Dataset with 'numbers' column.
        tokenizer: Tokenizer for chat template formatting.
        max_length: Maximum prompt length.

    Returns:
        Dataset with 'prompt' column for GRPOTrainer.
    """
    def format_example(example):
        numbers = example['numbers']
        if isinstance(numbers, str):
            import json
            try:
                numbers = json.loads(numbers)
            except (json.JSONDecodeError, TypeError):
                numbers = [int(n.strip()) for n in numbers.strip('[]').split(',')]

        prompt = format_prompt(numbers)

        # Format as Qwen chat
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback: plain text prompt
            formatted = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"

        return {"prompt": formatted}

    formatted = dataset.map(format_example, remove_columns=dataset.column_names)
    return formatted


def get_number_list(example: Dict) -> List[int]:
    """Extract number list from a dataset example."""
    numbers = example.get('numbers', [])
    if isinstance(numbers, str):
        import json
        try:
            numbers = json.loads(numbers)
        except (json.JSONDecodeError, TypeError):
            numbers = [int(n.strip()) for n in numbers.strip('[]').split(',')]
    return list(numbers)
