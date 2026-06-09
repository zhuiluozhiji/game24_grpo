"""Data loading and preprocessing for Game24 and Countdown experiments."""

from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from itertools import combinations_with_replacement
import ast
import json
import os
import re

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

from game24.config import DataConfig
from game24.utils import format_prompt, format_system_prompt


SYSTEM_PROMPT = format_system_prompt()
GAME24_CSV_URL = (
    "https://huggingface.co/datasets/test-time-compute/game-of-24/"
    "resolve/main/game24.csv"
)


def _dataset_split(ds) -> Dataset:
    """Return the first split when load_dataset returns a DatasetDict."""
    if isinstance(ds, dict):
        return ds.get("train", list(ds.values())[0])
    return ds


def _coerce_numbers(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, tuple):
        return [int(x) for x in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [int(x) for x in parsed]
        except (SyntaxError, ValueError):
            pass
        return [int(x) for x in re.findall(r"-?\d+", text)]
    return [int(value)]


def _coerce_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "solvable"}


def _coerce_target(value, default: int = 24) -> int:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    try:
        return int(float(text))
    except ValueError:
        nums = re.findall(r"-?\d+", text)
        return int(nums[0]) if nums else default


def _parse_percent(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        parsed = float(text)
        return parsed / 100.0 if parsed > 1.0 else parsed
    except ValueError:
        return None


def _coerce_solutions(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, tuple):
        return [str(x) for x in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except (SyntaxError, ValueError):
            pass
        return [text]
    return [str(value)]


def _normalize_example(
    example: Dict,
    source: str,
    default_target: int = 24,
    default_solvable: bool = True,
) -> Dict:
    numbers = (
        example.get("numbers")
        or example.get("nums")
        or example.get("Puzzles")
        or example.get("Puzzle")
        or example.get("puzzle")
    )
    target = example.get("target", example.get("Target"))
    out = {
        "numbers": _coerce_numbers(numbers),
        "target": _coerce_target(target, default_target),
        "solutions": _coerce_solutions(example.get("solutions")),
        "solution": example.get("solution"),
        "solvable": _coerce_bool(example.get("solvable"), default_solvable),
        "source": example.get("source", source),
    }
    if "Rank" in example:
        try:
            out["rank"] = int(float(example["Rank"]))
        except (TypeError, ValueError):
            pass
    if "Solved rate" in example:
        out["solved_rate"] = _parse_percent(example["Solved rate"])
    return out


def _normalize_dataset(
    dataset: Dataset,
    source: str,
    default_target: int = 24,
    default_solvable: bool = True,
) -> Dataset:
    return Dataset.from_list([
        _normalize_example(dict(row), source, default_target, default_solvable)
        for row in dataset
    ])


def _number_key(example: Dict) -> Tuple[int, ...]:
    return tuple(sorted(get_number_list(example)))


def _select_range(dataset: Dataset, start: int, end: int) -> Dataset:
    start = max(0, start)
    end = min(len(dataset), end)
    if end <= start:
        return Dataset.from_list([])
    return dataset.select(range(start, end))


def _exclude_indices(dataset: Dataset, start: int, end: int) -> Dataset:
    blocked = set(range(max(0, start), min(len(dataset), end)))
    return Dataset.from_list([row for i, row in enumerate(dataset) if i not in blocked])


def _load_official_game24_dataset(config: DataConfig) -> Optional[Dataset]:
    try:
        print(f"Loading official OOD dataset: {config.test_dataset}...")
        ood_ds = _dataset_split(load_dataset(config.test_dataset))
    except Exception as first_error:
        try:
            print(f"  Hub loader failed: {first_error}")
            print("  Falling back to raw CSV loader for official game24.csv...")
            ood_ds = _dataset_split(load_dataset("csv", data_files=GAME24_CSV_URL))
        except Exception as second_error:
            print(f"  Could not load official OOD dataset: {second_error}")
            return None
    official = _normalize_dataset(ood_ds, "test-time-compute/game-of-24", default_target=24)
    print(f"  Official game-of-24 examples: {len(official)}")
    return official


def load_24game_dataset(config: DataConfig) -> Dict[str, Dataset]:
    """Load and prepare 24-game datasets for training and evaluation.

    Returns a dict with keys:
        - 'train': solvable training set
        - 'test_solvable': held-out solvable test set
        - 'test_unsolvable': unsolvable set (for hallucination check)
        - 'test_ood': out-of-distribution test set
        - 'test_hard': official Tree-of-Thoughts hard split, indices 900:1000
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

    ds = _normalize_dataset(
        _dataset_split(ds),
        source="nlile/24-game",
        default_target=config.target,
    )

    official = _load_official_game24_dataset(config) if config.load_official_eval else None
    test_hard = None
    test_ood = None
    if official is not None:
        test_hard = _select_range(
            official,
            config.official_hard_start,
            config.official_hard_end,
        )
        test_ood = _exclude_indices(
            official,
            config.official_hard_start,
            config.official_hard_end,
        )
        if config.max_test_samples and len(test_ood) > config.max_test_samples:
            test_ood = test_ood.select(range(config.max_test_samples))
        print(f"  Official OOD excluding hard split: {len(test_ood)}")
        print(f"  Official hard split [{config.official_hard_start}:{config.official_hard_end}]: {len(test_hard)}")

    solvable = ds.filter(lambda x: _coerce_bool(x.get('solvable'), True) is True)
    unsolvable = ds.filter(lambda x: _coerce_bool(x.get('solvable'), True) is False)

    print(f"  Solvable examples: {len(solvable)}")
    print(f"  Unsolvable examples: {len(unsolvable)}")

    if config.exclude_hard_from_train and test_hard is not None:
        hard_keys = {_number_key(row) for row in test_hard}
        before = len(solvable)
        solvable = Dataset.from_list([row for row in solvable if _number_key(row) not in hard_keys])
        print(f"  Excluded official hard split from training candidates: {before - len(solvable)}")

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

    # Fall back only when the official OOD dataset is unavailable.
    if test_ood is None:
        print("  Falling back to synthetic OOD solvable puzzles.")
        test_ood = _make_synthetic_ood(config.max_test_samples or 200)
    if test_hard is None:
        test_hard = Dataset.from_list([])

    return {
        'train': train_solvable,
        'test_solvable': test_solvable,
        'test_unsolvable': test_unsolvable,
        'test_ood': test_ood,
        'test_hard': test_hard,
    }


def _make_synthetic_unsolvable(n_samples: int) -> Dataset:
    """Create deterministic unsolvable examples when the source has none."""
    examples = []
    for nums in combinations_with_replacement(range(1, 14), 4):
        nums = list(nums)
        if not _is_24_solvable(tuple(nums)):
            examples.append({
                "numbers": nums,
                "target": 24,
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
                "target": 24,
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
        numbers = get_number_list(example)
        target = get_target_value(example)

        prompt = format_prompt(numbers, target=target)

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

        return {"prompt": formatted, "numbers": numbers, "target": target}

    formatted = dataset.map(format_example, remove_columns=dataset.column_names)
    return formatted


def get_number_list(example: Dict) -> List[int]:
    """Extract number list from a dataset example."""
    numbers = (
        example.get('numbers')
        or example.get('nums')
        or example.get('Puzzles')
        or example.get('Puzzle')
        or []
    )
    return _coerce_numbers(numbers)


def get_target_value(example: Dict, default: int = 24) -> int:
    """Extract target value from a dataset example."""
    return _coerce_target(example.get("target", example.get("Target")), default)


def load_countdown_dataset(config: DataConfig) -> Dict[str, Dataset]:
    """Load Jiayi-Pan/Countdown-Tasks-3to4 and create train/test splits.

    The normalized schema matches Game24:
        numbers: input nums
        target: target value
        solvable: True
        source: countdown
    """
    print(f"Loading Countdown dataset: {config.countdown_dataset}...")
    ds = _dataset_split(load_dataset(config.countdown_dataset))
    ds = _normalize_dataset(
        ds,
        source="Jiayi-Pan/Countdown-Tasks-3to4",
        default_target=config.target,
        default_solvable=True,
    )
    ds = ds.shuffle(seed=42)
    if config.countdown_train_samples:
        n_train = min(config.countdown_train_samples, len(ds))
    elif config.max_train_samples:
        n_train = min(config.max_train_samples, len(ds))
    else:
        n_train = min(5000, max(len(ds) - config.countdown_test_samples, 1))

    train = ds.select(range(n_train))
    test_start = n_train
    test_end = min(len(ds), test_start + config.countdown_test_samples)
    test = ds.select(range(test_start, test_end))
    print(f"  Countdown train: {len(train)}")
    print(f"  Countdown test: {len(test)}")
    return {"train": train, "test_countdown": test}
