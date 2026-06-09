"""Reward functions for GRPO training on the 24-Point Game.

GRPO reward functions must have the signature:
    def reward_func(completions: List[str], **kwargs) -> List[float]

Where:
- completions: list of model-generated text completions
- kwargs: may include 'prompts', 'numbers' (ground truth data), etc.

Returns a list of float rewards, one per completion.
"""

from __future__ import annotations

import re
import ast
from typing import List, Dict, Any

from game24.utils import validate_solution, extract_answer_from_completion, extract_numbers


def format_reward(completions: List[str], **kwargs) -> List[float]:
    """Reward for correct R1-style format: <think>...</think><answer>...</answer>.

    Scoring:
        0.3 for having <think> tags
        0.3 for having <answer> tags
        0.4 for correct order (think before answer)
        Max: 1.0
    """
    rewards = []
    for completion in completions:
        score = 0.0
        has_think_open = '<think>' in completion
        has_think_close = '</think>' in completion
        has_answer_open = '<answer>' in completion
        has_answer_close = '</answer>' in completion

        if has_think_open and has_think_close:
            score += 0.3
        elif has_think_open or has_think_close:
            score += 0.1

        if has_answer_open and has_answer_close:
            score += 0.3
        elif has_answer_open or has_answer_close:
            score += 0.1

        # Correct order: think before answer
        if has_think_close and has_answer_open:
            think_end = completion.rfind('</think>')
            answer_start = completion.find('<answer>')
            if 0 <= think_end < answer_start:
                score += 0.4

        rewards.append(score)

    return rewards


def _coerce_numbers(nums):
    if isinstance(nums, str):
        return list(ast.literal_eval(nums))
    return list(nums)


def _target_at(targets, idx: int, default: int | float = 24) -> int | float:
    if targets is None:
        return default
    if isinstance(targets, (int, float)):
        return targets
    if isinstance(targets, str):
        return ast.literal_eval(targets)
    return targets[idx]


def accuracy_reward(
    completions: List[str],
    numbers: List[List[int]],
    targets: List[int | float] | int | float | None = None,
    **kwargs,
) -> List[float]:
    """Reward for correctly solving a target-number puzzle.

    Checks:
    - Expression can be extracted from completion
    - Expression evaluates to the example target
    - Expression uses exactly the given numbers, each once

    Args:
        completions: Model completions.
        numbers: List of input number lists corresponding to each completion.
        targets: Optional target values corresponding to each completion.

    Returns:
        Reward of 1.0 for a correct solution, -0.1 otherwise.
    """
    rewards = []
    if targets is None:
        targets = kwargs.get("target")
    if targets is None:
        targets = kwargs.get("targets")
    for idx, (completion, nums) in enumerate(zip(completions, numbers)):
        nums = _coerce_numbers(nums)
        target = _target_at(targets, idx)

        # Extract expression from completion
        expression = extract_answer_from_completion(completion)
        if expression is None:
            rewards.append(-0.1)  # Penalty for not providing an answer
            continue

        # Validate the solution
        is_valid, reason = validate_solution(nums, expression, target=target)

        rewards.append(1.0 if is_valid else -0.1)

    return rewards


def strict_format_reward(completions: List[str], **kwargs) -> List[float]:
    """Strict binary format reward: 1.0 if exactly matches expected format, else 0.0.

    Expected pattern: <think> content </think> <answer> content </answer>
    """
    # Pattern: think block followed by answer block (allows whitespace between)
    pattern = r'<think>.*?</think>\s*<answer>.*?</answer>'
    rewards = []
    for completion in completions:
        if re.search(pattern, completion, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def combined_reward(
    completions: List[str],
    numbers: List[List[int]],
    targets: List[int | float] | int | float | None = None,
    format_weight: float = 0.3,
    accuracy_weight: float = 0.7,
    **kwargs,
) -> List[float]:
    """Combined reward: weighted sum of format and accuracy rewards.

    This is a convenience wrapper that combines format and accuracy rewards
    with configurable weights.

    Args:
        completions: Model completions.
        numbers: Input number lists.
        targets: Optional target values.
        format_weight: Weight for format reward (default 0.3).
        accuracy_weight: Weight for accuracy reward (default 0.7).
    """
    fmt_scores = format_reward(completions)
    acc_scores = accuracy_reward(completions, numbers=numbers, targets=targets, **kwargs)

    combined = []
    for fmt, acc in zip(fmt_scores, acc_scores):
        combined.append(format_weight * fmt + accuracy_weight * acc)

    return combined


def make_reward_dict(numbers_column: str = "numbers", target_column: str = "target"):
    """Create a reward function dict that passes dataset columns to reward functions.

    TRL GRPOTrainer can accept reward functions that receive kwargs from the dataset.
    This helper ensures the 'numbers' column is passed correctly.

    Usage:
        from trl import GRPOTrainer
        trainer = GRPOTrainer(
            reward_funcs=[
                make_reward_dict("numbers")
            ],
            ...
        )
    """
    def reward_wrapper(completions: List[str], numbers: List = None, **kwargs) -> List[float]:
        if numbers is not None:
            return accuracy_reward(
                completions,
                numbers=numbers,
                targets=kwargs.get(target_column),
                **kwargs,
            )
        return [0.0] * len(completions)
    return reward_wrapper
