"""Reward functions v2 - prevents shortcut of outputting just "24".

Key change: accuracy reward is now binary with penalty for invalid-number shortcuts.
"""

import re
from typing import List, Dict, Any

from game24.utils import validate_solution, extract_answer_from_completion, extract_numbers, safe_eval, ExpressionError


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

        if has_think_close and has_answer_open:
            think_end = completion.rfind('</think>')
            answer_start = completion.find('<answer>')
            if 0 <= think_end < answer_start:
                score += 0.4

        rewards.append(score)

    return rewards


def accuracy_reward(completions: List[str], numbers: List[List[int]], **kwargs) -> List[float]:
    """Binary accuracy reward for 24-point solving.

    V2 CHANGES:
    - No partial credit for getting 24 with wrong numbers (prevents "24" shortcut)
    - Only rewards exactly correct solutions (right numbers, right result)
    - Penalty for invalid expressions

    Returns:
        1.0 - correct solution using exactly the given numbers
        -0.1 - wrong answer or invalid expression
    """
    rewards = []
    for completion, nums in zip(completions, numbers):
        nums = list(nums) if not isinstance(nums, str) else eval(nums)

        expression = extract_answer_from_completion(completion)
        if expression is None:
            rewards.append(-0.1)
            continue

        is_valid, reason = validate_solution(nums, expression, target=24)

        if is_valid:
            rewards.append(1.0)
        else:
            # No partial credit: must use correct numbers
            rewards.append(-0.1)

    return rewards


def strict_format_reward(completions: List[str], **kwargs) -> List[float]:
    """Strict binary format reward."""
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
    format_weight: float = 0.3,
    accuracy_weight: float = 0.7,
    **kwargs,
) -> List[float]:
    """Combined reward: weighted sum of format and accuracy rewards."""
    fmt_scores = format_reward(completions)
    acc_scores = accuracy_reward(completions, numbers=numbers)

    combined = []
    for fmt, acc in zip(fmt_scores, acc_scores):
        combined.append(format_weight * fmt + accuracy_weight * acc)

    return combined
