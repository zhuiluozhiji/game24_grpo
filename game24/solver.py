"""Small exact solver for generating arithmetic curriculum examples."""

from __future__ import annotations

from functools import lru_cache
from fractions import Fraction
from itertools import combinations_with_replacement
from typing import Iterable


def solve_target(numbers: Iterable[int], target: int = 24) -> str | None:
    """Return one exact expression that reaches target, or None."""
    items = tuple((Fraction(n), str(int(n))) for n in numbers)
    return _solve(items, Fraction(target))


def solve_24(numbers: Iterable[int], target: int = 24) -> str | None:
    """Backward-compatible 24-point solver wrapper."""
    return solve_target(numbers, target)


@lru_cache(maxsize=None)
def _solve(items: tuple[tuple[Fraction, str], ...], target: Fraction) -> str | None:
    if len(items) == 1:
        return items[0][1] if items[0][0] == target else None

    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            a, ae = items[i]
            b, be = items[j]
            rest = [items[k] for k in range(n) if k not in (i, j)]
            candidates = [
                (a + b, f"({ae}+{be})"),
                (a - b, f"({ae}-{be})"),
                (b - a, f"({be}-{ae})"),
                (a * b, f"({ae}*{be})"),
            ]
            if b != 0:
                candidates.append((a / b, f"({ae}/{be})"))
            if a != 0:
                candidates.append((b / a, f"({be}/{ae})"))

            for value, expr in candidates:
                next_items = tuple(sorted(rest + [(value, expr)], key=lambda x: (x[0], x[1])))
                found = _solve(next_items, target)
                if found is not None:
                    return found
    return None


def curriculum_examples(
    low: int,
    high: int,
    solvable: bool,
    limit: int | None = None,
    target: int = 24,
    num_numbers: int = 4,
):
    """Generate deterministic curriculum examples for a fixed target."""
    examples = []
    for nums in combinations_with_replacement(range(low, high + 1), num_numbers):
        expr = solve_target(nums, target=target)
        if (expr is not None) == solvable:
            examples.append({
                "numbers": list(nums),
                "target": target,
                "solution": expr,
                "solvable": solvable,
            })
            if limit and len(examples) >= limit:
                break
    return examples
