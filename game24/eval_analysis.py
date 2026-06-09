"""Small helpers for evaluation diagnostics."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(str(value).strip().replace("%", ""))
    except (TypeError, ValueError):
        return None
    if out > 1.0 and out <= 100.0:
        out /= 100.0
    return out


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def example_metadata(example: Dict[str, Any]) -> Dict[str, Any]:
    """Return stable metadata fields used by report diagnostics."""
    return {
        "source": example.get("source"),
        "rank": _coerce_int(example.get("rank")),
        "dataset_solved_rate": _coerce_float(example.get("solved_rate")),
    }


def solved_rate_bucket(rate: Optional[float]) -> Optional[str]:
    if rate is None:
        return None
    if rate < 0.25:
        return "0-25%"
    if rate < 0.50:
        return "25-50%"
    if rate < 0.75:
        return "50-75%"
    return "75-100%"


@dataclass
class DifficultyTracker:
    """Aggregate official game-of-24 rank/solved-rate diagnostics."""

    solved_rate_sum: float = 0.0
    solved_rate_count: int = 0
    rank_sum: float = 0.0
    rank_count: int = 0
    buckets: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add(self, metadata: Dict[str, Any], is_solved: bool) -> None:
        rank = metadata.get("rank")
        if rank is not None:
            self.rank_sum += float(rank)
            self.rank_count += 1

        solved_rate = metadata.get("dataset_solved_rate")
        bucket = solved_rate_bucket(solved_rate)
        if solved_rate is None or bucket is None:
            return

        self.solved_rate_sum += float(solved_rate)
        self.solved_rate_count += 1
        stats = self.buckets.setdefault(bucket, {"total": 0, "solved": 0})
        stats["total"] += 1
        stats["solved"] += int(is_solved)

    def summary(self) -> Optional[Dict[str, Any]]:
        if self.solved_rate_count == 0 and self.rank_count == 0:
            return None

        bucket_stats = {}
        for bucket in ["0-25%", "25-50%", "50-75%", "75-100%"]:
            stats = self.buckets.get(bucket)
            if not stats:
                continue
            total = stats["total"]
            bucket_stats[bucket] = {
                "total": total,
                "solved": stats["solved"],
                "solve_rate": stats["solved"] / total if total else 0.0,
            }

        return {
            "items_with_solved_rate": self.solved_rate_count,
            "avg_dataset_solved_rate": (
                self.solved_rate_sum / self.solved_rate_count
                if self.solved_rate_count
                else None
            ),
            "items_with_rank": self.rank_count,
            "avg_rank": self.rank_sum / self.rank_count if self.rank_count else None,
            "bucket_stats": bucket_stats,
        }
