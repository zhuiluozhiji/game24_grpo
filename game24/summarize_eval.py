"""Summarize evaluation JSON files into metrics and error counts."""

import argparse
import collections
import json
from pathlib import Path


def summarize_split(result):
    if result.get("error_counts"):
        errors = result["error_counts"]
    else:
        errors = collections.Counter()
        for item in result.get("sample_details", []):
            err = item.get("error_type")
            if err:
                errors[err] += 1
        errors = dict(errors)
    return {
        "total": result.get("total"),
        "best_of": result.get("best_of"),
        "solve_rate": result.get("solve_rate"),
        "format_rate": result.get("format_rate"),
        "hallucination_rate": result.get("hallucination_rate"),
        "errors": errors,
        "difficulty": result.get("difficulty"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    for file_name in args.files:
        path = Path(file_name)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"\n# {path}")
        for split, result in data.items():
            row = summarize_split(result)
            solve = row["solve_rate"]
            fmt = row["format_rate"]
            hall = row["hallucination_rate"]
            print(
                f"- {split}: total={row['total']} best_of={row['best_of']} "
                f"solve={solve:.3f} format={fmt:.3f} "
                f"hallucination={hall if hall is not None else '-'}"
            )
            if row["errors"]:
                print(f"  errors={row['errors']}")
            difficulty = row.get("difficulty")
            if difficulty:
                avg_sr = difficulty.get("avg_dataset_solved_rate")
                avg_rank = difficulty.get("avg_rank")
                print(
                    "  difficulty="
                    f"items={difficulty.get('items_with_solved_rate')} "
                    f"avg_dataset_solved_rate={avg_sr:.3f} "
                    f"avg_rank={avg_rank:.1f}"
                    if avg_sr is not None and avg_rank is not None
                    else f"  difficulty={difficulty}"
                )
                buckets = difficulty.get("bucket_stats") or {}
                if buckets:
                    print(f"  buckets={buckets}")


if __name__ == "__main__":
    main()
