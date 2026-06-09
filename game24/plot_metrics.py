"""Plot SFT/GRPO training curves from saved JSON metrics."""

import argparse
import json
import os
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def moving_average(values, window):
    if window <= 1 or len(values) <= window:
        return values
    out = []
    total = 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        denom = min(i + 1, window)
        out.append(total / denom)
    return out


def plot_series(metrics, keys, out_file, title, smooth):
    import matplotlib.pyplot as plt

    steps = [m.get("step", i + 1) for i, m in enumerate(metrics)]
    plt.figure(figsize=(8, 4.5))
    for key in keys:
        values = [m.get(key) for m in metrics if m.get(key) is not None]
        if not values:
            continue
        x = steps[:len(values)]
        plt.plot(x, moving_average(values, smooth), label=key)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Directory containing *_metrics.json files.")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--smooth", type=int, default=10)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plotted = []
    sft_path = run_dir / "sft_metrics.json"
    if sft_path.exists():
        metrics = load_json(sft_path)
        out_file = out_dir / "sft_loss.png"
        plot_series(metrics, ["loss"], out_file, "SFT loss", args.smooth)
        plotted.append(str(out_file))

    for name in ["grpo_metrics.json", "training_metrics.json"]:
        path = run_dir / name
        if not path.exists():
            continue
        metrics = load_json(path)
        reward_keys = [
            "avg_reward",
            "avg_accuracy",
            "avg_acc_reward",
            "avg_format",
            "avg_fmt_reward",
        ]
        out_file = out_dir / f"{path.stem}_rewards.png"
        plot_series(metrics, reward_keys, out_file, f"{path.stem} rewards", args.smooth)
        plotted.append(str(out_file))

        solved_file = out_dir / f"{path.stem}_solved.png"
        plot_series(metrics, ["solved"], solved_file, f"{path.stem} solved per group", args.smooth)
        plotted.append(str(solved_file))

    print(json.dumps({"plots": plotted}, indent=2))


if __name__ == "__main__":
    main()
