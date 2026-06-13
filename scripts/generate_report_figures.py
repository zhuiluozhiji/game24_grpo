#!/usr/bin/env python3
"""Generate PNG figures for the final report without external Python deps."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "report" / "figures"


COLORS = {
    "base": "#64748b",
    "sft": "#2563eb",
    "grpo": "#dc2626",
    "green": "#16a34a",
    "orange": "#ea580c",
    "purple": "#7c3aed",
    "grid": "#e5e7eb",
    "text": "#111827",
    "muted": "#6b7280",
    "bg": "#ffffff",
}


def read_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def pct(value: float) -> float:
    return value * 100.0


def esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_root(width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="{COLORS['bg']}"/>
<style>
text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; fill: {COLORS['text']}; }}
.title {{ font-size: 24px; font-weight: 700; }}
.axis {{ font-size: 13px; fill: {COLORS['muted']}; }}
.label {{ font-size: 13px; }}
.small {{ font-size: 12px; fill: {COLORS['muted']}; }}
.value {{ font-size: 12px; font-weight: 600; }}
</style>
{body}
</svg>
"""


def write_svg_png(name: str, svg: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = FIG_DIR / f"{name}.svg"
    png_path = FIG_DIR / f"{name}.png"
    svg_path.write_text(svg, encoding="utf-8")
    subprocess.run(["rsvg-convert", "-o", str(png_path), str(svg_path)], check=True)


def chart_area(width: int = 980, height: int = 560):
    return {"x": 84, "y": 80, "w": width - 130, "h": height - 160}


def y_map(value: float, max_value: float, area: dict) -> float:
    return area["y"] + area["h"] - (value / max_value) * area["h"]


def add_axes(area: dict, max_value: float, ticks=(0, 20, 40, 60, 80, 100)) -> str:
    parts = []
    for t in ticks:
        if t > max_value:
            continue
        y = y_map(t, max_value, area)
        parts.append(f'<line x1="{area["x"]}" y1="{y:.1f}" x2="{area["x"]+area["w"]}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(f'<text class="axis" x="{area["x"]-14}" y="{y+4:.1f}" text-anchor="end">{t}%</text>')
    parts.append(f'<line x1="{area["x"]}" y1="{area["y"]}" x2="{area["x"]}" y2="{area["y"]+area["h"]}" stroke="#9ca3af"/>')
    parts.append(f'<line x1="{area["x"]}" y1="{area["y"]+area["h"]}" x2="{area["x"]+area["w"]}" y2="{area["y"]+area["h"]}" stroke="#9ca3af"/>')
    return "\n".join(parts)


def draw_grouped_bars(title: str, groups: list[str], series: list[tuple[str, str, list[float]]], name: str, max_value: float = 100) -> None:
    width, height = 980, 560
    area = chart_area(width, height)
    body = [f'<text class="title" x="{width/2}" y="38" text-anchor="middle">{esc(title)}</text>']
    body.append(add_axes(area, max_value))
    group_w = area["w"] / len(groups)
    bar_w = min(46, group_w / (len(series) + 1.2))
    for gi, group in enumerate(groups):
        center = area["x"] + group_w * (gi + 0.5)
        body.append(f'<text class="axis" x="{center:.1f}" y="{area["y"]+area["h"]+34}" text-anchor="middle">{esc(group)}</text>')
        for si, (label, color, values) in enumerate(series):
            x = center - (len(series) * bar_w) / 2 + si * bar_w + 4
            v = values[gi]
            y = y_map(v, max_value, area)
            h = area["y"] + area["h"] - y
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-8:.1f}" height="{h:.1f}" rx="3" fill="{color}"/>')
            body.append(f'<text class="value" x="{x+(bar_w-8)/2:.1f}" y="{y-6:.1f}" text-anchor="middle">{v:.1f}</text>')
    legend_x = area["x"] + 8
    for i, (label, color, _) in enumerate(series):
        x = legend_x + i * 168
        body.append(f'<rect x="{x}" y="{height-42}" width="14" height="14" fill="{color}" rx="2"/>')
        body.append(f'<text class="label" x="{x+22}" y="{height-30}">{esc(label)}</text>')
    write_svg_png(name, svg_root(width, height, "\n".join(body)))


def draw_line_chart(title: str, labels: list[str], lines: list[tuple[str, str, list[float]]], name: str, max_value: float = 70) -> None:
    width, height = 980, 560
    area = chart_area(width, height)
    body = [f'<text class="title" x="{width/2}" y="38" text-anchor="middle">{esc(title)}</text>']
    body.append(add_axes(area, max_value, ticks=(0, 10, 20, 30, 40, 50, 60, 70)))
    step = area["w"] / (len(labels) - 1)
    xs = [area["x"] + i * step for i in range(len(labels))]
    for x, label in zip(xs, labels):
        body.append(f'<text class="axis" x="{x:.1f}" y="{area["y"]+area["h"]+34}" text-anchor="middle">{esc(label)}</text>')
    for label, color, values in lines:
        points = [(xs[i], y_map(values[i], max_value, area)) for i in range(len(values))]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        body.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in points:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        for x, y, v in [(points[i][0], points[i][1], values[i]) for i in range(len(values))]:
            body.append(f'<text class="value" x="{x:.1f}" y="{y-10:.1f}" text-anchor="middle">{v:.1f}</text>')
    for i, (label, color, _) in enumerate(lines):
        x = area["x"] + 8 + i * 154
        body.append(f'<line x1="{x}" y1="{height-42}" x2="{x+20}" y2="{height-42}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<circle cx="{x+10}" cy="{height-42}" r="4" fill="{color}"/>')
        body.append(f'<text class="label" x="{x+28}" y="{height-37}">{esc(label)}</text>')
    write_svg_png(name, svg_root(width, height, "\n".join(body)))


def draw_framework() -> None:
    width, height = 1120, 560
    boxes = [
        (70, 115, 190, 90, "Datasets", "24-game / official / Countdown"),
        (330, 115, 190, 90, "Prompt", "target-aware R1 format"),
        (590, 115, 190, 90, "SFT Warm-up", "format + expressions + refusal"),
        (850, 115, 190, 90, "GRPO RLVR", "group rewards from verifier"),
        (590, 335, 190, 90, "Best-of-N", "sample candidate pool"),
        (850, 335, 190, 90, "Verifier", "AST check + target value"),
        (330, 335, 190, 90, "Analysis", "errors / hard split / hallucination"),
        (70, 335, 190, 90, "Report", "solve rate + limits"),
    ]
    body = [f'<text class="title" x="{width/2}" y="46" text-anchor="middle">Overall RLVR Pipeline for Game-of-24</text>']
    def arrow(x1, y1, x2, y2):
        return f'''<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#374151" stroke-width="2" marker-end="url(#arrow)"/>'''
    body.append('''<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#374151"/></marker></defs>''')
    for i in range(3):
        x1 = boxes[i][0] + boxes[i][2]
        y = boxes[i][1] + boxes[i][3] / 2
        x2 = boxes[i + 1][0]
        body.append(arrow(x1 + 12, y, x2 - 12, y))
    body.append(arrow(945, 205, 945, 335))
    body.append(arrow(850, 380, 780, 380))
    body.append(arrow(590, 380, 520, 380))
    body.append(arrow(330, 380, 260, 380))
    for x, y, w, h, title, sub in boxes:
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>')
        body.append(f'<text class="label" x="{x+w/2}" y="{y+36}" text-anchor="middle" font-weight="700">{esc(title)}</text>')
        body.append(f'<text class="small" x="{x+w/2}" y="{y+62}" text-anchor="middle">{esc(sub)}</text>')
    write_svg_png("framework", svg_root(width, height, "\n".join(body)))


def game24_metric(path: str, split: str, key: str = "solve_rate") -> float:
    return pct(read_json(path)[split][key])


def main() -> None:
    draw_framework()

    groups = ["ID", "Official OOD", "ToT hard"]
    draw_grouped_bars(
        "Main Results: Greedy vs Best-of-16",
        groups,
        [
            ("Base greedy", COLORS["base"], [
                game24_metric("results/game24-15b-base/quick_eval_200.json", "in_distribution"),
                game24_metric("results/game24-15b-base/quick_eval_200.json", "out_of_distribution"),
                game24_metric("results/game24-15b-base/quick_eval_200.json", "tot_hard_900_1000"),
            ]),
            ("SFT best-of-16", COLORS["sft"], [
                game24_metric("results/game24-sft-15b-curriculum/bestof16_eval_200.json", "in_distribution"),
                game24_metric("results/game24-sft-15b-curriculum/bestof16_eval_200.json", "out_of_distribution"),
                game24_metric("results/game24-sft-15b-curriculum/bestof16_eval_200.json", "tot_hard_900_1000"),
            ]),
            ("SFT+GRPO best-of-16", COLORS["grpo"], [
                game24_metric("results/game24-grpo-15b-curriculum/bestof16_eval_200.json", "in_distribution"),
                game24_metric("results/game24-grpo-15b-curriculum/bestof16_eval_200.json", "out_of_distribution"),
                game24_metric("results/game24-grpo-15b-curriculum/bestof16_eval_200.json", "tot_hard_900_1000"),
            ]),
        ],
        "main_results",
        70,
    )

    labels = ["greedy", "bo1", "bo4", "bo8", "bo16"]
    paths = [
        "quick_eval_200.json",
        "bestof1_eval_200.json",
        "bestof4_eval_200.json",
        "bestof8_eval_200.json",
        "bestof16_eval_200.json",
    ]
    draw_line_chart(
        "Verifier-based Test-time Compute: SFT+GRPO",
        labels,
        [
            ("ID", COLORS["green"], [game24_metric(f"results/game24-grpo-15b-curriculum/{p}", "in_distribution") for p in paths]),
            ("Official OOD", COLORS["sft"], [game24_metric(f"results/game24-grpo-15b-curriculum/{p}", "out_of_distribution") for p in paths]),
            ("ToT hard", COLORS["grpo"], [game24_metric(f"results/game24-grpo-15b-curriculum/{p}", "tot_hard_900_1000") for p in paths]),
        ],
        "best_of_sweep",
        70,
    )

    draw_line_chart(
        "Unsolvable Hallucination vs Best-of-N",
        ["bo1", "bo4", "bo8", "bo16"],
        [
            ("Base", COLORS["base"], [game24_metric(f"results/game24-15b-base/bestof{n}_eval_200.json", "unsolvable", "hallucination_rate") for n in [1, 4, 8, 16]]),
            ("SFT", COLORS["sft"], [game24_metric(f"results/game24-sft-15b-curriculum/bestof{n}_eval_200.json", "unsolvable", "hallucination_rate") for n in [1, 4, 8, 16]]),
            ("SFT+GRPO", COLORS["grpo"], [game24_metric(f"results/game24-grpo-15b-curriculum/bestof{n}_eval_200.json", "unsolvable", "hallucination_rate") for n in [1, 4, 8, 16]]),
        ],
        "hallucination",
        100,
    )

    draw_grouped_bars(
        "GRPO Hyperparameter Stability Ablation",
        ["8e-7/s300", "3e-7/s300", "3e-7/s600"],
        [
            ("OOD bo8", COLORS["sft"], [
                game24_metric("results/game24-grpo-15b-curriculum/bestof8_eval_200.json", "out_of_distribution"),
                game24_metric("results/game24-grpo-15b-lr3e-7-s300/bestof8_eval_200.json", "out_of_distribution"),
                game24_metric("results/game24-grpo-15b-lr3e-7-s600/bestof8_eval_200.json", "out_of_distribution"),
            ]),
            ("Hard bo8", COLORS["grpo"], [
                game24_metric("results/game24-grpo-15b-curriculum/bestof8_eval_200.json", "tot_hard_900_1000"),
                game24_metric("results/game24-grpo-15b-lr3e-7-s300/bestof8_eval_200.json", "tot_hard_900_1000"),
                game24_metric("results/game24-grpo-15b-lr3e-7-s600/bestof8_eval_200.json", "tot_hard_900_1000"),
            ]),
            ("Greedy halluc.", COLORS["orange"], [
                game24_metric("results/game24-grpo-15b-curriculum/quick_eval_200.json", "unsolvable", "hallucination_rate"),
                game24_metric("results/game24-grpo-15b-lr3e-7-s300/quick_eval_200.json", "unsolvable", "hallucination_rate"),
                game24_metric("results/game24-grpo-15b-lr3e-7-s600/quick_eval_200.json", "unsolvable", "hallucination_rate"),
            ]),
        ],
        "grpo_ablation",
        50,
    )

    def countdown(path: str) -> float:
        return pct(read_json(path)["countdown"]["solve_rate"])

    draw_grouped_bars(
        "Countdown Arbitrary Target Generalization",
        ["SFT", "SFT+GRPO"],
        [
            ("Greedy", COLORS["base"], [
                countdown("results/countdown-sft-15b/quick_eval_200.json"),
                countdown("results/countdown-grpo-15b/quick_eval_200.json"),
            ]),
            ("Best-of-8", COLORS["green"], [
                countdown("results/countdown-sft-15b/bestof8_eval_200.json"),
                countdown("results/countdown-grpo-15b/bestof8_eval_200.json"),
            ]),
        ],
        "countdown",
        50,
    )


if __name__ == "__main__":
    main()
