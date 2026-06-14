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


def draw_error_breakdown() -> None:
    width, height = 980, 560
    area = chart_area(width, height)
    path = "results/game24-grpo-15b-curriculum/bestof16_eval_200.json"
    data = read_json(path)
    groups = [
        ("ID", "in_distribution"),
        ("Official OOD", "out_of_distribution"),
        ("ToT hard", "tot_hard_900_1000"),
    ]
    error_types = [
        ("wrong_value", "#dc2626"),
        ("refusal_or_no_answer", "#2563eb"),
        ("invalid_expression", "#ea580c"),
        ("number_mismatch", "#7c3aed"),
        ("format_error", "#64748b"),
    ]
    body = [f'<text class="title" x="{width/2}" y="38" text-anchor="middle">Error Breakdown among Failed Cases: SFT+GRPO Best-of-16</text>']
    body.append(add_axes(area, 100, ticks=(0, 20, 40, 60, 80, 100)))
    group_w = area["w"] / len(groups)
    bar_w = 88
    for gi, (label, split) in enumerate(groups):
        center = area["x"] + group_w * (gi + 0.5)
        counts = data[split]["error_counts"]
        total_errors = sum(counts.values())
        y_cursor = area["y"] + area["h"]
        body.append(f'<text class="axis" x="{center:.1f}" y="{area["y"]+area["h"]+34}" text-anchor="middle">{esc(label)}</text>')
        body.append(f'<text class="small" x="{center:.1f}" y="{area["y"]+area["h"]+54}" text-anchor="middle">errors={total_errors}</text>')
        for err, color in error_types:
            value = counts.get(err, 0)
            if value <= 0:
                continue
            share = value / total_errors * 100 if total_errors else 0
            h = (share / 100) * area["h"]
            y_cursor -= h
            x = center - bar_w / 2
            body.append(f'<rect x="{x:.1f}" y="{y_cursor:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color}"/>')
            if h >= 18:
                body.append(f'<text class="value" x="{center:.1f}" y="{y_cursor+h/2+4:.1f}" text-anchor="middle" fill="#ffffff">{share:.0f}%</text>')
        solved = data[split]["solved"]
        body.append(f'<text class="value" x="{center:.1f}" y="{y_cursor-8:.1f}" text-anchor="middle">solved={solved}</text>')
    legend_x, legend_y = area["x"] + 10, height - 48
    for i, (err, color) in enumerate(error_types):
        x = legend_x + i * 172
        body.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{color}" rx="2"/>')
        body.append(f'<text class="small" x="{x+20}" y="{legend_y+12}">{esc(err)}</text>')
    write_svg_png("error_breakdown", svg_root(width, height, "\n".join(body)))


def draw_main_results() -> None:
    width, height = 1180, 580
    panel_w = 500
    area1 = {"x": 80, "y": 100, "w": panel_w, "h": 330}
    area2 = {"x": 640, "y": 100, "w": panel_w, "h": 330}
    groups = ["ID", "Official OOD", "ToT hard"]
    stages = [
        ("Base", COLORS["base"], "results/game24-15b-base"),
        ("SFT", COLORS["sft"], "results/game24-sft-15b-curriculum"),
        ("SFT+GRPO", COLORS["grpo"], "results/game24-grpo-15b-curriculum"),
    ]
    body = [f'<text class="title" x="{width/2}" y="38" text-anchor="middle">Main Results under Matched Decoding Settings</text>']

    def panel(area: dict, title: str, filename: str) -> None:
        body.append(f'<text class="label" x="{area["x"]+area["w"]/2}" y="{area["y"]-28}" text-anchor="middle" font-weight="700">{esc(title)}</text>')
        body.append(add_axes(area, 50, ticks=(0, 10, 20, 30, 40, 50)))
        group_w = area["w"] / len(groups)
        bar_w = min(34, group_w / (len(stages) + 1.0))
        for gi, group in enumerate(groups):
            center = area["x"] + group_w * (gi + 0.5)
            split = ["in_distribution", "out_of_distribution", "tot_hard_900_1000"][gi]
            body.append(f'<text class="axis" x="{center:.1f}" y="{area["y"]+area["h"]+34}" text-anchor="middle">{esc(group)}</text>')
            for si, (stage, color, base) in enumerate(stages):
                value = game24_metric(f"{base}/{filename}", split)
                x = center - (len(stages) * bar_w) / 2 + si * bar_w + 3
                y = y_map(value, 50, area)
                h = area["y"] + area["h"] - y
                body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-6:.1f}" height="{h:.1f}" rx="3" fill="{color}"/>')
                if value >= 3:
                    body.append(f'<text class="value" x="{x+(bar_w-6)/2:.1f}" y="{y-6:.1f}" text-anchor="middle">{value:.1f}</text>')

    panel(area1, "Greedy decoding", "quick_eval_200.json")
    panel(area2, "Verifier best-of-8", "bestof8_eval_200.json")
    legend_y = height - 48
    for i, (stage, color, _base) in enumerate(stages):
        x = 360 + i * 150
        body.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{color}" rx="2"/>')
        body.append(f'<text class="label" x="{x+22}" y="{legend_y+12}">{esc(stage)}</text>')
    write_svg_png("main_results", svg_root(width, height, "\n".join(body)))


def draw_difficulty_analysis() -> None:
    width, height = 1080, 560
    panel_w = 440
    area1 = {"x": 90, "y": 100, "w": panel_w, "h": 330}
    area2 = {"x": 610, "y": 100, "w": panel_w, "h": 330}
    data = read_json("results/game24-grpo-15b-curriculum/bestof16_eval_200.json")
    splits = [
        ("Official OOD", "out_of_distribution", COLORS["sft"]),
        ("ToT hard", "tot_hard_900_1000", COLORS["grpo"]),
    ]
    body = [f'<text class="title" x="{width/2}" y="38" text-anchor="middle">Official Difficulty vs Model Performance</text>']

    def panel(area: dict, title: str, values: list[float], suffix: str, max_value: float = 100) -> None:
        body.append(f'<text class="label" x="{area["x"]+area["w"]/2}" y="{area["y"]-28}" text-anchor="middle" font-weight="700">{esc(title)}</text>')
        body.append(add_axes(area, max_value, ticks=(0, 20, 40, 60, 80, 100)))
        group_w = area["w"] / len(splits)
        bar_w = 76
        for i, ((label, _split, color), value) in enumerate(zip(splits, values)):
            center = area["x"] + group_w * (i + 0.5)
            y = y_map(value, max_value, area)
            h = area["y"] + area["h"] - y
            body.append(f'<rect x="{center-bar_w/2:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="4" fill="{color}"/>')
            body.append(f'<text class="value" x="{center:.1f}" y="{y-8:.1f}" text-anchor="middle">{value:.1f}{suffix}</text>')
            body.append(f'<text class="axis" x="{center:.1f}" y="{area["y"]+area["h"]+34}" text-anchor="middle">{esc(label)}</text>')

    official_values = [
        pct(data[split]["difficulty"]["avg_dataset_solved_rate"])
        for _label, split, _color in splits
    ]
    model_values = [
        pct(data[split]["solve_rate"])
        for _label, split, _color in splits
    ]
    panel(area1, "Official avg solved rate", official_values, "%")
    panel(area2, "Model solve rate (best-of-16)", model_values, "%")

    rank_y = 488
    body.append(f'<text class="small" x="{width/2}" y="{rank_y}" text-anchor="middle">Avg rank: Official OOD = 100.5, ToT hard = 950.5. Higher rank indicates harder official benchmark items.</text>')
    write_svg_png("difficulty_analysis", svg_root(width, height, "\n".join(body)))


def draw_case_study_bestof() -> None:
    width, height = 1120, 560
    body = [f'<text class="title" x="{width/2}" y="38" text-anchor="middle">Case Study: Candidate Budget Reveals a Correct Expression</text>']
    body.append('''<defs><marker id="arrow2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#374151"/></marker></defs>''')
    body.append(f'<rect x="54" y="70" width="1012" height="430" rx="18" fill="#f8fafc" stroke="#93c5fd" stroke-width="2" stroke-dasharray="8 8"/>')
    body.append(f'<path d="M54 70 H490 L445 118 H54 Z" fill="#bfdbfe"/>')
    body.append(f'<text x="76" y="103" font-size="20" font-weight="700">Verifier-based best-of-N selection</text>')

    body.append(f'<rect x="90" y="150" width="210" height="230" rx="12" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>')
    body.append(f'<text class="label" x="195" y="188" text-anchor="middle" font-weight="700">Problem</text>')
    body.append(f'<text class="small" x="195" y="226" text-anchor="middle">numbers = [2, 3, 6, 10]</text>')
    body.append(f'<text class="small" x="195" y="254" text-anchor="middle">target = 24</text>')
    body.append(f'<text class="small" x="195" y="304" text-anchor="middle">Verifier checks:</text>')
    body.append(f'<text class="small" x="195" y="330" text-anchor="middle">AST + numbers + target</text>')

    rows = [
        ("Greedy", "(10-((2-3)*6))", "wrong_value", "#ef4444", "×"),
        ("Best-of-8", "((6-(2-3))*10)", "wrong_value", "#ef4444", "×"),
        ("Best-of-16", "(10-3*2)*6", "correct", "#16a34a", "✓"),
    ]
    x0, y0 = 380, 145
    body.append(f'<rect x="{x0}" y="{y0}" width="450" height="250" rx="12" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>')
    body.append(f'<text class="label" x="{x0+225}" y="{y0+36}" text-anchor="middle" font-weight="700">Candidate outputs from the same model</text>')
    for i, (mode, expr, verdict, color, mark) in enumerate(rows):
        y = y0 + 68 + i * 58
        fill = "#ecfdf5" if verdict == "correct" else "#fff7ed"
        stroke = "#16a34a" if verdict == "correct" else "#fed7aa"
        body.append(f'<rect x="{x0+24}" y="{y}" width="402" height="42" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
        body.append(f'<text class="small" x="{x0+44}" y="{y+26}" font-weight="700">{esc(mode)}</text>')
        body.append(f'<text class="small" x="{x0+142}" y="{y+26}" font-family="Menlo, monospace">{esc(expr)}</text>')
        body.append(f'<text x="{x0+344}" y="{y+27}" font-size="18" fill="{color}" font-weight="700">{mark}</text>')
        body.append(f'<text class="small" x="{x0+366}" y="{y+26}" fill="{color}">{esc(verdict)}</text>')

    body.append(f'<line x1="312" y1="265" x2="360" y2="265" stroke="#374151" stroke-width="2.5" marker-end="url(#arrow2)"/>')
    body.append(f'<line x1="842" y1="265" x2="890" y2="265" stroke="#374151" stroke-width="2.5" marker-end="url(#arrow2)"/>')

    body.append(f'<rect x="905" y="170" width="130" height="170" rx="12" fill="#ecfdf5" stroke="#86efac" stroke-width="1.5"/>')
    body.append(f'<text class="label" x="970" y="214" text-anchor="middle" font-weight="700">Selected</text>')
    body.append(f'<text x="970" y="260" text-anchor="middle" font-size="38" fill="#16a34a" font-weight="700">✓</text>')
    body.append(f'<text class="small" x="970" y="296" text-anchor="middle">best-of-16</text>')
    body.append(f'<text class="small" x="970" y="320" text-anchor="middle">passes verifier</text>')

    body.append(f'<rect x="245" y="420" width="630" height="48" rx="10" fill="#eff6ff" stroke="#bfdbfe"/>')
    body.append(f'<text class="label" x="560" y="450" text-anchor="middle">Increasing candidate budget exposes a valid expression; the symbolic verifier extracts it from the sampled pool.</text>')
    write_svg_png("case_study_bestof", svg_root(width, height, "\n".join(body)))


def game24_metric(path: str, split: str, key: str = "solve_rate") -> float:
    return pct(read_json(path)[split][key])


def main() -> None:
    draw_main_results()

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
    draw_error_breakdown()
    draw_difficulty_analysis()
    draw_case_study_bestof()


if __name__ == "__main__":
    main()
