#!/usr/bin/env python3
"""Export publication figures from validated Run 2 analysis, without API calls.

python scripts/plot_run2.py --analysis-dir analysis/run2 --out-dir figures/run2
PDF/SVG are vector exports; PNG is 300 dpi. Empty runs have no data marks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from compressors import ALL_CONDITIONS

LABELS = {"none": "None\n(baseline)", "sliding_window": "Sliding\nwindow",
          "unconditional": "Unconditional", "destination": "Destination",
          "instruction_strip": "Instruction\nstrip", "allowlist": "Allowlist"}
COLORS = ("#0072B2", "#D55E00", "#334155")


def setup_matplotlib():
    cache = Path(tempfile.gettempdir()) / "run2-matplotlib-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#64748b", "axes.linewidth": .65,
        "xtick.color": "#334155", "ytick.color": "#334155",
        "grid.color": "#cbd5e1", "grid.linewidth": .5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 300, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def task_token_ratios(analysis):
    """Per-task means preserve task identity, without character estimates."""
    grouped = {}
    for row in analysis.get("per_task", []):
        counts = row.get("token_counts", {})
        if isinstance(counts, str):
            counts = json.loads(counts)
        before, after = counts.get("handoff", {}), counts.get("compressed", {})
        denominator, numerator = before.get("tokens"), after.get("tokens")
        if not isinstance(denominator, int) or not isinstance(numerator, int) or denominator <= 0:
            continue
        if before.get("tokenizer") != after.get("tokenizer") or before.get("tokenizer") in {None, "unavailable"}:
            continue
        key = (row["condition"], row["task_id"])
        grouped.setdefault(key, []).append(numerator / denominator)
    return [{"condition": condition, "task_id": tid, "observed_repeats": len(values),
             "mean_retained_token_ratio": statistics.mean(values)}
            for (condition, tid), values in sorted(grouped.items())]


def _base_axes(ax, title, ylabel):
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.set_xticks(range(len(ALL_CONDITIONS)), [LABELS[c] for c in ALL_CONDITIONS])
    ax.set_xlim(-.6, len(ALL_CONDITIONS) - .4)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.tick_params(axis="x", length=0, pad=9, labelsize=7)
    ax.tick_params(axis="y", length=3, width=.6)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


def _empty_axes(ax, text):
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.yaxis.grid(False)
    ax.spines["left"].set_visible(False)
    ax.text(.5, .53, text, ha="center", va="center", transform=ax.transAxes,
            color="#475569", fontsize=10, linespacing=1.5)


def _jitter(count, seed):
    rng = random.Random(seed)
    return [rng.uniform(-.15, .15) for _ in range(count)]


def _point_interval(ax, x, center, low, high, color):
    if low is not None and high is not None:
        ax.errorbar(x, center, yerr=[[max(0, center-low)], [max(0, high-center)]],
                    fmt="D", markersize=5, color=color, ecolor=color,
                    capsize=3, elinewidth=1.4, capthick=1.1, zorder=5)
    else:
        ax.plot([x], [center], marker="D", markersize=5, color=color, zorder=5)


def draw_success(ax, analysis):
    _base_axes(ax, "A  Required-write success", "Success (%)")
    rows = {r["condition"]: r for r in analysis.get("summary", [])}
    if not analysis.get("n_records"):
        _empty_axes(ax, "No measured success rates\nAll six conditions remain unverified")
        return
    ax.set_ylim(-5, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    for i, condition in enumerate(ALL_CONDITIONS):
        row = rows.get(condition, {})
        if row.get("success_pct") is None:
            ax.text(i, 48, "Not run", ha="center", color="#64748b", rotation=90, fontsize=8)
            continue
        clusters = {}
        for r in analysis.get("per_task", []):
            if r["condition"] == condition:
                clusters.setdefault(r["task_id"], []).append(int(r["task_success"]))
        points = [100 * statistics.mean(v) for v in clusters.values()]
        ax.scatter([i+j for j in _jitter(len(points), 100+i)], points, s=13,
                   color=COLORS[0], alpha=.22, linewidths=0, zorder=2)
        _point_interval(ax, i, row["success_pct"], row.get("success_ci95_low_pct"),
                        row.get("success_ci95_high_pct"), COLORS[0])
        ax.text(i, 105, f"n={row.get('n', 0)}", ha="center", color="#475569", fontsize=7)


def draw_delta(ax, analysis):
    _base_axes(ax, "B  Change against matched baseline", "Success change (percentage points)")
    rows = {r["condition"]: r for r in analysis.get("summary", [])}
    if not analysis.get("n_records"):
        _empty_axes(ax, "No measured paired differences\nNo claims about compression effects")
        return
    ax.axhline(0, color="#64748b", linewidth=.8, linestyle="--", zorder=1)
    extents = [0]
    for i, condition in enumerate(ALL_CONDITIONS):
        if condition == "none":
            ax.text(i, .5, "Reference", ha="center", va="center", rotation=90,
                    color="#64748b", fontsize=8, transform=ax.get_xaxis_transform())
            continue
        row = rows.get(condition, {})
        center = row.get("paired_delta_pp")
        if center is None:
            ax.text(i, .5, "Not paired", ha="center", va="center", rotation=90,
                    color="#64748b", fontsize=8, transform=ax.get_xaxis_transform())
            continue
        points = [r["delta_pp"] for r in analysis.get("paired_tasks", []) if r["condition"] == condition]
        ax.scatter([i+j for j in _jitter(len(points), 200+i)], points, s=13,
                   color=COLORS[1], alpha=.25, linewidths=0, zorder=2)
        low, high = row.get("paired_delta_ci95_low_pp"), row.get("paired_delta_ci95_high_pp")
        _point_interval(ax, i, center, low, high, COLORS[1])
        extents.extend(points + [center] + ([low, high] if low is not None and high is not None else []))
    span = max(max(extents)-min(extents), 20)
    ax.set_ylim(max(-110, min(extents)-.14*span), min(110, max(extents)+.14*span))


def draw_tokens(ax, analysis):
    _base_axes(ax, "C  Measured handoff compression", "Compressed tokens / handoff tokens")
    ratios = task_token_ratios(analysis)
    if not ratios:
        _empty_axes(ax, "No measured token ratios" if not analysis.get("n_records") else
                    "Tokenizer counts unavailable\nNo character-to-token estimates plotted")
        return
    ax.axhline(1, color="#64748b", linewidth=.8, linestyle="--", zorder=1)
    rows = {r["condition"]: r for r in analysis.get("summary", [])}
    maximum = 1
    for i, condition in enumerate(ALL_CONDITIONS):
        points = [r["mean_retained_token_ratio"] for r in ratios if r["condition"] == condition]
        if not points:
            continue
        ax.scatter([i+j for j in _jitter(len(points), 300+i)], points, s=14,
                   color=COLORS[2], alpha=.28, linewidths=0, zorder=2)
        center = rows.get(condition, {}).get("compression_retained_token_ratio")
        if center is not None:
            ax.plot([i], [center], "D", markersize=5, color=COLORS[2], zorder=5)
        maximum = max(maximum, *points, center or 0)
    ax.set_ylim(0, maximum * 1.13)


def _decorate(fig, analysis, panel_count):
    mock = analysis.get("mock", False)
    empty = not analysis.get("n_records")
    title = ("MOCK INFRASTRUCTURE CHECK — no scientific model results" if mock else
             "Run 2 · GPT-4o · no adversarial attack")
    fig.suptitle(title, x=.02, ha="left", y=.985, fontsize=13,
                 color="#9f1239" if mock else "#0f172a", fontweight="bold")
    if empty:
        subtitle = "No measured GPT-4o results — credential and budget unavailable" if not mock else "No completed mock records"
    else:
        subtitle = f"{len(analysis.get('task_ids', []))} tasks · {analysis.get('repeats', '?')} repeat(s) · " + (
            "complete planned sweep" if analysis.get("complete") else "partial / smoke results")
    fig.text(.02, .925, subtitle, ha="left", fontsize=9, color="#475569")
    if mock:
        fig.text(.5, .51, "MOCK — INFRASTRUCTURE ONLY", ha="center", va="center",
                 rotation=17, fontsize=24 if panel_count == 3 else 17,
                 color="#9f1239", alpha=.14, fontweight="bold")
    caption = ("Diamonds: condition estimates. Dots: task means. Intervals: 95% task-cluster bootstrap; "
               "18 tasks give limited uncertainty resolution.\n"
               "Token diamonds use pooled token totals; task dots use mean within-task ratios. "
               "Required-write success is the current scorer proxy, not execution utility.")
    if panel_count == 1:
        caption = ("Diamonds: estimates; dots: task means. 95% task-cluster intervals.\n"
                   "Small fixed suite; boundary bootstrap intervals may degenerate.\n"
                   "Tokens: pooled-total diamonds and within-task mean-ratio dots.\n"
                   "Required-write success is a scorer proxy, not execution utility.")
    if empty:
        caption = ("All six implemented conditions are shown; no rates, differences, token ratios, or uncertainty bounds were measured.\n"
                   "Axes contain no zero-value placeholders. These figures document the blocked experiment and are ready to regenerate from measured records.")
        if panel_count == 1:
            caption = ("Six implemented conditions; no measured data or zero-value placeholders.\n"
                       "Figures regenerate from validated records once the experiment can run.\n"
                       "No effects of compression or uncertainty intervals are claimed.")
    fig.text(.02, .025, caption, ha="left", va="bottom", fontsize=7, color="#475569", linespacing=1.45)


def make_figures(analysis):
    setup_matplotlib()
    import matplotlib.pyplot as plt
    drawers = (draw_success, draw_delta, draw_tokens)
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.6))
    for ax, drawer in zip(axes, drawers):
        drawer(ax, analysis)
    _decorate(fig, analysis, 3)
    fig.subplots_adjust(left=.055, right=.985, bottom=.235, top=.82, wspace=.3)
    figures = {"run2_overview": fig}
    for name, drawer in zip(("task_success", "paired_delta", "token_retention"), drawers):
        single, ax = plt.subplots(figsize=(7.2, 5.6))
        drawer(ax, analysis)
        _decorate(single, analysis, 1)
        single.subplots_adjust(left=.12, right=.98, bottom=.28, top=.81)
        figures[name] = single
    return figures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.analysis_dir / "analysis.json"
    source_bytes = source.read_bytes()
    analysis = json.loads(source_bytes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures = make_figures(analysis)
    for name, fig in figures.items():
        for extension in ("pdf", "svg", "png"):
            fig.savefig(args.out_dir / f"{name}.{extension}", dpi=300, facecolor="white")
    import matplotlib.pyplot as plt
    for figure in figures.values():
        plt.close(figure)
    ratios = task_token_ratios(analysis)
    with (args.out_dir / "token_ratio_task_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "task_id", "observed_repeats", "mean_retained_token_ratio"])
        writer.writeheader()
        writer.writerows(ratios)
    metadata = {"analysis_sha256": hashlib.sha256(source_bytes).hexdigest(), "run_id": analysis.get("run_id"),
                "mock": bool(analysis.get("mock")), "n_records": analysis.get("n_records", 0),
                "data_status": "measured" if analysis.get("n_records") else "no_measured_results",
                "interval": "95% percentile task-cluster bootstrap, preserving all repeats and paired arms",
                "token_ratio": "actual tokenizer counts only; pooled-total estimate with per-task mean-ratio points",
                "formats": ["pdf", "svg", "png"], "png_dpi": 300,
                "limitations": "Scorer utility proxy; lexical retention is not semantic retention; no causal claims."}
    (args.out_dir / "figure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(figures)} figures in PDF/SVG/PNG to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
