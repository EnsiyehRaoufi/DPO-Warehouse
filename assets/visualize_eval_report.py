#!/usr/bin/env python3
"""visualize_eval_report.py - plots for eval_report_<timestamp>.json,
this project's DPO before/after evaluation output.

usage:
    # single run - the 4 core plots
    python visualize_eval_report.py out/eval_report_20260821_085504.json

    # also add the "what's driving the regressions" breakdown, using the
    # matching log file (regression_cases lives there now, not in the
    # report - see eval_model.py)
    python visualize_eval_report.py out/eval_report_20260821_085504.json \\
        --log out/eval_report_log_20260821_085504.json

    # multiple runs - adds a trend-across-runs plot too, sorted by
    # training_settings.trained_at when available (falls back to
    # filename order if a report has no training_settings, e.g. an
    # older report from before that field existed)
    python visualize_eval_report.py out/eval_report_*.json --out-dir out/plots

all plots save as PNGs to --out-dir (default out/plots) and nothing is
shown interactively - this is meant to run headlessly (Colab, CI, a
plain script), not in a notebook with inline display.
"""

import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (report field prefix, display label, whether higher is better) - the
# 4 dimension metrics eval_report.json reports as *_before/*_after pairs.
# hallucination_rate is the one where LOWER is better - handled explicitly
# throughout rather than assumed, since silently treating it like the
# other 3 would make an improving model look like it's getting worse.
DIM_METRICS = [
    ("factual_correctness", "Factual Correctness\n(grounding)", True),
    ("operational_quality", "Operational Quality", True),
    ("hallucination_rate", "Hallucination Rate", False),
    ("uncertainty_handling", "Uncertainty Handling", True),
]

COLOR_BASE = "#94a3b8"
COLOR_DPO = "#2563eb"
COLOR_GOOD = "#16a34a"
COLOR_BAD = "#dc2626"
COLOR_NEUTRAL = "#94a3b8"


def load_report(path):
    """Loads one eval_report_<timestamp>.json file and tags it with its
    source path, used for sorting when multiple reports are given."""
    d = json.load(open(path))
    d["_path"] = path
    return d


def plot_before_after(report, out_dir):
    """grouped bar chart: base vs DPO for each of the 4 dimension metrics,
    each bar colored green/red depending on whether that specific change
    was actually an improvement (accounting for hallucination_rate's
    reversed direction, not just whether the bar went up)."""
    labels = [label for _, label, _ in DIM_METRICS]
    before = [report[f"{key}_before"] for key, _, _ in DIM_METRICS]
    after = [report[f"{key}_after"] for key, _, _ in DIM_METRICS]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars1 = ax.bar(x - width / 2, before, width, label="Base model", color=COLOR_BASE)
    bars2 = ax.bar(x + width / 2, after, width, label="DPO-tuned model", color=COLOR_DPO)

    for i, (key, _, higher_is_better) in enumerate(DIM_METRICS):
        # a difference this small (< 0.005) rounds to the same displayed
        # label either way - coloring it green/red anyway would visually
        # claim a change the reader can't actually see in the numbers
        # shown, so it gets a neutral color instead of a misleading one.
        diff = after[i] - before[i]
        if abs(diff) < 0.005:
            bars2[i].set_color(COLOR_NEUTRAL)
        else:
            improved = (diff > 0) if higher_is_better else (diff < 0)
            bars2[i].set_color(COLOR_GOOD if improved else COLOR_BAD)

    ax.set_ylabel("Score (0-1)")
    ax.set_title("Before vs after DPO - per-dimension scores\n"
                 "(DPO bar: green=improved, red=regressed, gray=change too small to show in the label)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right")
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # hallucination_rate is the one dimension where lower = better -
    # flagged directly on the chart so it's never misread at a glance
    halluc_idx = [k for k, _, _ in DIM_METRICS].index("hallucination_rate")
    ax.annotate("lower is better \u2193", xy=(halluc_idx, max(before[halluc_idx], after[halluc_idx]) + 0.05),
                ha="center", fontsize=8, style="italic", color="#64748b")

    fig.tight_layout()
    path = os.path.join(out_dir, "before_after_dimensions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_win_tie_regress(report, out_dir):
    """pie chart of improved/tied/regressed prompt counts."""
    values = [report["n_improved"], report["n_tied"], report["n_regressions"]]
    labels = ["Improved", "Tied", "Regressed"]
    colors = [COLOR_GOOD, COLOR_NEUTRAL, COLOR_BAD]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.pie(values, labels=labels, colors=colors, startangle=90, textprops={"fontsize": 11},
           autopct=lambda pct: f"{pct:.0f}%\n({int(round(pct / 100 * total))})")
    ax.set_title(f"DPO vs base outcome across {total} held-out prompts\n"
                 f"win_rate_vs_baseline = {report['win_rate_vs_baseline']:.1%}")
    fig.tight_layout()
    path = os.path.join(out_dir, "win_tie_regress.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_aggregate_score(report, out_dir):
    """baseline_score vs post_dpo_score, delta annotated in the title."""
    fig, ax = plt.subplots(figsize=(5, 5.5))
    vals = [report["baseline_score"], report["post_dpo_score"]]
    delta = report["delta"]
    bars = ax.bar(["Base model", "DPO-tuned"], vals, color=[COLOR_BASE, COLOR_GOOD if delta >= 0 else COLOR_BAD],
                   width=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Weighted aggregate score")
    ax.set_title(f"Aggregate eval score\ndelta = {delta:+.4f}", color=(COLOR_GOOD if delta >= 0 else COLOR_BAD))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "aggregate_score.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_regression_dim_breakdown(log, out_dir):
    """which of the 7 rubric dimensions is actually driving the
    regressions - needs the LOG file (regression_cases lives there, not
    in the report, since the report/log split - see eval_model.py)."""
    cases = (log or {}).get("regression_cases")
    if not cases:
        return None
    dim_drops = Counter()
    for c in cases:
        for k in c["dims_base"]:
            if c["dims_dpo"][k] < c["dims_base"][k]:
                dim_drops[k] += 1
    if not dim_drops:
        return None

    dims = sorted(dim_drops, key=lambda k: -dim_drops[k])
    counts = [dim_drops[k] for k in dims]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.barh(dims, counts, color=COLOR_BAD)
    ax.bar_label(bars, padding=3)
    ax.set_xlabel(f"# of {len(cases)} regression cases where this dimension dropped\n(a case can involve more than one dimension)")
    ax.set_title("What's driving the regressions?")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "regression_dimension_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_trend(reports, out_dir):
    """how delta/win_rate/uncertainty/hallucination moved across several
    runs - only produced when 2+ report files are given. sorted by
    training_settings.trained_at when available, falling back to the
    order the files were given in (e.g. for an older report saved before
    training_settings existed)."""
    if len(reports) < 2:
        return None

    def sort_key(r):
        ts = (r.get("training_settings") or {}).get("trained_at")
        return (0, ts) if ts else (1, r["_path"])

    reports = sorted(reports, key=sort_key)
    run_labels = []
    for i, r in enumerate(reports):
        ts = r.get("training_settings") or {}
        label = f"run {i + 1}"
        if ts:
            label += f"\nr={ts.get('lora_rank', '?')} b={ts.get('beta', '?')} ep={ts.get('epochs', '?')}"
        run_labels.append(label)

    x = np.arange(len(reports))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        ("delta", "Delta (post_dpo - baseline)", axes[0][0], True),
        ("win_rate_vs_baseline", "Win Rate vs Baseline", axes[0][1], True),
        ("uncertainty_handling_after", "Uncertainty Handling (after)", axes[1][0], True),
        ("hallucination_rate_after", "Hallucination Rate (after)", axes[1][1], False),
    ]
    for key, title, ax, higher_is_better in panels:
        vals = [r.get(key) for r in reports]
        ax.plot(x, vals, marker="o", color=COLOR_DPO, linewidth=2)
        if key == "delta":
            ax.axhline(0, color=COLOR_NEUTRAL, linewidth=0.8, linestyle="--")
        ax.set_title(title + ("" if higher_is_better else "  (lower is better \u2193)"))
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Metric trends across training runs", fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, "trend_across_runs.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description="Visualize eval_report.json results")
    ap.add_argument("reports", nargs="+", help="one or more eval_report_<timestamp>.json files")
    ap.add_argument("--log", default=None,
                     help="matching eval_report_log_<timestamp>.json - enables the regression-dimension "
                          "breakdown plot (only used for the LAST report given, if multiple)")
    ap.add_argument("--out-dir", default="out/plots", help="directory to save PNGs into")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    reports = [load_report(p) for p in args.reports]
    latest = reports[-1]

    saved = [
        plot_before_after(latest, args.out_dir),
        plot_win_tie_regress(latest, args.out_dir),
        plot_aggregate_score(latest, args.out_dir),
    ]

    if args.log:
        log = json.load(open(args.log))
        p = plot_regression_dim_breakdown(log, args.out_dir)
        if p:
            saved.append(p)

    p = plot_trend(reports, args.out_dir)
    if p:
        saved.append(p)

    print(f"saved {len(saved)} plot(s) to {args.out_dir}:")
    for p in saved:
        print(" ", p)


if __name__ == "__main__":
    main()
