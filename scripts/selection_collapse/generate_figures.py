#!/usr/bin/env python3
"""Generate figures for the Selection Collapse experiment.

Figure 1: Two-panel selection collapse (the paper's key result).
Figure 2: Hard-negative comparison bar chart at N=50.

Data source: data/selection_collapse/results/metrics/summary.json
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "data" / "selection_collapse" / "results" / "metrics"
FIGURES_DIR = ROOT / "data" / "selection_collapse" / "figures"
PAPER_DIR = ROOT / "paper"


def load_summary():
    """Load summary.json and index by (pool_size, condition)."""
    data = json.loads((METRICS_DIR / "summary.json").read_text())
    indexed = {}
    for entry in data:
        key = (entry["pool_size"], entry["condition"])
        indexed[key] = entry
    return data, indexed


def fig_selection_collapse(data, indexed):
    """Figure 1: Two-panel selection collapse figure.

    Panel (a): Awareness, Selection, and Selection|Aware vs pool size.
    Panel (b): Reasoning vs Attention failure decomposition.
    """
    # Extract random-distractor curves across pool sizes
    pool_sizes = [5, 50, 200, 1000]
    awareness = []
    awareness_se = []
    selection = []
    selection_se = []
    sel_given_aware = []

    for n in pool_sizes:
        entry = indexed[(n, "random")]
        awareness.append(entry["awareness_recall_at5"])
        awareness_se.append(entry["awareness_se"])
        selection.append(entry["selection_accuracy"])
        selection_se.append(entry["selection_se"])
        sel_given_aware.append(entry["selection_given_aware"])

    awareness = np.array(awareness)
    awareness_se = np.array(awareness_se)
    selection = np.array(selection)
    selection_se = np.array(selection_se)
    sel_given_aware = np.array(sel_given_aware)
    xs = np.array(pool_sizes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ── Panel (a): Selection Collapse Under Distractor Scaling ────
    ax1.plot(xs, awareness, 's--', color='#2563eb', linewidth=2, markersize=6,
             label='Awareness Recall@5')
    ax1.fill_between(xs, awareness - awareness_se, awareness + awareness_se,
                     color='#2563eb', alpha=0.15)

    ax1.plot(xs, selection, 'o-', color='#dc2626', linewidth=2, markersize=6,
             label='Selection Accuracy')
    ax1.fill_between(xs, selection - selection_se, selection + selection_se,
                     color='#dc2626', alpha=0.15)

    ax1.plot(xs, sel_given_aware, '^-', color='#16a34a', linewidth=2, markersize=6,
             label='Selection|Aware')

    # Vertical shaded region: collapse zone
    ax1.axvspan(50, 200, alpha=0.10, color='#f59e0b', zorder=0)
    ax1.text(np.sqrt(50 * 200), 10, 'Selection\nCollapse', ha='center', fontsize=9,
             color='#f59e0b', fontweight='bold')

    ax1.set_xscale('log')
    ax1.set_xticks(pool_sizes)
    ax1.set_xticklabels([str(n) for n in pool_sizes], fontsize=9)
    ax1.set_xlabel('Pool Size', fontsize=11)
    ax1.set_ylabel('Rate (%)', fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.set_title('(a) Selection Collapse Under Distractor Scaling',
                  fontsize=10, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel (b): Reasoning vs. Attention Failure ────────────────
    ax2.plot(xs, selection, 'o-', color='#dc2626', linewidth=2, markersize=6,
             label='Selection (unconditional)')
    ax2.plot(xs, sel_given_aware, '^-', color='#16a34a', linewidth=2, markersize=6,
             label='Selection|Aware')

    ax2.fill_between(xs, selection, sel_given_aware,
                     color='#6366f1', alpha=0.10, label='Awareness-attributable gap')

    ax2.set_xscale('log')
    ax2.set_xticks(pool_sizes)
    ax2.set_xticklabels([str(n) for n in pool_sizes], fontsize=9)
    ax2.set_xlabel('Pool Size', fontsize=11)
    ax2.set_ylabel('Rate (%)', fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.set_title('(b) Reasoning vs. Attention Failure',
                  fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout(w_pad=3)

    # Save to both locations
    for d in [FIGURES_DIR, PAPER_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_selection_collapse.pdf", bbox_inches='tight')
        fig.savefig(d / "fig_selection_collapse.png", dpi=150, bbox_inches='tight')

    plt.close(fig)
    print("Saved Figure 1: Selection Collapse (two-panel)")


def fig_hard_negative_comparison(data, indexed):
    """Figure 2: Hard-negative comparison bar chart at N=50.

    Three grouped bars (Awareness, Selection, Selection|Aware)
    comparing random vs hard-negative distractors.
    """
    random_entry = indexed[(50, "random")]
    hard_entry = indexed[(50, "hard_negative")]

    metrics = ['Awareness\nRecall@5', 'Selection\nAccuracy', 'Selection|\nAware']
    random_vals = [
        random_entry["awareness_recall_at5"],
        random_entry["selection_accuracy"],
        random_entry["selection_given_aware"],
    ]
    hard_vals = [
        hard_entry["awareness_recall_at5"],
        hard_entry["selection_accuracy"],
        hard_entry["selection_given_aware"],
    ]

    x = np.arange(len(metrics))
    width = 0.3

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(x - width / 2, random_vals, width, label='Random', color='#60a5fa')
    ax.bar(x + width / 2, hard_vals, width, label='Hard-negative', color='#f87171')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel('Rate (%)', fontsize=11)
    ax.set_title('Hard-Negative vs. Random Distractors (N=50)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    for d in [FIGURES_DIR, PAPER_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_hard_negative.pdf", bbox_inches='tight')
        fig.savefig(d / "fig_hard_negative.png", dpi=150, bbox_inches='tight')

    plt.close(fig)
    print("Saved Figure 2: Hard-negative comparison bar chart")


if __name__ == "__main__":
    data, indexed = load_summary()
    fig_selection_collapse(data, indexed)
    fig_hard_negative_comparison(data, indexed)
    print("All selection collapse figures generated.")
