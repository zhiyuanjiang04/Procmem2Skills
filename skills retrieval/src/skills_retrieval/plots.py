from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STRATEGY_LABEL = {
    "random": "Random distractors",
    "hard_neg_semantic": "Hard neg (semantic)",
    "easy_neg": "Easy neg (cluster-far)",
    "hard_neg_functional": "Hard neg (functional)",
    "adversarial": "Adversarial",
}
STRATEGY_COLOR = {
    "random": "#4C72B0",
    "hard_neg_semantic": "#C44E52",
    "easy_neg": "#55A868",
    "hard_neg_functional": "#8172B2",
    "adversarial": "#937860",
}


def _strategies_present(agg: dict) -> list[str]:
    seen: dict[str, None] = {}
    for (strategy, _n) in agg.keys():
        seen.setdefault(strategy, None)
    return list(seen.keys())


def _ns_for_strategy(agg: dict, strategy: str) -> list[int]:
    return sorted(n for (s, n) in agg.keys() if s == strategy)


def plot_collapse_curve(agg: dict, out_path: Path, metric: str = "awareness_recall5") -> None:
    """Plot metric vs pool size N on log x-axis, one line per strategy, CI band."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for strategy in _strategies_present(agg):
        ns = _ns_for_strategy(agg, strategy)
        means = [agg[(strategy, n)][metric]["mean"] for n in ns]
        los = [agg[(strategy, n)][metric]["ci_lo"] for n in ns]
        his = [agg[(strategy, n)][metric]["ci_hi"] for n in ns]
        color = STRATEGY_COLOR.get(strategy, None)
        label = STRATEGY_LABEL.get(strategy, strategy)
        ax.plot(ns, means, marker="o", label=label, color=color)
        ax.fill_between(ns, los, his, alpha=0.2, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Pool size N")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(f"Selection collapse: {metric}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_selection_aware_divergence(agg: dict, out_path: Path) -> None:
    """Two subplots: (left) awareness_recall5 and selection_top1 overlaid per strategy;
    (right) sel|aware − awareness_recall5 per strategy (the divergence).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for strategy in _strategies_present(agg):
        ns = _ns_for_strategy(agg, strategy)
        color = STRATEGY_COLOR.get(strategy, None)
        label = STRATEGY_LABEL.get(strategy, strategy)

        aware_means = [agg[(strategy, n)]["awareness_recall5"]["mean"] for n in ns]
        sel_means = [agg[(strategy, n)]["selection_top1"]["mean"] for n in ns]
        sa_means = [agg[(strategy, n)]["selection_given_aware"]["mean"] for n in ns]

        ax1.plot(ns, aware_means, marker="o", linestyle="-", color=color, label=f"{label} (aware)")
        ax1.plot(ns, sel_means, marker="x", linestyle="--", color=color, alpha=0.6, label=f"{label} (sel)")

        diverge = [sa - aw for sa, aw in zip(sa_means, aware_means)]
        ax2.plot(ns, diverge, marker="s", color=color, label=label)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("Pool size N")
        ax.grid(True, which="both", alpha=0.3)
    ax1.set_ylabel("rate")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(frameon=False, fontsize=8, loc="lower left")
    ax1.set_title("Awareness vs Selection")
    ax2.set_ylabel("sel|aware − recall@5")
    ax2.axhline(0, color="black", alpha=0.3, linewidth=0.8)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    ax2.set_title("Selection | Aware divergence")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
