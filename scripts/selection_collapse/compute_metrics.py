#!/usr/bin/env python3
"""
Compute metrics for the Selection Collapse experiment.

Reads raw results from the awareness probe (Stage 1) and selection (Stage 2),
computes three key metrics per (pool_size, condition) group:

- Awareness Recall@5: does the model recognize the GT skill in its top-5?
- Selection Accuracy: does the model use the GT skill?
- Selection|Aware: does the model use the GT skill GIVEN it recognized it?

Also computes bootstrap standard errors for each metric.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Project root and data paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "selection_collapse" / "results"


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_awareness_recall(results: List[Dict[str, Any]]) -> float:
    """Fraction of results where gt_in_top5 is True."""
    if not results:
        return 0.0
    return sum(1 for r in results if r["gt_in_top5"]) / len(results)


def compute_selection_accuracy(results: List[Dict[str, Any]]) -> float:
    """Fraction of results where any_gt_selected is True."""
    if not results:
        return 0.0
    return sum(1 for r in results if r["any_gt_selected"]) / len(results)


def compute_selection_given_aware(
    awareness_results: List[Dict[str, Any]],
    selection_results: List[Dict[str, Any]],
) -> float:
    """P(select GT | GT in awareness top-5).

    Joins awareness and selection results on (task_id, trial), filters to
    cases where gt_in_top5 is True, then computes the fraction where
    any_gt_selected is True.
    """
    # Build lookup from selection results
    sel_lookup: Dict[tuple, bool] = {}
    for r in selection_results:
        key = (r["task_id"], r["trial"])
        sel_lookup[key] = r["any_gt_selected"]

    # Filter to aware cases and check selection
    aware_selected = 0
    aware_total = 0
    for r in awareness_results:
        if not r["gt_in_top5"]:
            continue
        key = (r["task_id"], r["trial"])
        if key not in sel_lookup:
            continue
        aware_total += 1
        if sel_lookup[key]:
            aware_selected += 1

    if aware_total == 0:
        return 0.0
    return aware_selected / aware_total


def bootstrap_se(
    values: List[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> float:
    """Compute standard error of the mean via bootstrap resampling."""
    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0
    # Draw n_bootstrap resamples, compute mean of each
    means = np.array([
        rng.choice(arr, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    return float(np.std(means, ddof=0))


# ---------------------------------------------------------------------------
# Full pipeline: compute_all_metrics
# ---------------------------------------------------------------------------

def _load_json_results(directory: Path) -> List[Dict[str, Any]]:
    """Load all .json files from a directory into a flat list of result dicts."""
    results = []
    if not directory.exists():
        return results
    for json_path in sorted(directory.glob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            results.extend(data)
        elif isinstance(data, dict):
            results.append(data)
    return results


def _group_by(results: List[Dict[str, Any]], keys: tuple) -> Dict[tuple, List[Dict]]:
    """Group results by the specified keys."""
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in results:
        group_key = tuple(r.get(k) for k in keys)
        groups[group_key] = groups.get(group_key, [])
        groups[group_key].append(r)
    return groups


def compute_all_metrics() -> List[Dict[str, Any]]:
    """Read awareness + selection results, compute all metrics, save summary.

    Reads from:
      data/selection_collapse/results/awareness/*.json
      data/selection_collapse/results/selection/*.json

    Saves to:
      data/selection_collapse/results/metrics/summary.json

    Returns the summary list.
    """
    awareness_dir = DATA_DIR / "awareness"
    selection_dir = DATA_DIR / "selection"
    metrics_dir = DATA_DIR / "metrics"

    awareness_all = _load_json_results(awareness_dir)
    selection_all = _load_json_results(selection_dir)

    # Group by (pool_size, condition)
    group_keys = ("pool_size", "condition")
    aw_groups = _group_by(awareness_all, group_keys)
    sel_groups = _group_by(selection_all, group_keys)

    # Union of all group keys
    all_keys = set(aw_groups.keys()) | set(sel_groups.keys())

    summary = []
    for gk in sorted(all_keys):
        pool_size, condition = gk
        aw = aw_groups.get(gk, [])
        sel = sel_groups.get(gk, [])

        # Awareness recall
        awareness_recall = compute_awareness_recall(aw)
        aw_values = [1.0 if r["gt_in_top5"] else 0.0 for r in aw]
        aw_se = bootstrap_se(aw_values) if aw_values else 0.0

        # Selection accuracy
        sel_acc = compute_selection_accuracy(sel)
        sel_values = [1.0 if r["any_gt_selected"] else 0.0 for r in sel]
        sel_se = bootstrap_se(sel_values) if sel_values else 0.0

        # Selection | Aware
        sel_given_aware = compute_selection_given_aware(aw, sel)

        # Independent solve rate: fraction with independent_solved=True in selection
        indep_values = [1.0 if r.get("independent_solved", False) else 0.0 for r in sel]
        indep_rate = float(np.mean(indep_values)) if indep_values else 0.0

        # Average prompt tokens
        prompt_tokens = [r.get("prompt_tokens", 0) for r in sel if "prompt_tokens" in r]
        avg_prompt_tokens = float(np.mean(prompt_tokens)) if prompt_tokens else 0.0

        n_tasks = max(len(aw), len(sel))

        entry = {
            "pool_size": pool_size,
            "condition": condition,
            "n_tasks": n_tasks,
            "awareness_recall_at5": round(awareness_recall * 100, 1),
            "awareness_se": round(aw_se * 100, 1),
            "selection_accuracy": round(sel_acc * 100, 1),
            "selection_se": round(sel_se * 100, 1),
            "selection_given_aware": round(sel_given_aware * 100, 1),
            "independent_solve_rate": round(indep_rate * 100, 1),
            "avg_prompt_tokens": round(avg_prompt_tokens, 0),
        }
        summary.append(entry)

    # Save
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    _print_table(summary)

    return summary


def _print_table(summary: List[Dict[str, Any]]) -> None:
    """Print a human-readable summary table to stdout."""
    if not summary:
        print("No results to display.")
        return

    header = (
        f"{'Pool':>6} {'Condition':<12} {'N':>4} "
        f"{'Aware%':>7} {'SE':>5} "
        f"{'Select%':>8} {'SE':>5} "
        f"{'Sel|Aw%':>8} "
        f"{'Indep%':>7} "
        f"{'AvgTok':>8}"
    )
    print(header)
    print("-" * len(header))
    for e in summary:
        print(
            f"{e['pool_size']:>6} {e['condition']:<12} {e['n_tasks']:>4} "
            f"{e['awareness_recall_at5']:>7.1f} {e['awareness_se']:>5.1f} "
            f"{e['selection_accuracy']:>8.1f} {e['selection_se']:>5.1f} "
            f"{e['selection_given_aware']:>8.1f} "
            f"{e['independent_solve_rate']:>7.1f} "
            f"{e['avg_prompt_tokens']:>8.0f}"
        )


if __name__ == "__main__":
    compute_all_metrics()
