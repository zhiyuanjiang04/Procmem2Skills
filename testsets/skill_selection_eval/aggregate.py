"""Aggregate results JSONL into selection-accuracy metrics.

Metrics (only meaningful when GT is in pool, i.e. noise_mode != 'none'):
    Hit@1   = top1 ∈ gt_names               (Harbor-faithful primary metric)
    Recall  = any(s in gt_names for s in all_selected)
    MRR     = 1 / (rank of first GT in all_selected, else 0)

For noise_mode == 'none' (no-GT condition):
    refusal_rate    = fraction of trials with no_pick (top1 is None)
    hard_pick_rate  = 1 - refusal_rate (model "hallucinated" a selection)

Position-bias quick check (only when --position-bins given): bins gt_positions
by quantile of the (sorted) shuffled pool index, reports Hit@1 per bin.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def hit_at_1(top1: str | None, gt_names: list[str]) -> int:
    return int(top1 is not None and top1 in gt_names)


def recall_any(all_selected: list[str], gt_names: list[str]) -> int:
    return int(bool(set(all_selected) & set(gt_names)))


def mrr(all_selected: list[str], gt_names: list[str]) -> float:
    gt = set(gt_names)
    for i, s in enumerate(all_selected, 1):
        if s in gt:
            return 1.0 / i
    return 0.0


def summarise(rows: list[dict]) -> None:
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    errors: list[dict] = []
    for r in rows:
        if "error" in r:
            errors.append(r)
            continue
        by_cell[(r["pool_size"], r["noise_mode"])].append(r)

    print(f"\nTotal rows: {len(rows)}, errors: {len(errors)}")
    if errors:
        kinds: dict[str, int] = defaultdict(int)
        for e in errors:
            kinds[e["error"].split(":")[0]] += 1
        print("  error kinds:", dict(kinds))

    print("\n=== Metrics by (pool_size, noise_mode) ===")
    print(f"{'size':>6} {'noise':<8} {'n':>4} {'Hit@1':>7} {'Recall':>7} {'MRR':>7} {'refusal':>8}")
    for (sz, mode), trials in sorted(by_cell.items()):
        n = len(trials)
        if mode == "none":
            refusal = sum(1 for t in trials if not t["top1"]) / n
            hit, rec, mrr_val = float("nan"), float("nan"), float("nan")
            print(f"{sz:>6} {mode:<8} {n:>4} {'-':>7} {'-':>7} {'-':>7} {refusal:>8.2%}")
        else:
            hit = sum(hit_at_1(t["top1"], t["gt_names"]) for t in trials) / n
            rec = sum(recall_any(t["all_selected"], t["gt_names"]) for t in trials) / n
            mrr_val = statistics.fmean(mrr(t["all_selected"], t["gt_names"]) for t in trials)
            refusal = sum(1 for t in trials if not t["top1"]) / n
            print(f"{sz:>6} {mode:<8} {n:>4} {hit:>7.2%} {rec:>7.2%} {mrr_val:>7.3f} {refusal:>8.2%}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args(argv)
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    summarise(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
