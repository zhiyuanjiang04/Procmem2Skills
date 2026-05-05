"""Aggregate terminal-bench harness results.

TB has no validated GT skills (P4 scope), so metrics report selection
behavior given a candidate pool — not retrieval recall.

Per (pool_size, noise_mode) cell:
    n              : number of trials
    pick_rate      : 1 - refusal (model picked SOME skill)
    pick_in_topk   : top-1 ∈ topk_names (the retrieved candidate set)
    agree_top1     : top-1 == proxy_gt_name (the FAISS top-1 retrieval)
    refusal_rate   : top1 is None (model emitted NONE)

Note: pick_in_topk and agree_top1 are undefined for noise_mode='none'
(no topk in pool) and reported as '-'.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


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
            kinds[e.get("error", "?").split(":")[0]] += 1
        print("  error kinds:", dict(kinds))

    print("\n=== TB Metrics by (pool_size, noise_mode) ===")
    print(f"{'size':>6} {'noise':<10} {'n':>4} {'pick':>7} {'in_topk':>8} "
          f"{'top1=ret':>9} {'refusal':>8}")
    for (sz, mode), trials in sorted(by_cell.items()):
        n = len(trials)
        picks = [t for t in trials if t.get("top1")]
        pick_rate = len(picks) / n if n else 0.0
        refusal = 1.0 - pick_rate
        if mode == "none":
            in_topk = float("nan")
            agree_top1 = float("nan")
            print(f"{sz:>6} {mode:<10} {n:>4} {pick_rate:>7.2%} {'-':>8} {'-':>9} {refusal:>8.2%}")
        else:
            in_topk = sum(1 for t in trials if t.get("top1") in (t.get("topk_names") or [])) / n
            agree_top1 = sum(1 for t in trials if t.get("top1") and t.get("top1") == t.get("proxy_gt_name")) / n
            print(f"{sz:>6} {mode:<10} {n:>4} {pick_rate:>7.2%} {in_topk:>8.2%} {agree_top1:>9.2%} {refusal:>8.2%}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args(argv)
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    summarise(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
