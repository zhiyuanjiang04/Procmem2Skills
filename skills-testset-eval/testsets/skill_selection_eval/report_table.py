"""Aggregate unified eval results into a Hit@1 success-rate table.

Reads one or two JSONL files produced by run_unified.py and prints:

  Dataset           | Pool |  Random |  Hard (emb-near) |  Easy (emb-far)
  SkillsBench (88)  |    5 |  76.1 % |          67.0 %  |         78.4 %
  ...               |  50  |  ...    |          ...     |         ...
  ...               | 500  |  ...    |          ...     |         ...
  TerminalBench (62)|    5 |  74.2 % |          69.4 %  |         75.8 %
  ...

Metric: Hit@1 = top-1 model selection is a validated GT skill.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _dataset_label(dataset: str, n_tasks: int) -> str:
    if dataset == "sb":
        return f"SkillsBench ({n_tasks})"
    return f"TerminalBench ({n_tasks})"


def _pct(v: float | None) -> str:
    if v is None:
        return "  —  "
    return f"{v * 100:5.1f}%"


def build_table(rows: list[dict]) -> str:
    # Group by (dataset, pool_size, noise_mode)
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    task_counts: dict[str, set] = defaultdict(set)

    for r in rows:
        if "error" in r:
            continue
        ds = r.get("dataset", "sb")
        task_counts[ds].add(r["task_id"])
        key = (ds, r["pool_size"], r["noise_mode"])
        by_cell[key].append(r)

    noise_cols = ["random", "hard", "easy"]
    datasets = sorted(task_counts.keys())

    header = (
        f"{'Dataset':<24} {'Pool':>5}  "
        f"{'Random':>8}  {'Emb-Near (hard)':>15}  {'Emb-Far (easy)':>14}  {'n':>4}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for ds in datasets:
        n_tasks = len(task_counts[ds])
        label = _dataset_label(ds, n_tasks)
        pool_sizes = sorted({k[1] for k in by_cell if k[0] == ds})
        for i, sz in enumerate(pool_sizes):
            prefix = f"{label:<24}" if i == 0 else f"{'':24}"
            vals = []
            n_trials = 0
            for mode in noise_cols:
                cell = by_cell.get((ds, sz, mode), [])
                n_trials = len(cell)
                if not cell:
                    vals.append("  N/A ")
                else:
                    hit = sum(r.get("hit1", 0) for r in cell) / len(cell)
                    vals.append(_pct(hit))
            lines.append(
                f"{prefix} {sz:>5}  "
                f"{vals[0]:>8}  {vals[1]:>15}  {vals[2]:>14}  {n_trials:>4}"
            )
        lines.append(sep)

    return "\n".join(lines)


def summarise_json(rows: list[dict]) -> list[dict]:
    """Return machine-readable summary list (for downstream use)."""
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        key = (r.get("dataset", "sb"), r["pool_size"], r["noise_mode"])
        by_cell[key].append(r)
    out = []
    for (ds, sz, mode), cell in sorted(by_cell.items()):
        n = len(cell)
        hit = sum(r.get("hit1", 0) for r in cell) / n if n else None
        recall = sum(r.get("recall", 0) for r in cell) / n if n else None
        refusal = sum(1 for r in cell if not r.get("top1")) / n if n else None
        out.append({"dataset": ds, "pool_size": sz, "noise_mode": mode,
                    "n": n, "hit1": hit, "recall": recall, "refusal": refusal})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, nargs="+", required=True,
                   help="One or more JSONL files from run_unified.py")
    p.add_argument("--json", dest="output_json", type=Path, default=None,
                   help="Also write machine-readable summary JSON here")
    args = p.parse_args(argv)

    rows = _load_rows(args.results)
    errors = sum(1 for r in rows if "error" in r)
    print(f"Loaded {len(rows)} rows ({errors} errors) from {[str(x) for x in args.results]}\n")

    print(build_table(rows))

    if args.output_json:
        summary = summarise_json(rows)
        args.output_json.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary JSON written to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
