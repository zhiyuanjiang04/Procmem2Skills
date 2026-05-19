"""Aggregate run_pass_rate.py JSONL into a pass-rate table.

Columns: dataset | n_noise | pool_size_avg | n | hit1 | recall | refusal

hit1   = top-1 picked is in GT set  (selection-pass; proxy for execution pass)
recall = |picked ∩ GT| / |GT|       (multi-GT scoring)
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _load(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build(rows: list[dict]) -> tuple[str, list[dict]]:
    by_cell = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        by_cell[(r.get("dataset"), r["n_noise"])].append(r)

    summary = []
    for (ds, nn), cell in sorted(by_cell.items()):
        hit = sum(r["hit1"] for r in cell) / len(cell)
        rec = sum(r["recall"] for r in cell) / len(cell)
        ref = sum(r.get("refusal", 0) for r in cell) / len(cell)
        pool_avg = statistics.mean(r["pool_size"] for r in cell)
        summary.append({
            "dataset": ds,
            "n_noise": nn,
            "pool_size_avg": round(pool_avg, 1),
            "n_trials": len(cell),
            "pass_rate_selection": round(hit, 4),
            "recall": round(rec, 4),
            "refusal": round(ref, 4),
        })

    # Pretty print
    lines = []
    header = f"{'dataset':<7} {'n_noise':>7} {'pool_avg':>8} {'n':>4} {'pass(sel)':>10} {'recall':>8} {'refusal':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for s in summary:
        lines.append(
            f"{s['dataset']:<7} {s['n_noise']:>7} {s['pool_size_avg']:>8} "
            f"{s['n_trials']:>4} {s['pass_rate_selection']*100:>9.1f}% "
            f"{s['recall']*100:>7.1f}% {s['refusal']*100:>7.1f}%"
        )
    return "\n".join(lines), summary


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, nargs="+", required=True)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--md", type=Path, default=None)
    args = p.parse_args(argv)

    rows = _load(args.results)
    table_str, summary = build(rows)
    print(table_str)

    if args.json:
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {args.json}", file=__import__("sys").stderr)
    if args.md:
        # Markdown variant
        md_lines = ["| dataset | n_noise | pool_size_avg | n | pass-rate (selection) | recall | refusal |",
                    "|---|---|---|---|---|---|---|"]
        for s in summary:
            md_lines.append(
                f"| {s['dataset']} | {s['n_noise']} | {s['pool_size_avg']} | {s['n_trials']} "
                f"| {s['pass_rate_selection']*100:.1f}% | {s['recall']*100:.1f}% | {s['refusal']*100:.1f}% |"
            )
        args.md.write_text("\n".join(md_lines) + "\n")
        print(f"Wrote {args.md}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
