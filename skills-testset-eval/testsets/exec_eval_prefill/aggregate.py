"""Aggregate run_trial.py JSONL → pass-rate table."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _load(paths):
    rows = []
    for p in paths:
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build(rows):
    by_cell = defaultdict(list)
    skipped = defaultdict(int)
    errors = defaultdict(int)
    for r in rows:
        if r.get("skipped"):
            skipped[(r.get("dataset", "?"), r.get("pool_size", 0), r.get("noise_mode", "?"))] += 1
            continue
        if "error" in r and r.get("pass") is None:
            errors[(r.get("dataset", "?"), r.get("pool_size", 0), r.get("noise_mode", "?"))] += 1
            continue
        by_cell[(r["dataset"], r["pool_size"], r["noise_mode"])].append(r)

    summary = []
    for key in sorted(set(list(by_cell.keys()) + list(skipped.keys()) + list(errors.keys()))):
        ds, sz, mode = key
        cell = by_cell.get(key, [])
        n = len(cell)
        if n:
            pr = sum(r["pass"] for r in cell) / n
            wall = statistics.median(r.get("agent_wall_s", 0) for r in cell)
        else:
            pr = None
            wall = None
        summary.append({
            "dataset": ds, "pool_size": sz, "noise_mode": mode,
            "n_trials": n,
            "n_skipped": skipped.get(key, 0),
            "n_errors": errors.get(key, 0),
            "pass_rate": round(pr, 4) if pr is not None else None,
            "median_wall_s": round(wall, 1) if wall is not None else None,
        })
    return summary


def fmt_md(summary):
    lines = [
        "| dataset | pool_size | noise | n | pass-rate | skip | err | wall_s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        pr = f"{s['pass_rate']*100:.1f}%" if s["pass_rate"] is not None else "—"
        wall = f"{s['median_wall_s']:.0f}" if s["median_wall_s"] is not None else "—"
        lines.append(
            f"| {s['dataset']} | {s['pool_size']} | {s['noise_mode']} | "
            f"{s['n_trials']} | **{pr}** | {s['n_skipped']} | {s['n_errors']} | {wall} |"
        )
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, nargs="+", required=True)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--md", type=Path, default=None)
    args = p.parse_args(argv)

    rows = _load(args.results)
    summary = build(rows)
    md = fmt_md(summary)
    print(md)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2))
    if args.md:
        args.md.write_text(md + "\n")


if __name__ == "__main__":
    main()
