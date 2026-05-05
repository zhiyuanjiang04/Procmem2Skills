"""Aggregate Opus-judge JSONL into the validated terminal-bench testset.

Inputs:
  --judge   testsets/runs/<date>-tb-judge.jsonl   (per-(task,skill) verdicts)
  --tasks   testsets/data/terminal_bench_tasks.jsonl
  --topk    testsets/data/terminal_bench_topk.jsonl  (for skill-name lookup)

Outputs:
  --out     testsets/data/terminal_bench_validated.jsonl
            One row per *kept* task: {task_id, task_description,
            gt_skills:[{skill_id,slug,name,rank,reason}], n_yes, n_no}
  --report  optional markdown summary path

Drop rule: tasks where every top-K candidate is NO (or unparseable / error)
are dropped — they have no validated GT.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--judge", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--report", default=None)
    args = p.parse_args()

    tasks = {}
    with open(args.tasks) as f:
        for line in f:
            r = json.loads(line)
            tasks[r["task_id"]] = r

    judgements: dict[str, list[dict]] = {}
    verdict_counts: Counter[str] = Counter()
    with open(args.judge) as f:
        for line in f:
            r = json.loads(line)
            judgements.setdefault(r["task_id"], []).append(r)
            verdict_counts[r["verdict"]] += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    rows_kept = []
    rows_dropped = []
    with out_path.open("w") as out_f:
        for tid, task in tasks.items():
            judges = judgements.get(tid, [])
            yes = [j for j in judges if j["verdict"] == "yes"]
            if not yes:
                dropped += 1
                rows_dropped.append(tid)
                continue
            yes_sorted = sorted(yes, key=lambda j: j["rank"])
            row = {
                "task_id": tid,
                "task_description": task["task_description"],
                "gt_skills": [
                    {
                        "skill_id": j["skill_id"],
                        "slug": j["skill_slug"],
                        "name": j["skill_name"],
                        "rank": j["rank"],
                        "reason": j["reason"],
                    }
                    for j in yes_sorted
                ],
                "n_yes": len(yes),
                "n_judged": len(judges),
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            rows_kept.append((tid, len(yes)))

    summary = (
        f"# Terminal-Bench validated testset summary\n\n"
        f"- Source judge: `{args.judge}`\n"
        f"- Tasks total: {len(tasks)}\n"
        f"- Tasks kept (≥1 YES): **{kept}**\n"
        f"- Tasks dropped (all NO/error): {dropped}\n"
        f"- Verdict mix: {dict(verdict_counts)}\n"
        f"- Multi-GT tasks (n_yes ≥ 2): "
        f"{sum(1 for _, n in rows_kept if n >= 2)}\n"
        f"\n## GT-count distribution (kept tasks)\n"
    )
    dist = Counter(n for _, n in rows_kept)
    for n in sorted(dist):
        summary += f"- {n} GT skills: {dist[n]} tasks\n"

    if args.report:
        Path(args.report).write_text(summary)
    print(summary, file=sys.stderr)
    print(f"[aggregate_judge] wrote {kept} rows to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
