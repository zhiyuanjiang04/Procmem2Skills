"""Build a JSONL of terminal-bench task IDs + instruction text.

Walks terminal-bench/original-tasks/<task_id>/task.yaml, extracts the
`instruction:` field, writes one row per task to:
    testsets/data/terminal_bench_tasks.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

TB_DIR = Path("/anvil/projects/x-cis260386/william/procmem2skills/terminal-bench/original-tasks")
OUT = Path(__file__).resolve().parents[2] / "testsets" / "data" / "terminal_bench_tasks.jsonl"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = []
    for task_dir in sorted(TB_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        ty = task_dir / "task.yaml"
        if not ty.exists():
            skipped.append((task_dir.name, "no task.yaml"))
            continue
        try:
            data = yaml.safe_load(ty.read_text())
        except Exception as e:
            skipped.append((task_dir.name, f"yaml: {e}"))
            continue
        instr = (data or {}).get("instruction", "")
        if not isinstance(instr, str) or not instr.strip():
            skipped.append((task_dir.name, "empty instruction"))
            continue
        rows.append({
            "task_id": task_dir.name,
            "task_description": instr.strip(),
            "category": data.get("category"),
            "difficulty": data.get("difficulty"),
            "tags": data.get("tags") or [],
        })
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"Wrote {len(rows)} tasks to {OUT}")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for s in skipped[:10]:
            print(f"  {s[0]}: {s[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
