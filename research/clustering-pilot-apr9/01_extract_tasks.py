"""Extract task descriptions and metadata from terminal-bench original-tasks."""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
TASKS_DIR = ROOT.parent / "terminal-bench" / "original-tasks"
OUT = ROOT / "outputs" / "tasks.jsonl"


def main():
    if not TASKS_DIR.exists():
        raise SystemExit(f"tasks dir not found: {TASKS_DIR}")

    rows = []
    skipped = []
    for task_dir in sorted(TASKS_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        yaml_path = task_dir / "task.yaml"
        if not yaml_path.exists():
            skipped.append((task_dir.name, "no task.yaml"))
            continue
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
        except Exception as e:
            skipped.append((task_dir.name, f"yaml parse: {e}"))
            continue
        if not data or not isinstance(data, dict):
            skipped.append((task_dir.name, "empty yaml"))
            continue
        instruction = data.get("instruction", "").strip()
        if not instruction:
            skipped.append((task_dir.name, "no instruction"))
            continue
        rows.append({
            "task_id": task_dir.name,
            "instruction": instruction,
            "category": data.get("category", ""),
            "tags": data.get("tags") or [],
            "difficulty": data.get("difficulty", ""),
            "author_name": data.get("author_name", ""),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"extracted {len(rows)} tasks → {OUT}")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for name, reason in skipped[:10]:
            print(f"  {name}: {reason}")
        if len(skipped) > 10:
            print(f"  ... ({len(skipped) - 10} more)")


if __name__ == "__main__":
    main()
