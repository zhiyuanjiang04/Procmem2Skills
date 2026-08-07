#!/usr/bin/env python3
"""Build SkillsBench task, GT-skill, and trace-stub manifests."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import (
    DEFAULT_ROOT,
    DEFAULT_SKILLSBENCH_TASKS,
    normalize_slug,
    skill_description,
    skill_name_from_md,
    task_description,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SkillsBench manifests for skill retrieval.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_SKILLSBENCH_TASKS)
    parser.add_argument("--benchmark", default="skillsbench")
    parser.add_argument("--task-name", action="append", default=[], help="Optional task filter; repeatable.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_skill_dir(src_skill_md: Path, dst_dir: Path, *, overwrite: bool) -> Path:
    src_dir = src_skill_md.parent
    if dst_dir.exists():
        if not overwrite:
            return dst_dir / "SKILL.md"
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    return dst_dir / "SKILL.md"


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    tasks_root = args.tasks_root.resolve()
    task_filter = {x.strip() for x in args.task_name if x.strip()}

    gt_pool = root / "pools" / "gt_pool" / args.benchmark
    gt_primary = root / "pools" / "gt_primary" / args.benchmark
    trace_root = root / "pools" / "trace_stubs" / args.benchmark
    manifests = root / "manifests"

    task_rows: list[dict] = []
    active_rows: list[dict] = []
    skipped: list[dict] = []

    for task_dir in sorted(p for p in tasks_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        task_name = task_dir.name
        if task_filter and task_name not in task_filter:
            continue
        skill_mds = sorted((task_dir / "environment" / "skills").glob("*/SKILL.md"))
        if not skill_mds:
            skipped.append({"task_name": task_name, "reason": "no_environment_skills"})
            continue

        gt_entries: list[dict] = []
        for skill_md in skill_mds:
            skill_slug = normalize_slug(skill_md.parent.name, default="skill")
            dst = gt_pool / task_name / skill_slug
            copied_md = copy_skill_dir(skill_md, dst, overwrite=args.overwrite)
            gt_entries.append(
                {
                    "skill_name": skill_name_from_md(copied_md),
                    "skill_slug": skill_slug,
                    "skill_md": str(copied_md),
                    "description": skill_description(copied_md),
                }
            )

        primary = gt_entries[0]
        primary_dst_dir = gt_primary / task_name
        primary_md = copy_skill_dir(Path(primary["skill_md"]), primary_dst_dir, overwrite=True)
        primary = {**primary, "primary_skill_md": str(primary_md)}

        desc = task_description(task_dir)
        task_rows.append(
            {
                "benchmark": args.benchmark,
                "task_name": task_name,
                "task_dir": str(task_dir),
                "task_description": desc,
                "primary_gt_skill": primary,
                "gt_skills": gt_entries,
                "gt_skill_count": len(gt_entries),
            }
        )
        active_rows.append({"benchmark": args.benchmark, "task_name": task_name, "task_dir": str(task_dir)})

        stub_dir = trace_root / task_name
        stub = {
            "benchmark": args.benchmark,
            "task_name": task_name,
            "trial_name": f"{task_name}__retrieval_stub",
            "status": "success",
            "steps": [],
            "source": "skill_retrieval_trace_stub",
        }
        write_json(stub_dir / "stub.json", stub)

    n_tasks = write_jsonl(manifests / f"task_skill_map.{args.benchmark}.jsonl", task_rows)
    write_jsonl(manifests / f"active_tasks.{args.benchmark}.jsonl", active_rows)
    write_json(manifests / f"task_skill_map.{args.benchmark}.summary.json", {
        "benchmark": args.benchmark,
        "tasks_root": str(tasks_root),
        "task_count": n_tasks,
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
        "gt_pool": str(gt_pool),
        "gt_primary": str(gt_primary),
        "trace_stub_root": str(trace_root),
    })
    print(f"wrote {n_tasks} task rows")
    print(f"gt_pool={gt_pool}")
    print(f"trace_stub_root={trace_root}")
    if skipped:
        print(f"skipped {len(skipped)} tasks without environment skills")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
