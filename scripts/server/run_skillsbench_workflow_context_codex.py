#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from procmem2skills.runtime.workflow_memory import WorkflowMemoryIndex, normalize_task_key

MARKER_BEGIN = "<!-- PROCMEM_WORKFLOW_CONTEXT_BEGIN -->"
MARKER_END = "<!-- PROCMEM_WORKFLOW_CONTEXT_END -->"
DEFAULT_WORKFLOW_INTRO = (
    "## Retrieved Workflow Memory\n"
    "Below are induced workflows from prior runs of this same task. "
    "Use them as procedural hints, adapt to current environment, and verify before execution.\n"
)


@dataclass(frozen=True)
class SkillEntry:
    skill_id: str
    path: Path
    search_blob: str


def normalize_experiment_name(value: str, *, max_length: int = 56) -> str:
    tokens = re.findall(r"[a-z0-9]+", (value or "").strip().lower())
    if not tokens:
        return "experiment"
    slug = "-".join(tokens)
    if len(slug) <= max_length:
        return slug
    trimmed = slug[:max_length].rstrip("-")
    return trimmed or "experiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified SkillsBench codex-native launcher with 3 modes: "
            "no-skills | workflow | skills. "
            "Always uses native codex agent, never custom agent."
        )
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--mode", choices=["no-skills", "workflow", "skills"], default="workflow")

    parser.add_argument("--model", default="openai/gpt-5.3-codex")
    parser.add_argument("--dataset", default="skillsbench")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/skillsbench"))
    parser.add_argument(
        "--skillsbench-tasks-path",
        type=Path,
        default=Path("benchmarks/skillsbench/tasks"),
        help="Source SkillsBench tasks directory.",
    )

    # workflow mode
    parser.add_argument("--workflow-memory-path", type=Path, help="Grouped workflow JSON file.")
    parser.add_argument(
        "--workflow-attempt-filter",
        choices=["all", "success-only"],
        default="all",
        help="Which attempts are eligible when injecting workflows.",
    )
    parser.add_argument("--workflow-max-attempts", type=int, default=5)
    parser.add_argument("--workflow-max-workflows-per-attempt", type=int, default=2)
    parser.add_argument("--workflow-max-steps-per-workflow", type=int, default=12)
    parser.add_argument("--workflow-intro-text", default=DEFAULT_WORKFLOW_INTRO)

    # skills mode
    parser.add_argument("--skills-repository", type=Path, help="Distilled skill repository path.")
    parser.add_argument(
        "--skills-injection-scope",
        choices=["all", "same-task"],
        default="all",
        help="Inject all skills into every task, or only task-matched skills.",
    )
    parser.add_argument(
        "--skills-prefix",
        default="procmem--",
        help="Prefix for injected skill directory names to avoid collisions.",
    )

    # run controls
    parser.add_argument("--n-attempts", type=int, default=5)
    parser.add_argument("--n-concurrent", type=int, default=5)
    parser.add_argument("--task-name", action="append", default=[])
    parser.add_argument("--exclude-task-name", action="append", default=[])
    parser.add_argument("--n-tasks", type=int)
    parser.add_argument("--base-url")
    parser.add_argument("--working-dir")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_path(project_root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (project_root / path).resolve()


def _copy_tasks(source_tasks_dir: Path, destination_tasks_dir: Path) -> None:
    if not source_tasks_dir.is_dir():
        raise FileNotFoundError(f"skillsbench tasks path not found: {source_tasks_dir}")
    if destination_tasks_dir.exists():
        shutil.rmtree(destination_tasks_dir)
    shutil.copytree(source_tasks_dir, destination_tasks_dir)


def _strip_previous_workflow_block(text: str) -> str:
    if MARKER_BEGIN in text and MARKER_END in text:
        return text.split(MARKER_BEGIN, 1)[0].rstrip()
    return text.rstrip()


def _load_grouped_workflows(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"workflow payload must be an object: {path}")
    return payload


def _filter_workflows_by_attempt_status(payload: dict, attempt_filter: str) -> dict:
    if attempt_filter == "all":
        return payload
    filtered: dict = {}
    for task_key, attempts in payload.items():
        if not isinstance(attempts, list):
            continue
        kept = []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            status = str(attempt.get("status") or "").strip().lower()
            if status == "success":
                kept.append(attempt)
        filtered[str(task_key)] = kept
    return filtered


def apply_workflow_context(
    *,
    tasks_dir: Path,
    workflow_memory_path: Path,
    workflow_attempt_filter: str,
    intro_text: str,
    max_attempts: int,
    max_workflows_per_attempt: int,
    max_steps_per_workflow: int,
) -> dict:
    payload = _load_grouped_workflows(workflow_memory_path)
    payload = _filter_workflows_by_attempt_status(payload, workflow_attempt_filter)
    index = WorkflowMemoryIndex.from_grouped_attempts(payload)

    task_total = 0
    task_injected = 0
    task_missing_workflow = 0

    for task_dir in sorted([p for p in tasks_dir.iterdir() if p.is_dir()]):
        instruction_file = task_dir / "instruction.md"
        if not instruction_file.is_file():
            continue
        task_total += 1

        original = instruction_file.read_text(encoding="utf-8")
        base_text = _strip_previous_workflow_block(original)
        rendered = index.render_task_memory(
            task_dir.name,
            max_attempts=max_attempts,
            max_workflows_per_attempt=max_workflows_per_attempt,
            max_steps_per_workflow=max_steps_per_workflow,
        )

        if rendered.text.strip() == "<none>":
            task_missing_workflow += 1
            instruction_file.write_text(base_text + "\n", encoding="utf-8")
            continue

        block = (
            "\n\n"
            f"{MARKER_BEGIN}\n\n"
            f"{intro_text.strip()}\n\n"
            f"{rendered.text.strip()}\n\n"
            f"{MARKER_END}\n"
        )
        instruction_file.write_text(base_text + block, encoding="utf-8")
        task_injected += 1

    return {
        "mode": "workflow",
        "workflow_memory_path": str(workflow_memory_path),
        "workflow_attempt_filter": workflow_attempt_filter,
        "max_attempts": max_attempts,
        "max_workflows_per_attempt": max_workflows_per_attempt,
        "max_steps_per_workflow": max_steps_per_workflow,
        "task_total": task_total,
        "task_injected": task_injected,
        "task_missing_workflow": task_missing_workflow,
    }


def _slug(text: str) -> str:
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


def _list_skill_entries(skill_repository: Path) -> list[SkillEntry]:
    if not skill_repository.is_dir():
        raise FileNotFoundError(f"skills repository not found: {skill_repository}")

    entries: list[SkillEntry] = []
    skill_dirs: list[Path] = []
    for skill_md in sorted(skill_repository.rglob("SKILL.md")):
        child = skill_md.parent
        parts = child.relative_to(skill_repository).parts
        if any(part.startswith(".") for part in parts):
            continue
        skill_dirs.append(child)

    for child in skill_dirs:
        body = (child / "SKILL.md").read_text(encoding="utf-8")
        relative = child.relative_to(skill_repository).as_posix()
        skill_id = relative.replace("/", "__")
        blob = _slug(relative + "\n" + body)
        entries.append(SkillEntry(skill_id=skill_id, path=child, search_blob=blob))
    return entries


def _task_matched_skills(task_name: str, entries: list[SkillEntry]) -> list[SkillEntry]:
    task_key = normalize_task_key(task_name)
    if not task_key:
        return []
    task_slug = _slug(task_key)
    task_parts = [part for part in task_slug.split("-") if len(part) >= 4]

    matched: list[SkillEntry] = []
    for entry in entries:
        blob = entry.search_blob
        if task_slug and task_slug in blob:
            matched.append(entry)
            continue
        if task_parts and all(part in blob for part in task_parts):
            matched.append(entry)
            continue
    return matched


def apply_skill_injection(
    *,
    tasks_dir: Path,
    skill_repository: Path,
    injection_scope: str,
    skills_prefix: str,
) -> dict:
    entries = _list_skill_entries(skill_repository)
    task_total = 0
    task_with_injected_skills = 0
    total_injected_skill_dirs = 0

    for task_dir in sorted([p for p in tasks_dir.iterdir() if p.is_dir()]):
        env_skill_root = task_dir / "environment" / "skills"
        env_skill_root.mkdir(parents=True, exist_ok=True)
        task_total += 1

        selected = entries if injection_scope == "all" else _task_matched_skills(task_dir.name, entries)
        if not selected:
            continue

        injected_here = 0
        for entry in selected:
            dst_name = f"{skills_prefix}{entry.skill_id}" if skills_prefix else entry.skill_id
            dst = env_skill_root / dst_name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(entry.path, dst)
            injected_here += 1

        if injected_here > 0:
            task_with_injected_skills += 1
            total_injected_skill_dirs += injected_here

    return {
        "mode": "skills",
        "skills_repository": str(skill_repository),
        "skills_injection_scope": injection_scope,
        "skills_prefix": skills_prefix,
        "skill_entry_count": len(entries),
        "task_total": task_total,
        "task_with_injected_skills": task_with_injected_skills,
        "total_injected_skill_dirs": total_injected_skill_dirs,
    }


def _build_harbor_launcher_command(
    *,
    experiment_id: str,
    model: str,
    dataset: str,
    prepared_tasks_dir: Path,
    n_attempts: int,
    n_concurrent: int,
    task_names: list[str],
    exclude_task_names: list[str],
    n_tasks: int | None,
    base_url: str | None,
    working_dir: str | None,
    dry_run: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/server/run_skillsbench_harbor_experiment.py",
        "--experiment-id",
        experiment_id,
        "--model",
        model,
        "--dataset",
        dataset,
        "--source-mode",
        "path",
        "--skillsbench-path",
        str(prepared_tasks_dir),
        "--memory-setting",
        "none",
        "--native-agent",
        "codex",
        "--n-attempts",
        str(n_attempts),
        "--n-concurrent",
        str(n_concurrent),
    ]
    for task in task_names:
        cmd.extend(["--task-name", task])
    for task in exclude_task_names:
        cmd.extend(["--exclude-task-name", task])
    if n_tasks is not None:
        cmd.extend(["--n-tasks", str(n_tasks)])
    if base_url:
        cmd.extend(["--base-url", base_url])
    if working_dir:
        cmd.extend(["--working-dir", working_dir])
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def _render(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    requested_experiment_id = args.experiment_id
    normalized_experiment_id = normalize_experiment_name(requested_experiment_id, max_length=56)

    run_dir = (_resolve_path(project_root, args.output_root) or project_root / "experiments/skillsbench") / normalized_experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source_tasks_dir = _resolve_path(project_root, args.skillsbench_tasks_path)
    if source_tasks_dir is None:
        raise ValueError("skillsbench tasks path cannot be None")
    prepared_tasks_dir = run_dir / "prepared-tasks"
    _copy_tasks(source_tasks_dir, prepared_tasks_dir)

    mode_payload: dict
    if args.mode == "no-skills":
        mode_payload = {
            "mode": "no-skills",
            "task_total": len([p for p in prepared_tasks_dir.iterdir() if p.is_dir()]),
        }
    elif args.mode == "workflow":
        workflow_memory_path = _resolve_path(project_root, args.workflow_memory_path)
        if workflow_memory_path is None:
            raise ValueError("workflow mode requires --workflow-memory-path")
        mode_payload = apply_workflow_context(
            tasks_dir=prepared_tasks_dir,
            workflow_memory_path=workflow_memory_path,
            workflow_attempt_filter=args.workflow_attempt_filter,
            intro_text=args.workflow_intro_text,
            max_attempts=max(0, int(args.workflow_max_attempts)),
            max_workflows_per_attempt=max(0, int(args.workflow_max_workflows_per_attempt)),
            max_steps_per_workflow=max(0, int(args.workflow_max_steps_per_workflow)),
        )
    else:
        skills_repository = _resolve_path(project_root, args.skills_repository)
        if skills_repository is None:
            raise ValueError("skills mode requires --skills-repository")
        mode_payload = apply_skill_injection(
            tasks_dir=prepared_tasks_dir,
            skill_repository=skills_repository,
            injection_scope=args.skills_injection_scope,
            skills_prefix=args.skills_prefix,
        )

    launch_cmd = _build_harbor_launcher_command(
        experiment_id=normalized_experiment_id,
        model=args.model,
        dataset=args.dataset,
        prepared_tasks_dir=prepared_tasks_dir,
        n_attempts=max(1, int(args.n_attempts)),
        n_concurrent=max(1, int(args.n_concurrent)),
        task_names=args.task_name,
        exclude_task_names=args.exclude_task_name,
        n_tasks=args.n_tasks,
        base_url=args.base_url,
        working_dir=args.working_dir,
        dry_run=args.dry_run,
    )

    manifest = {
        "experiment_id": normalized_experiment_id,
        "requested_experiment_id": requested_experiment_id,
        "mode": args.mode,
        "model": args.model,
        "dataset": args.dataset,
        "native_agent": "codex",
        "source_tasks_dir": str(source_tasks_dir),
        "prepared_tasks_dir": str(prepared_tasks_dir),
        "mode_payload": mode_payload,
        "harbor_launch_command": _render(launch_cmd),
    }
    (run_dir / "mode_injection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    if args.dry_run:
        return 0

    subprocess.run(launch_cmd, check=True, cwd=project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
