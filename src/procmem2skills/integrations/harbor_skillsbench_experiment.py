from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from procmem2skills.integrations.harbor_terminal_experiment import (
    discover_job_dir,
    dump_manifest,
    ensure_job_dir_alias,
    ensure_openrouter_env,
    import_harbor_job,
    normalize_experiment_name,
    render_command,
    resolve_harbor_binary,
    run_harbor_job,
)

DEFAULT_AGENT_IMPORT_PATH = "procmem2skills.integrations.harbor_terminal_agent:SkillAwareTerminalAgent"
_MEMORY_SETTING_ALIASES = {
    "none": "none",
    "off": "none",
    "baseline": "none",
    "skills": "skills",
    "skill": "skills",
    "skill-aware": "skills",
    "skill-injection": "skills",
    "workflow": "workflows",
    "workflows": "workflows",
    "workflow-context": "workflows",
    "workflow-injection": "workflows",
}


def _split_dataset_spec(dataset: str) -> tuple[str, str | None]:
    text = str(dataset or "").strip().lower()
    if not text:
        return "", None
    if "@" not in text:
        return text, None
    name, version = text.split("@", 1)
    return name.strip(), (version.strip() or None)


def _dataset_exists(harbor_bin: Path, dataset_spec: str) -> bool:
    dataset_name, dataset_version = _split_dataset_spec(dataset_spec)
    if not dataset_name:
        return False
    try:
        completed = subprocess.run(
            [str(harbor_bin), "datasets", "list"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False

    for raw in completed.stdout.splitlines():
        if "│" not in raw:
            continue
        parts = [part.strip().lower() for part in raw.split("│")]
        if len(parts) < 3:
            continue
        row_name = parts[1]
        row_version = parts[2] or None
        if row_name != dataset_name:
            continue
        if dataset_version is None:
            return True
        if row_version == dataset_version:
            return True
    return False


def _discover_skillsbench_path(project_root: Path, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        resolved = explicit_path.resolve()
        return resolved if resolved.exists() else None

    candidates = [
        project_root / "benchmarks" / "skillsbench" / "tasks",
        project_root / "benchmarks" / "skillsbench",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _build_job_name(*, experiment_id: str, model: str, phase: str = "live") -> str:
    experiment_slug = normalize_experiment_name(experiment_id, max_length=28)
    model_slug = normalize_experiment_name(model.split("/")[-1], max_length=20)
    return normalize_experiment_name(f"sb-{phase}-{experiment_slug}-{model_slug}", max_length=72)


def _resolve_memory_setting(*, requested: str | None, legacy_agent_mode: str | None) -> str:
    if requested:
        normalized = _MEMORY_SETTING_ALIASES.get(str(requested).strip().lower())
        if normalized is None:
            supported = ", ".join(sorted(set(_MEMORY_SETTING_ALIASES.keys())))
            raise ValueError(f"unsupported --memory-setting={requested}. supported: {supported}")
        return normalized
    mode = (legacy_agent_mode or "native").strip().lower()
    return "skills" if mode == "skill-aware" else "none"


def _build_harbor_command(
    *,
    harbor_bin: Path,
    jobs_dir: Path,
    job_name: str,
    source_mode: str,
    dataset: str,
    path: Path | None,
    model: str,
    memory_setting: str,
    native_agent: str,
    agent_import_path: str,
    skill_repository: Path | None,
    top_k_skills: int,
    skill_selection_mode: str,
    skill_candidate_pool: int,
    workflow_memory_path: Path | None,
    workflow_max_attempts: int | None,
    workflow_max_workflows_per_attempt: int | None,
    workflow_max_steps_per_workflow: int | None,
    n_concurrent: int,
    n_attempts: int,
    task_names: list[str],
    exclude_task_names: list[str],
    n_tasks: int | None,
    environment_type: str,
    max_steps: int | None,
    command_timeout_sec: int | None,
    base_url: str | None,
    working_dir: str | None,
    passthrough_args: list[str],
) -> list[str]:
    command = [
        str(harbor_bin),
        "run",
        "--job-name",
        job_name,
        "--jobs-dir",
        str(jobs_dir),
        "--model",
        model,
        "--env",
        environment_type,
        "--n-concurrent",
        str(n_concurrent),
        "--n-attempts",
        str(n_attempts),
    ]
    if source_mode == "dataset":
        command.extend(["--dataset", dataset])
    else:
        if path is None:
            raise ValueError("source_mode=path requires a resolved --skillsbench-path")
        command.extend(["--path", str(path)])

    if memory_setting == "none":
        command.extend(["--agent", native_agent])
    else:
        command.extend(["--agent-import-path", agent_import_path])
        command.extend(["--ak", f"memory_setting={memory_setting}"])
        if memory_setting == "skills":
            if skill_repository is None:
                raise ValueError("memory_setting=skills requires skill_repository")
            command.extend(["--ak", f"skill_repository={skill_repository}"])
            command.extend(["--ak", f"top_k_skills={top_k_skills}"])
            command.extend(["--ak", f"skill_selection_mode={skill_selection_mode}"])
            command.extend(["--ak", f"skill_candidate_pool={skill_candidate_pool}"])
        elif memory_setting == "workflows":
            if workflow_memory_path is None:
                raise ValueError("memory_setting=workflows requires workflow_memory_path")
            command.extend(["--ak", f"workflow_memory_path={workflow_memory_path}"])
            if workflow_max_attempts is not None and workflow_max_attempts > 0:
                command.extend(["--ak", f"workflow_max_attempts={workflow_max_attempts}"])
            if workflow_max_workflows_per_attempt is not None and workflow_max_workflows_per_attempt > 0:
                command.extend(["--ak", f"workflow_max_workflows_per_attempt={workflow_max_workflows_per_attempt}"])
            if workflow_max_steps_per_workflow is not None and workflow_max_steps_per_workflow > 0:
                command.extend(["--ak", f"workflow_max_steps_per_workflow={workflow_max_steps_per_workflow}"])
        else:
            raise ValueError(f"unsupported memory_setting: {memory_setting}")
        if max_steps is not None and max_steps > 0:
            command.extend(["--ak", f"max_steps={max_steps}"])
        if command_timeout_sec is not None and command_timeout_sec > 0:
            command.extend(["--ak", f"command_timeout_sec={command_timeout_sec}"])

    for task_name in task_names:
        command.extend(["--task-name", task_name])
    for task_name in exclude_task_names:
        command.extend(["--exclude-task-name", task_name])
    if n_tasks is not None:
        command.extend(["--n-tasks", str(n_tasks)])

    if base_url:
        if memory_setting == "none":
            command.extend(["--ae", f"OPENAI_BASE_URL={base_url}"])
            command.extend(["--ae", f"OPENROUTER_BASE_URL={base_url}"])
        else:
            command.extend(["--ak", f"base_url={base_url}"])
    if working_dir and memory_setting != "none":
        command.extend(["--ak", f"working_dir={working_dir}"])

    if passthrough_args:
        command.extend(passthrough_args)
    return command


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SkillsBench via Harbor with registry-first fallback to local path.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="skillsbench", help="Harbor dataset name@version when source-mode uses registry.")
    parser.add_argument("--source-mode", choices=["auto", "dataset", "path"], default="auto")
    parser.add_argument("--skillsbench-path", type=Path, help="Local skillsbench task/dataset directory (e.g., benchmarks/skillsbench/tasks).")

    parser.add_argument("--import-benchmark", default="terminal-bench")
    parser.add_argument("--harness", default="skills-bench/harness")
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))

    parser.add_argument(
        "--memory-setting",
        default=None,
        help=(
            "Unified memory setting for comparison: none | skills | workflows "
            "(aliases: skill-aware, workflow-context). "
            "When omitted, legacy --agent-mode is used for backward compatibility."
        ),
    )
    parser.add_argument("--agent-mode", choices=["skill-aware", "native"], default="native")
    parser.add_argument("--native-agent", default="codex")
    parser.add_argument("--agent-import-path", default=DEFAULT_AGENT_IMPORT_PATH)
    parser.add_argument("--skill-repository", type=Path)
    parser.add_argument("--top-k-skills", type=int, default=3)
    parser.add_argument("--skill-selection-mode", choices=["agent-first", "vector"], default="agent-first")
    parser.add_argument("--skill-candidate-pool", type=int, default=12)
    parser.add_argument("--workflow-memory-path", type=Path, help="Grouped workflow memory JSON (task -> attempts -> workflows).")
    parser.add_argument("--workflow-max-attempts", type=int, default=0, help="Optional cap per task when injecting workflows; <=0 means all.")
    parser.add_argument(
        "--workflow-max-workflows-per-attempt",
        type=int,
        default=0,
        help="Optional cap per attempt when injecting workflows; <=0 means all.",
    )
    parser.add_argument(
        "--workflow-max-steps-per-workflow",
        type=int,
        default=0,
        help="Optional cap of rendered steps per workflow; <=0 means all.",
    )

    parser.add_argument("--task-name", action="append", default=[])
    parser.add_argument("--exclude-task-name", action="append", default=[])
    parser.add_argument("--n-tasks", type=int)
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--n-concurrent", type=int, default=8)
    parser.add_argument("--environment-type", default="docker")

    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--command-timeout-sec", type=int, default=180)
    parser.add_argument("--base-url")
    parser.add_argument("--working-dir")

    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args, passthrough_args = parser.parse_known_args(argv)

    project_root = Path(__file__).resolve().parents[3]
    normalized_experiment_id = normalize_experiment_name(args.experiment_id, max_length=56)
    dataset_slug = normalize_experiment_name(args.dataset or "skillsbench", max_length=48)
    run_dir = (project_root / args.output_root / dataset_slug / normalized_experiment_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    imported_dir = run_dir / "imported"
    imported_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir = run_dir / "harbor-jobs"

    harbor_bin = resolve_harbor_binary(project_root)
    local_path = _discover_skillsbench_path(project_root, args.skillsbench_path)
    dataset_available = _dataset_exists(harbor_bin, args.dataset)

    source_mode = args.source_mode
    if source_mode == "auto":
        if dataset_available:
            source_mode = "dataset"
        elif local_path is not None:
            source_mode = "path"
        else:
            source_mode = "dataset"
    if source_mode == "path" and local_path is None:
        raise FileNotFoundError(
            "skillsbench local path not found. Pass --skillsbench-path or clone to benchmarks/skillsbench/."
        )

    harbor_job_name = _build_job_name(experiment_id=normalized_experiment_id, model=args.model, phase="live")
    memory_setting = _resolve_memory_setting(requested=args.memory_setting, legacy_agent_mode=args.agent_mode)

    skill_repository: Path | None = None
    if memory_setting == "skills":
        if args.skill_repository is None:
            raise ValueError("memory-setting=skills requires --skill-repository")
        skill_repository = args.skill_repository.resolve()
        skill_repository.mkdir(parents=True, exist_ok=True)

    workflow_memory_path: Path | None = None
    if memory_setting == "workflows":
        if args.workflow_memory_path is None:
            raise ValueError("memory-setting=workflows requires --workflow-memory-path")
        workflow_memory_path = args.workflow_memory_path.resolve()
        if not workflow_memory_path.is_file():
            raise FileNotFoundError(f"workflow memory file not found: {workflow_memory_path}")

    command = _build_harbor_command(
        harbor_bin=harbor_bin,
        jobs_dir=jobs_dir,
        job_name=harbor_job_name,
        source_mode=source_mode,
        dataset=args.dataset,
        path=local_path,
        model=args.model,
        memory_setting=memory_setting,
        native_agent=args.native_agent,
        agent_import_path=args.agent_import_path,
        skill_repository=skill_repository,
        top_k_skills=args.top_k_skills,
        skill_selection_mode=args.skill_selection_mode,
        skill_candidate_pool=args.skill_candidate_pool,
        workflow_memory_path=workflow_memory_path,
        workflow_max_attempts=args.workflow_max_attempts,
        workflow_max_workflows_per_attempt=args.workflow_max_workflows_per_attempt,
        workflow_max_steps_per_workflow=args.workflow_max_steps_per_workflow,
        n_concurrent=args.n_concurrent,
        n_attempts=args.n_attempts,
        task_names=args.task_name,
        exclude_task_names=args.exclude_task_name,
        n_tasks=args.n_tasks,
        environment_type=args.environment_type,
        max_steps=args.max_steps,
        command_timeout_sec=args.command_timeout_sec,
        base_url=args.base_url,
        working_dir=args.working_dir,
        passthrough_args=passthrough_args,
    )

    manifest_path = run_dir / "harbor-manifest.json"
    runbook_path = run_dir / "HARBOR_RUNBOOK.md"
    imported_path = imported_dir / "live-trajectories.jsonl"
    manifest = {
        "experiment_id": normalized_experiment_id,
        "requested_experiment_id": args.experiment_id,
        "dataset": args.dataset,
        "source_mode": source_mode,
        "dataset_available": dataset_available,
        "skillsbench_path": str(local_path) if local_path else None,
        "harness": args.harness,
        "import_benchmark": args.import_benchmark,
        "model": args.model,
        "memory_setting": memory_setting,
        "legacy_agent_mode": args.agent_mode,
        "native_agent": args.native_agent if memory_setting == "none" else None,
        "agent_import_path": args.agent_import_path if memory_setting != "none" else None,
        "n_attempts": args.n_attempts,
        "n_concurrent": args.n_concurrent,
        "task_names": args.task_name,
        "exclude_task_names": args.exclude_task_name,
        "n_tasks": args.n_tasks,
        "skill_repository": str(skill_repository) if skill_repository is not None else None,
        "workflow_memory_path": str(workflow_memory_path) if workflow_memory_path is not None else None,
        "workflow_max_attempts": args.workflow_max_attempts,
        "workflow_max_workflows_per_attempt": args.workflow_max_workflows_per_attempt,
        "workflow_max_steps_per_workflow": args.workflow_max_steps_per_workflow,
        "harbor_job_name": harbor_job_name,
        "harbor_command": render_command(command),
        "jobs_dir": str(jobs_dir),
        "imported_path": str(imported_path),
        "manifest_path": str(manifest_path),
        "runbook_path": str(runbook_path),
        "harbor_passthrough_args": passthrough_args,
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    ensure_openrouter_env()
    run_harbor_job(command=command, project_root=project_root)
    job_dir = discover_job_dir(jobs_dir, harbor_job_name)
    alias_path = ensure_job_dir_alias(jobs_dir=jobs_dir, alias_name=harbor_job_name, actual_job_dir=job_dir)
    imported_count = import_harbor_job(
        job_dir,
        imported_path,
        dataset=args.dataset,
        import_benchmark=args.import_benchmark,
        harness=args.harness,
    )

    manifest["job_dir"] = str(job_dir)
    manifest["job_dir_alias"] = str(alias_path) if alias_path is not None else None
    manifest["imported_trajectory_count"] = imported_count
    dump_manifest(manifest_path, manifest)
    runbook_path.write_text(
        "\n".join(
            [
                "# SkillsBench Harbor Experiment",
                "",
                f"- Experiment ID: {normalized_experiment_id}",
                f"- Dataset: {args.dataset}",
                f"- Source Mode: {source_mode}",
                f"- SkillsBench Path: {local_path if local_path else '-'}",
                f"- Model: {args.model}",
                f"- Memory Setting: {memory_setting}",
                f"- Skill Repository: {skill_repository if skill_repository else '-'}",
                f"- Workflow Memory Path: {workflow_memory_path if workflow_memory_path else '-'}",
                f"- Harbor Job Name: {harbor_job_name}",
                "",
                "## Harbor Command",
                "",
                "```bash",
                render_command(command),
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
