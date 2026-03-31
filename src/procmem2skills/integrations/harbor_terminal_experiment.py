from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from procmem2skills.analysis.failure import build_failure_analysis_from_trajectories
from procmem2skills.evaluation.pipeline import SkillDistillationPipeline
from procmem2skills.importers import import_alfworld, import_mind2web, import_terminal_bench, import_webarena
from procmem2skills.models import Trajectory
from procmem2skills.packager.materialize import materialize_skill_repository_standard_llm
from procmem2skills.packager.skill_writer import SkillWriter
from procmem2skills.recorder.jsonl import load_trajectories, write_trajectories


def normalize_experiment_name(value: str, *, max_length: int = 56) -> str:
    tokens = re.findall(r"[a-z0-9]+", (value or "").strip().lower())
    if not tokens:
        return "experiment"
    slug = "-".join(tokens)
    if len(slug) <= max_length:
        return slug
    trimmed = slug[:max_length].rstrip("-")
    return trimmed or "experiment"


def build_harbor_job_name(
    *,
    experiment_id: str,
    dataset: str,
    model: str,
    phase: str | None = None,
    max_length: int = 72,
) -> str:
    benchmark = _benchmark_slug(dataset)
    phase_slug = normalize_experiment_name(phase or "run", max_length=16)
    experiment_slug = normalize_experiment_name(experiment_id, max_length=28)
    model_slug = normalize_experiment_name(model.split("/")[-1], max_length=20)
    name = f"{benchmark}-{phase_slug}-{experiment_slug}-{model_slug}"
    return normalize_experiment_name(name, max_length=max_length)


def ensure_job_dir_alias(*, jobs_dir: Path, alias_name: str, actual_job_dir: Path) -> Path | None:
    alias = jobs_dir / normalize_experiment_name(alias_name, max_length=72)
    actual = actual_job_dir.resolve()
    if alias.resolve() == actual if alias.exists() else False:
        return alias
    try:
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        alias.symlink_to(actual)
        return alias
    except Exception:
        return None


def _benchmark_slug(dataset: str) -> str:
    name, _ = split_dataset_spec(dataset)
    if name == "terminal-bench":
        return "tb"
    if name == "alfworld":
        return "alf"
    if name == "webarena":
        return "wa"
    return normalize_experiment_name(name or "bench", max_length=8)


def split_dataset_spec(dataset: str) -> tuple[str, str | None]:
    text = str(dataset or "").strip().lower()
    if not text:
        raise ValueError("dataset spec cannot be empty")
    if "@" not in text:
        return text, None
    name, version = text.split("@", 1)
    normalized_name = name.strip()
    normalized_version = version.strip() or None
    if not normalized_name:
        raise ValueError(f"invalid dataset spec: {dataset}")
    return normalized_name, normalized_version


def dataset_storage_slug(dataset: str) -> str:
    name, version = split_dataset_spec(dataset)
    if version:
        return normalize_experiment_name(f"{name}-{version}", max_length=48)
    return normalize_experiment_name(name, max_length=48)


def resolve_import_benchmark(*, dataset: str, import_benchmark: str) -> str:
    requested = str(import_benchmark or "auto").strip().lower()
    if requested and requested != "auto":
        return requested
    dataset_name, _ = split_dataset_spec(dataset)
    if dataset_name in {"terminal-bench", "terminal-bench-sample"}:
        return "terminal-bench"
    if dataset_name in {"mind2web", "webarena", "alfworld"}:
        return dataset_name
    raise ValueError(
        f"unsupported Harbor dataset for auto importer: {dataset}. "
        "Pass --import-benchmark to select a concrete importer."
    )


def resolve_benchmark_python(project_root: Path) -> Path:
    override = os.environ.get("PROCMEM_BENCHMARK_PYTHON")
    if override:
        return Path(override)
    py312 = project_root / ".venv-py312" / "bin" / "python"
    if py312.exists():
        return py312
    return project_root / ".venv" / "bin" / "python"


def resolve_harbor_binary(project_root: Path) -> Path:
    python_bin = resolve_benchmark_python(project_root)
    return python_bin.with_name("harbor")


def _import_by_benchmark(*, benchmark: str, path: Path, harness: str = "") -> list[Trajectory]:
    normalized = benchmark.strip().lower()
    if normalized == "terminal-bench":
        return import_terminal_bench(path, harness=harness or "terminal-bench/harness")
    if normalized == "mind2web":
        return import_mind2web(path)
    if normalized == "webarena":
        return import_webarena(path)
    if normalized == "alfworld":
        return import_alfworld(path)
    raise ValueError(f"unsupported import benchmark: {benchmark}")


def load_bootstrap_trajectories(path: Path, *, import_benchmark: str, harness: str) -> list:
    if path.suffix == ".jsonl":
        try:
            trajectories = load_trajectories(path)
        except Exception:
            trajectories = []
        if trajectories:
            return trajectories
    return _import_by_benchmark(benchmark=import_benchmark, path=path, harness=harness)


def materialize_bootstrap_skills(
    *,
    bootstrap_input: Path,
    bootstrap_skill_dir: Path,
    import_benchmark: str,
    harness: str,
    min_support: int,
    similarity_threshold: float,
    structure_threshold: float,
    cluster_backend: str,
    cluster_embedding_model: str | None,
    cluster_embedding_base_url: str | None,
    cluster_dbscan_eps: float,
    cluster_dbscan_min_samples: int,
    cluster_embedding_strict: bool,
    workflow_aggregation_mode: str,
    per_task_skill_namespace: bool,
    skill_creator_model: str | None,
    skill_creator_base_url: str | None,
    skill_creator_agent_style: str,
    skill_creator_system_prompt: str | None,
) -> tuple[int, int, dict]:
    trajectories = load_bootstrap_trajectories(
        bootstrap_input,
        import_benchmark=import_benchmark,
        harness=harness,
    )
    failure_analysis = build_failure_analysis_from_trajectories(trajectories)
    result = SkillDistillationPipeline(
        min_support=min_support,
        similarity_threshold=similarity_threshold,
        structure_threshold=structure_threshold,
        cluster_backend=cluster_backend,
        cluster_embedding_model=cluster_embedding_model,
        cluster_embedding_base_url=cluster_embedding_base_url,
        cluster_dbscan_eps=cluster_dbscan_eps,
        cluster_dbscan_min_samples=cluster_dbscan_min_samples,
        cluster_embedding_strict=cluster_embedding_strict,
        workflow_aggregation_mode=workflow_aggregation_mode,
        per_task_skill_namespace=per_task_skill_namespace,
    ).distill(trajectories)
    _, generation_meta = materialize_skill_repository_standard_llm(
        skills=result.skills,
        output_dir=bootstrap_skill_dir,
        writer=SkillWriter(),
        model=skill_creator_model,
        base_url=skill_creator_base_url,
        skill_creator_agent_style=skill_creator_agent_style,
        skill_creator_system_prompt=skill_creator_system_prompt,
        failure_analysis=failure_analysis.get("global"),
        failure_analysis_by_task=failure_analysis.get("by_task"),
    )
    return len(result.trajectories), len(result.skills), generation_meta


def build_harbor_run_command(
    *,
    harbor_bin: Path,
    jobs_dir: Path,
    job_name: str,
    dataset: str,
    model: str,
    agent_mode: str,
    native_agent: str | None,
    agent_import_path: str,
    skill_repository: Path,
    top_k_skills: int,
    skill_selection_mode: str,
    skill_candidate_pool: int,
    agent_kwargs: dict[str, str] | None,
    max_steps: int | None,
    command_timeout_sec: int | None,
    environment_type: str,
    n_concurrent: int,
    task_names: list[str] | None,
    n_tasks: int | None,
    n_attempts: int,
    base_url: str | None,
    working_dir: str | None,
    extra_args: list[str] | None = None,
) -> list[str]:
    command = [
        str(harbor_bin),
        "run",
        "--job-name",
        job_name,
        "--jobs-dir",
        str(jobs_dir),
        "--dataset",
        dataset,
        "--model",
        model,
        "--env",
        environment_type,
        "--n-concurrent",
        str(n_concurrent),
        "--n-attempts",
        str(n_attempts),
    ]
    mode = (agent_mode or "skill-aware").strip().lower()
    if mode == "native":
        if not native_agent:
            raise ValueError("native agent mode requires --native-agent")
        command.extend(["--agent", native_agent])
    else:
        command.extend(["--agent-import-path", agent_import_path])
        command.extend(["--ak", f"skill_repository={skill_repository}"])
        command.extend(["--ak", f"top_k_skills={top_k_skills}"])
        command.extend(["--ak", f"skill_selection_mode={skill_selection_mode}"])
        command.extend(["--ak", f"skill_candidate_pool={skill_candidate_pool}"])
    for key in sorted((agent_kwargs or {}).keys()):
        value = agent_kwargs[key]
        command.extend(["--ak", f"{key}={value}"])
    if mode != "native" and max_steps is not None:
        command.extend(["--ak", f"max_steps={max_steps}"])
    if mode != "native" and command_timeout_sec is not None:
        command.extend(["--ak", f"command_timeout_sec={command_timeout_sec}"])
    for task_name in task_names or []:
        command.extend(["--task-name", task_name])
    if n_tasks is not None:
        command.extend(["--n-tasks", str(n_tasks)])
    if base_url:
        if mode == "native":
            command.extend(["--ae", f"OPENAI_BASE_URL={base_url}"])
            command.extend(["--ae", f"OPENROUTER_BASE_URL={base_url}"])
        else:
            command.extend(["--ak", f"base_url={base_url}"])
    if working_dir and mode != "native":
        command.extend(["--ak", f"working_dir={working_dir}"])
    if extra_args:
        command.extend(extra_args)
    return command


def render_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def discover_job_dir(jobs_dir: Path, job_name: str) -> Path:
    direct = jobs_dir / job_name
    if direct.exists():
        return direct
    candidates = sorted((path for path in jobs_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no Harbor job directory found under {jobs_dir}")
    return candidates[0]


def import_harbor_job(
    job_dir: Path,
    output_path: Path,
    *,
    dataset: str,
    import_benchmark: str,
    harness: str,
) -> int:
    benchmark = resolve_import_benchmark(dataset=dataset, import_benchmark=import_benchmark)
    trajectories = _import_by_benchmark(benchmark=benchmark, path=job_dir, harness=harness)
    write_trajectories(output_path, trajectories)
    return len(trajectories)


def run_harbor_job(
    *,
    command: list[str],
    project_root: Path,
    jobs_dir: Path | None = None,
    job_name: str | None = None,
    show_progress: bool = True,
    progress_interval_sec: int = 20,
    expected_trials: int | None = None,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src") + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    if not show_progress:
        subprocess.run(command, check=True, env=env)
        return

    process = subprocess.Popen(command, env=env)
    interval = max(2, int(progress_interval_sec))
    start = time.monotonic()

    while True:
        try:
            return_code = process.wait(timeout=interval)
            break
        except subprocess.TimeoutExpired:
            snapshot = collect_harbor_progress_snapshot(jobs_dir=jobs_dir, job_name=job_name)
            if snapshot is None:
                continue
            line = format_harbor_progress_line(
                snapshot=snapshot,
                elapsed_sec=time.monotonic() - start,
                expected_trials=expected_trials,
            )
            print(line, file=sys.stderr, flush=True)

    final_snapshot = collect_harbor_progress_snapshot(jobs_dir=jobs_dir, job_name=job_name)
    if final_snapshot is not None:
        line = format_harbor_progress_line(
            snapshot=final_snapshot,
            elapsed_sec=time.monotonic() - start,
            expected_trials=expected_trials,
        )
        print(f"{line} | finished", file=sys.stderr, flush=True)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def collect_harbor_progress_snapshot(*, jobs_dir: Path | None, job_name: str | None) -> dict | None:
    if jobs_dir is None:
        return None
    if not jobs_dir.exists():
        return None
    job_dir = _resolve_progress_job_dir(jobs_dir=jobs_dir, job_name=job_name)
    if job_dir is None:
        return None

    completed = 0
    success = 0
    failure = 0
    trajectory_count = 0
    result_files = sorted(job_dir.glob("*/result.json"))
    for result_file in result_files:
        completed += 1
        reward = _extract_reward_from_result(result_file)
        if reward is None:
            continue
        if reward >= 1.0:
            success += 1
        else:
            failure += 1
    trajectory_count = sum(1 for _ in job_dir.glob("*/agent/trajectory.json"))
    return {
        "job_name": job_dir.name,
        "job_dir": str(job_dir),
        "completed": completed,
        "success": success,
        "failure": failure,
        "trajectory_count": trajectory_count,
    }


def format_harbor_progress_line(*, snapshot: dict, elapsed_sec: float, expected_trials: int | None) -> str:
    elapsed = _format_seconds(max(0.0, float(elapsed_sec)))
    completed = int(snapshot.get("completed", 0))
    success = int(snapshot.get("success", 0))
    failure = int(snapshot.get("failure", 0))
    trajectory_count = int(snapshot.get("trajectory_count", 0))
    job_name = str(snapshot.get("job_name", "harbor-job"))

    parts = [f"[progress] {job_name}", f"elapsed {elapsed}"]
    if expected_trials is not None and expected_trials > 0:
        percent = min(100.0, max(0.0, 100.0 * completed / expected_trials))
        parts.append(f"completed {completed}/{expected_trials} ({percent:.1f}%)")
        if completed > 0 and completed < expected_trials:
            eta_sec = (elapsed_sec / completed) * (expected_trials - completed)
            parts.append(f"eta {_format_seconds(max(0.0, eta_sec))}")
    else:
        parts.append(f"completed {completed}")
    parts.append(f"success {success}")
    parts.append(f"failure {failure}")
    parts.append(f"traj {trajectory_count}")
    return " | ".join(parts)


def estimate_expected_trials(*, n_tasks: int | None, task_names: list[str] | None, n_attempts: int) -> int | None:
    attempts = max(1, int(n_attempts))
    if n_tasks is not None and n_tasks > 0:
        return n_tasks * attempts
    if task_names:
        return len(task_names) * attempts
    return None


def _resolve_progress_job_dir(*, jobs_dir: Path, job_name: str | None) -> Path | None:
    candidate: Path | None = None
    if job_name:
        direct = jobs_dir / job_name
        if direct.exists() and direct.is_dir():
            candidate = direct
        else:
            normalized = normalize_experiment_name(job_name, max_length=72)
            alt = jobs_dir / normalized
            if alt.exists() and alt.is_dir():
                candidate = alt
    if candidate is not None:
        return candidate
    candidates = sorted((path for path in jobs_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    return candidates[0]


def _extract_reward_from_result(path: Path) -> float | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    candidates = [
        payload.get("reward"),
        (payload.get("verifier_result") or {}).get("rewards", {}).get("reward"),
        (payload.get("agent_result") or {}).get("score"),
    ]
    for value in candidates:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _format_seconds(raw_seconds: float) -> str:
    seconds = int(round(raw_seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remain = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remain:02d}"


def dump_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_openrouter_env() -> None:
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openrouter_key and not openai_key:
        os.environ["OPENAI_API_KEY"] = openrouter_key
        openai_key = openrouter_key
    if openrouter_key and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    if openrouter_key or openai_key:
        return
    raise RuntimeError(
        "OPENROUTER_API_KEY or OPENAI_API_KEY must be set before running Harbor live experiments"
    )


def _optional_positive(value: int | None) -> int | None:
    if value is None:
        return None
    return value if value > 0 else None


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Bootstrap skills and launch a Harbor job with a skill-aware agent.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--dataset", default="terminal-bench@2.0")
    parser.add_argument(
        "--import-benchmark",
        default="auto",
        help="Importer benchmark for bootstrap/job import: auto|terminal-bench|mind2web|webarena|alfworld",
    )
    parser.add_argument("--harness", default="", help="Optional harness override used for imported trajectories.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--agent-mode", choices=["skill-aware", "native"], default="skill-aware")
    parser.add_argument("--native-agent", default="terminus-2")
    parser.add_argument(
        "--agent-import-path",
        default="procmem2skills.integrations.harbor_terminal_agent:SkillAwareTerminalAgent",
    )
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--bootstrap-input", type=Path)
    parser.add_argument("--skill-repository", type=Path)
    parser.add_argument("--task-name", action="append", default=[])
    parser.add_argument("--n-tasks", type=int)
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--n-concurrent", type=int, default=8)
    parser.add_argument("--environment-type", default="docker")
    parser.add_argument("--top-k-skills", type=int, default=3)
    parser.add_argument("--skill-selection-mode", choices=["agent-first", "vector"], default="agent-first")
    parser.add_argument("--skill-candidate-pool", type=int, default=12)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Per-episode step cap; set <=0 to disable step cap.",
    )
    parser.add_argument(
        "--command-timeout-sec",
        type=int,
        default=180,
        help="Per-command timeout in seconds; set <=0 to disable timeout.",
    )
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--similarity-threshold", type=float, default=0.34)
    parser.add_argument("--structure-threshold", type=float, default=0.5)
    parser.add_argument("--cluster-backend", default="auto", choices=["auto", "lexical", "embedding-dbscan"])
    parser.add_argument("--cluster-embedding-model")
    parser.add_argument("--cluster-embedding-base-url")
    parser.add_argument("--cluster-dbscan-eps", type=float, default=0.35)
    parser.add_argument("--cluster-dbscan-min-samples", type=int, default=2)
    parser.add_argument("--cluster-embedding-strict", action="store_true")
    parser.add_argument(
        "--workflow-aggregation-mode",
        default="global",
        choices=["global", "per-task", "global-dbscan-qwen"],
    )
    parser.add_argument(
        "--no-per-task-skill-namespace",
        action="store_true",
        help="Disable task prefix for skill IDs in per-task aggregation mode.",
    )
    parser.add_argument("--skill-generation-mode", default="llm-agent", choices=["heuristic", "llm-agent"])
    parser.add_argument("--skill-creator-model")
    parser.add_argument("--skill-creator-base-url")
    parser.add_argument(
        "--skill-creator-agent-style",
        default="codex",
        choices=["codex", "claude-code", "cc", "opencode"],
    )
    parser.add_argument("--skill-creator-system-prompt")
    parser.add_argument("--skill-generation-strict-llm", action="store_true")
    parser.add_argument("--working-dir")
    parser.add_argument("--base-url")
    parser.add_argument(
        "--no-progress-display",
        action="store_true",
        help="Disable periodic progress display while Harbor is running.",
    )
    parser.add_argument(
        "--progress-interval-sec",
        type=int,
        default=20,
        help="Progress display refresh interval in seconds.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args, harbor_passthrough_args = parser.parse_known_args(argv)

    if not args.skill_repository and not args.bootstrap_input:
        raise ValueError("either --skill-repository or --bootstrap-input is required")

    requested_experiment_id = args.experiment_id
    normalized_experiment_id = normalize_experiment_name(requested_experiment_id, max_length=56)
    dataset_name, dataset_version = split_dataset_spec(args.dataset)
    dataset_slug = dataset_storage_slug(args.dataset)
    resolved_harness = (args.harness or f"{dataset_name}/harness").strip()
    resolved_import_benchmark = resolve_import_benchmark(
        dataset=args.dataset,
        import_benchmark=args.import_benchmark,
    )
    project_root = Path(__file__).resolve().parents[3]
    run_dir = (project_root / args.output_root / dataset_slug / normalized_experiment_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    imported_dir = run_dir / "imported"
    imported_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir = run_dir / "harbor-jobs"
    manifest_path = run_dir / "harbor-manifest.json"
    runbook_path = run_dir / "HARBOR_RUNBOOK.md"
    harbor_job_name = build_harbor_job_name(
        experiment_id=normalized_experiment_id,
        dataset=args.dataset,
        model=args.model,
        phase="live",
    )

    if args.skill_repository:
        skill_repository = args.skill_repository.resolve()
        bootstrap_input = None
        bootstrap_trajectory_count = None
        bootstrap_skill_count = None
    else:
        bootstrap_input = args.bootstrap_input.resolve()
        skill_repository = run_dir / "bootstrap-skills"
        bootstrap_trajectory_count, bootstrap_skill_count = None, None

    harbor_bin = resolve_harbor_binary(project_root)
    max_steps = _optional_positive(args.max_steps)
    command_timeout_sec = _optional_positive(args.command_timeout_sec)
    command = build_harbor_run_command(
        harbor_bin=harbor_bin,
        jobs_dir=jobs_dir,
        job_name=harbor_job_name,
        dataset=args.dataset,
        model=args.model,
        agent_mode=args.agent_mode,
        native_agent=args.native_agent,
        agent_import_path=args.agent_import_path,
        skill_repository=skill_repository,
        top_k_skills=args.top_k_skills,
        skill_selection_mode=args.skill_selection_mode,
        skill_candidate_pool=args.skill_candidate_pool,
        agent_kwargs=None,
        max_steps=max_steps,
        command_timeout_sec=command_timeout_sec,
        environment_type=args.environment_type,
        n_concurrent=args.n_concurrent,
        task_names=args.task_name,
        n_tasks=args.n_tasks,
        n_attempts=args.n_attempts,
        base_url=args.base_url,
        working_dir=args.working_dir,
        extra_args=harbor_passthrough_args,
    )

    manifest = {
        "experiment_id": normalized_experiment_id,
        "requested_experiment_id": requested_experiment_id,
        "dataset": args.dataset,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "dataset_slug": dataset_slug,
        "harness": resolved_harness,
        "import_benchmark": resolved_import_benchmark,
        "model": args.model,
        "agent_mode": args.agent_mode,
        "native_agent": args.native_agent if args.agent_mode == "native" else None,
        "agent_import_path": args.agent_import_path if args.agent_mode != "native" else None,
        "task_names": args.task_name,
        "n_attempts": args.n_attempts,
        "skill_selection_mode": args.skill_selection_mode,
        "skill_candidate_pool": args.skill_candidate_pool,
        "max_steps": max_steps,
        "command_timeout_sec": command_timeout_sec,
        "skill_repository": str(skill_repository),
        "bootstrap_input": str(bootstrap_input) if bootstrap_input else None,
        "harbor_command": render_command(command),
        "harbor_job_name": harbor_job_name,
        "harbor_passthrough_args": harbor_passthrough_args,
        "progress_display_enabled": not args.no_progress_display,
        "progress_interval_sec": args.progress_interval_sec,
        "jobs_dir": str(jobs_dir),
        "imported_path": str(imported_dir / "live-trajectories.jsonl"),
        "manifest_path": str(manifest_path),
        "runbook_path": str(runbook_path),
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    ensure_openrouter_env()

    if bootstrap_input:
        bootstrap_trajectory_count, bootstrap_skill_count, generation_meta = materialize_bootstrap_skills(
            bootstrap_input=bootstrap_input,
            bootstrap_skill_dir=skill_repository,
            import_benchmark=resolved_import_benchmark,
            harness=resolved_harness,
            min_support=args.min_support,
            similarity_threshold=args.similarity_threshold,
            structure_threshold=args.structure_threshold,
            cluster_backend=args.cluster_backend,
            cluster_embedding_model=args.cluster_embedding_model,
            cluster_embedding_base_url=args.cluster_embedding_base_url,
            cluster_dbscan_eps=args.cluster_dbscan_eps,
            cluster_dbscan_min_samples=args.cluster_dbscan_min_samples,
            cluster_embedding_strict=args.cluster_embedding_strict,
            workflow_aggregation_mode=args.workflow_aggregation_mode,
            per_task_skill_namespace=not args.no_per_task_skill_namespace,
            skill_creator_model=args.skill_creator_model,
            skill_creator_base_url=args.skill_creator_base_url,
            skill_creator_agent_style=args.skill_creator_agent_style,
            skill_creator_system_prompt=args.skill_creator_system_prompt,
        )
        manifest["bootstrap_trajectory_count"] = bootstrap_trajectory_count
        manifest["bootstrap_skill_count"] = bootstrap_skill_count
        manifest["skill_generation"] = generation_meta

    expected_trials = estimate_expected_trials(
        n_tasks=args.n_tasks,
        task_names=args.task_name,
        n_attempts=args.n_attempts,
    )
    run_harbor_job(
        command=command,
        project_root=project_root,
        jobs_dir=jobs_dir,
        job_name=harbor_job_name,
        show_progress=not args.no_progress_display,
        progress_interval_sec=args.progress_interval_sec,
        expected_trials=expected_trials,
    )
    job_dir = discover_job_dir(jobs_dir, harbor_job_name)
    alias_path = ensure_job_dir_alias(jobs_dir=jobs_dir, alias_name=harbor_job_name, actual_job_dir=job_dir)
    imported_path = imported_dir / "live-trajectories.jsonl"
    manifest["job_dir"] = str(job_dir)
    manifest["job_dir_alias"] = str(alias_path) if alias_path is not None else None
    manifest["imported_trajectory_count"] = import_harbor_job(
        job_dir,
        imported_path,
        dataset=args.dataset,
        import_benchmark=resolved_import_benchmark,
        harness=resolved_harness,
    )
    dump_manifest(manifest_path, manifest)
    formal_benchmark = "terminal-bench" if resolved_import_benchmark.startswith("terminal-bench") else resolved_import_benchmark
    runbook_path.write_text(
        "\n".join(
            [
                "# Harbor Experiment",
                "",
                f"- Experiment ID: {normalized_experiment_id}",
                f"- Requested Experiment ID: {requested_experiment_id}",
                f"- Dataset: {args.dataset}",
                f"- Model: {args.model}",
                f"- Harbor Job Name: {harbor_job_name}",
                f"- Skill Repository: {skill_repository}",
                "",
                "## Harbor Command",
                "",
                "```bash",
                render_command(command),
                "```",
                "",
                "## Import Follow-up",
                "",
                "```bash",
                f"bash scripts/server/run_formal_experiment.sh --benchmark {shlex.quote(formal_benchmark)} --experiment-id {shlex.quote(normalized_experiment_id + '-posthoc')} --memory-mode offline --input-path {shlex.quote(str(job_dir))}",
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
