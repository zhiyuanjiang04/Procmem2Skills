from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from procmem2skills.analysis.failure import extract_failure_signals_from_text
from procmem2skills.evaluation.pipeline import SkillDistillationPipeline
from procmem2skills.importers import import_terminal_bench
from procmem2skills.models import AtomicSkill, Trajectory
from procmem2skills.packager.materialize import materialize_skill_repository_standard_llm
from procmem2skills.packager.skill_writer import SkillWriter
from procmem2skills.recorder.jsonl import write_trajectories

from procmem2skills.integrations.harbor_terminal_experiment import (
    build_harbor_job_name,
    build_harbor_run_command,
    dataset_storage_slug,
    discover_job_dir,
    dump_manifest,
    ensure_openrouter_env,
    ensure_job_dir_alias,
    import_harbor_job,
    normalize_experiment_name,
    render_command,
    resolve_harbor_binary,
    run_harbor_job,
)
from procmem2skills.integrations.native_skill_prompt import build_native_skill_prompt_template

try:  # pragma: no cover - depends on harbor package at runtime
    from harbor.registry.client.factory import RegistryClientFactory
except Exception:  # pragma: no cover - fallback for non-harbor environments
    RegistryClientFactory = None


@dataclass(frozen=True)
class TerminalBenchTrialRecord:
    task_name: str
    trial_name: str
    reward: float | None
    exception_type: str | None
    trial_dir: Path


def load_trial_records(job_dir: Path) -> list[TerminalBenchTrialRecord]:
    root_result = (job_dir / "result.json").resolve()
    records: list[TerminalBenchTrialRecord] = []
    for result_path in sorted(job_dir.rglob("result.json")):
        if result_path.resolve() == root_result:
            continue
        payload = _load_json_object(result_path)
        task_name = str(payload.get("task_name") or payload.get("task_id") or result_path.parent.name.split("__", 1)[0])
        trial_name = str(payload.get("trial_name") or result_path.parent.name)
        exception_info = payload.get("exception_info")
        exception_type = str(exception_info.get("exception_type")) if isinstance(exception_info, dict) and exception_info.get("exception_type") else None
        reward = _extract_reward(payload)
        records.append(
            TerminalBenchTrialRecord(
                task_name=task_name,
                trial_name=trial_name,
                reward=reward,
                exception_type=exception_type,
                trial_dir=result_path.parent,
            )
        )
    return records


def summarize_task_success(job_dir: Path) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = defaultdict(lambda: {"attempts": 0, "successes": 0, "errors": 0})
    for record in load_trial_records(job_dir):
        summary[record.task_name]["attempts"] += 1
        if record.reward is not None and record.reward >= 1.0:
            summary[record.task_name]["successes"] += 1
        if record.exception_type:
            summary[record.task_name]["errors"] += 1
    return dict(summary)


def summarize_task_failures(
    records: list[TerminalBenchTrialRecord],
    *,
    allowed_tasks: set[str] | None = None,
    max_signals_per_task: int = 8,
) -> dict:
    buckets: dict[str, dict] = defaultdict(
        lambda: {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "errors": 0,
            "signal_counter": Counter(),
            "exception_counter": Counter(),
            "sample_failures": [],
        }
    )
    global_signal_counter: Counter[str] = Counter()

    for record in records:
        if allowed_tasks and record.task_name not in allowed_tasks:
            continue
        bucket = buckets[record.task_name]
        bucket["attempts"] += 1

        success = record.reward is not None and record.reward >= 1.0 and not record.exception_type
        if success:
            bucket["successes"] += 1
            continue

        bucket["failures"] += 1
        if record.exception_type:
            bucket["errors"] += 1
            bucket["exception_counter"][record.exception_type] += 1

        signals = _extract_trial_failure_signals(record)
        if not signals and record.exception_type:
            signals = [record.exception_type]
        if not signals:
            signals = ["unknown-failure"]
        for signal in signals:
            bucket["signal_counter"][signal] += 1
            global_signal_counter[signal] += 1

        if len(bucket["sample_failures"]) < 3:
            bucket["sample_failures"].append(
                {
                    "trial_name": record.trial_name,
                    "reward": record.reward,
                    "exception_type": record.exception_type,
                    "signals": signals[:3],
                }
            )

    by_task: dict[str, dict] = {}
    total_attempts = 0
    total_failures = 0
    for task_name, bucket in sorted(buckets.items()):
        total_attempts += bucket["attempts"]
        total_failures += bucket["failures"]
        by_task[task_name] = {
            "attempts": bucket["attempts"],
            "successes": bucket["successes"],
            "failures": bucket["failures"],
            "errors": bucket["errors"],
            "failure_signals": [
                {"signature": signature, "count": count}
                for signature, count in bucket["signal_counter"].most_common(max_signals_per_task)
            ],
            "exceptions": [
                {"exception_type": exception_type, "count": count}
                for exception_type, count in bucket["exception_counter"].most_common(max_signals_per_task)
            ],
            "sample_failures": bucket["sample_failures"],
        }

    return {
        "global": {
            "total_attempts": total_attempts,
            "failed_attempts": total_failures,
            "top_failure_signals": [
                {"signature": signature, "count": count}
                for signature, count in global_signal_counter.most_common(max_signals_per_task)
            ],
        },
        "by_task": by_task,
    }


def load_success_trajectories(job_dir: Path, allowed_tasks: set[str] | None = None) -> list[Trajectory]:
    return load_task_trajectories(job_dir, allowed_tasks=allowed_tasks, outcome="success")


def load_failed_trajectories(job_dir: Path, allowed_tasks: set[str] | None = None) -> list[Trajectory]:
    return load_task_trajectories(job_dir, allowed_tasks=allowed_tasks, outcome="failure")


def load_task_trajectories(
    job_dir: Path,
    *,
    allowed_tasks: set[str] | None = None,
    outcome: str = "all",
) -> list[Trajectory]:
    selected: list[Trajectory] = []
    normalized_outcome = outcome.strip().lower()
    if normalized_outcome not in {"all", "success", "failure"}:
        raise ValueError(f"unsupported outcome filter: {outcome}")
    for record in load_trial_records(job_dir):
        if allowed_tasks and record.task_name not in allowed_tasks:
            continue
        success = _record_is_success(record)
        if normalized_outcome == "success" and not success:
            continue
        if normalized_outcome == "failure" and success:
            continue
        for trajectory in import_terminal_bench(record.trial_dir):
            if allowed_tasks and trajectory.task_id not in allowed_tasks:
                continue
            selected.append(trajectory)
    return selected


def distill_atomic_skills(
    trajectories: list[Trajectory],
    *,
    cluster_backend: str,
    cluster_embedding_model: str | None,
    cluster_embedding_base_url: str | None,
    cluster_dbscan_eps: float,
    cluster_dbscan_min_samples: int,
    cluster_embedding_strict: bool,
    workflow_aggregation_mode: str,
    per_task_skill_namespace: bool,
) -> list[AtomicSkill]:
    if not trajectories:
        return []
    result = SkillDistillationPipeline(
        min_support=1,
        cluster_backend=cluster_backend,
        cluster_embedding_model=cluster_embedding_model,
        cluster_embedding_base_url=cluster_embedding_base_url,
        cluster_dbscan_eps=cluster_dbscan_eps,
        cluster_dbscan_min_samples=cluster_dbscan_min_samples,
        cluster_embedding_strict=cluster_embedding_strict,
        workflow_aggregation_mode=workflow_aggregation_mode,
        per_task_skill_namespace=per_task_skill_namespace,
    ).distill(trajectories)
    return result.skills


def materialize_distilled_skill_repository(
    *,
    skills: list[AtomicSkill],
    output_dir: Path,
    skill_creator_model: str | None,
    skill_creator_base_url: str | None,
    skill_creator_agent_style: str,
    skill_creator_system_prompt: str | None,
    failure_analysis: dict | None = None,
    failure_analysis_by_task: dict[str, dict] | None = None,
) -> tuple[int, dict]:
    _, generation_meta = materialize_skill_repository_standard_llm(
        skills=skills,
        output_dir=output_dir,
        writer=SkillWriter(),
        model=skill_creator_model,
        base_url=skill_creator_base_url,
        skill_creator_agent_style=skill_creator_agent_style,
        skill_creator_system_prompt=skill_creator_system_prompt,
        failure_analysis=failure_analysis,
        failure_analysis_by_task=failure_analysis_by_task,
    )
    return len(skills), generation_meta


def merge_atomic_skills(skill_groups: list[list[AtomicSkill]]) -> list[AtomicSkill]:
    merged: dict[str, AtomicSkill] = {}
    order: list[str] = []

    for group in skill_groups:
        for skill in group:
            existing = merged.get(skill.skill_id)
            if existing is None:
                merged[skill.skill_id] = skill
                order.append(skill.skill_id)
                continue
            merged[skill.skill_id] = existing.model_copy(
                update={
                    "support": int(existing.support) + int(skill.support),
                    "preconditions": sorted(set(existing.preconditions + skill.preconditions)),
                    "verification": sorted(set(existing.verification + skill.verification)),
                    "failure_recovery": sorted(set(existing.failure_recovery + skill.failure_recovery)),
                    "benchmark_origins": sorted(set(existing.benchmark_origins + skill.benchmark_origins)),
                    "harness_origins": sorted(set(existing.harness_origins + skill.harness_origins)),
                    "agent_origins": sorted(set(existing.agent_origins + skill.agent_origins)),
                    "task_origins": sorted(set(existing.task_origins + skill.task_origins)),
                    "source_workflow_ids": sorted(set(existing.source_workflow_ids + skill.source_workflow_ids)),
                }
            )

    return [merged[skill_id] for skill_id in order]


def select_transfer_tasks(
    *,
    task_names: list[str],
    strong_summary: dict[str, dict[str, int]],
    weak_summary: dict[str, dict[str, int]],
) -> tuple[list[str], set[str], set[str], dict[str, dict]]:
    candidate_tasks: list[str] = []
    success_tasks: set[str] = set()
    failure_only_tasks: set[str] = set()
    transfer_plan_by_task: dict[str, dict] = {}

    for task in task_names:
        strong = strong_summary.get(task, {})
        weak = weak_summary.get(task, {})
        strong_attempts = int(strong.get("attempts", 0))
        weak_attempts = int(weak.get("attempts", 0))
        strong_successes = int(strong.get("successes", 0))
        weak_successes = int(weak.get("successes", 0))
        strong_failures = max(0, strong_attempts - strong_successes)
        weak_failures = max(0, weak_attempts - weak_successes)

        strategy = "skip-no-transfer"
        include = False
        if weak_successes > 0:
            strategy = "skip-weak-has-success"
        elif strong_successes > 0:
            strategy = "success-transfer"
            include = True
            success_tasks.add(task)
        elif strong_failures > 0 or weak_failures > 0:
            strategy = "failure-reflection"
            include = True
            failure_only_tasks.add(task)
        else:
            strategy = "skip-insufficient-data"

        if include:
            candidate_tasks.append(task)

        transfer_plan_by_task[task] = {
            "strategy": strategy,
            "strong_attempts": strong_attempts,
            "strong_successes": strong_successes,
            "strong_failures": strong_failures,
            "weak_attempts": weak_attempts,
            "weak_successes": weak_successes,
            "weak_failures": weak_failures,
        }

    return candidate_tasks, success_tasks, failure_only_tasks, transfer_plan_by_task


def build_failure_guardrails(
    trajectories: list[Trajectory],
    *,
    allowed_tasks: set[str] | None = None,
    failure_analysis_by_task: dict[str, dict] | None = None,
    max_signatures_per_task: int = 5,
    max_sequences_per_task: int = 2,
    max_commands_per_sequence: int = 5,
) -> dict:
    by_task: dict[str, dict] = {}
    global_signals: Counter[str] = Counter()

    if failure_analysis_by_task:
        for task_name, payload in failure_analysis_by_task.items():
            if allowed_tasks and task_name not in allowed_tasks:
                continue
            if not isinstance(payload, dict):
                continue
            bucket = by_task.setdefault(task_name, {"signal_counter": Counter(), "sequences": []})
            for entry in payload.get("failure_signals") or []:
                if not isinstance(entry, dict):
                    continue
                signature = str(entry.get("signature") or "").strip()
                if not signature:
                    continue
                count = int(entry.get("count") or 1)
                bucket["signal_counter"][signature] += count
                global_signals[signature] += count

    for trajectory in trajectories:
        task_name = trajectory.task_id
        if not task_name:
            continue
        if allowed_tasks and task_name not in allowed_tasks:
            continue
        if _trajectory_is_success(trajectory):
            continue

        bucket = by_task.setdefault(task_name, {"signal_counter": Counter(), "sequences": []})
        sequence: list[str] = []

        for event in trajectory.events:
            action = event.action
            if action is not None:
                command = str(action.arguments.get("command") or action.raw or "").strip()
                command = command.splitlines()[0].strip() if command else ""
                if command and len(sequence) < max_commands_per_sequence:
                    sequence.append(command)

            result = event.result
            if result is None:
                continue
            for signal in extract_failure_signals_from_text(result.output_text or ""):
                bucket["signal_counter"][signal] += 1
                global_signals[signal] += 1

        if sequence and len(bucket["sequences"]) < max_sequences_per_task:
            bucket["sequences"].append(" -> ".join(sequence))

    rendered: dict[str, dict] = {}
    for task_name, payload in sorted(by_task.items()):
        rendered[task_name] = {
            "failure_signatures": [
                {"signature": signature, "count": count}
                for signature, count in payload["signal_counter"].most_common(max_signatures_per_task)
            ],
            "failed_command_sequences": payload["sequences"][:max_sequences_per_task],
        }

    return {
        "global_failure_signatures": [
            {"signature": signature, "count": count}
            for signature, count in global_signals.most_common(max_signatures_per_task)
        ],
        "by_task": rendered,
    }


def phase_command(
    *,
    experiment_id: str,
    project_root: Path,
    run_root: Path,
    phase_name: str,
    dataset: str,
    model: str,
    agent_mode: str,
    native_agent: str | None,
    agent_import_path: str,
    skill_repository: Path,
    task_names: list[str],
    n_attempts: int,
    top_k_skills: int,
    skill_selection_mode: str,
    skill_candidate_pool: int,
    agent_kwargs: dict[str, str] | None,
    max_steps: int | None,
    command_timeout_sec: int | None,
    environment_type: str,
    n_concurrent: int,
    base_url: str | None,
    harbor_extra_args: list[str] | None,
) -> tuple[list[str], Path, str]:
    phase_jobs_dir = run_root / phase_name / "harbor-jobs"
    phase_job_name = build_harbor_job_name(
        experiment_id=experiment_id,
        dataset=dataset,
        model=model,
        phase=phase_name,
    )
    command = build_harbor_run_command(
        harbor_bin=resolve_harbor_binary(project_root),
        jobs_dir=phase_jobs_dir,
        job_name=phase_job_name,
        dataset=dataset,
        model=model,
        agent_mode=agent_mode,
        native_agent=native_agent,
        agent_import_path=agent_import_path,
        skill_repository=skill_repository,
        top_k_skills=top_k_skills,
        skill_selection_mode=skill_selection_mode,
        skill_candidate_pool=skill_candidate_pool,
        agent_kwargs=agent_kwargs,
        max_steps=max_steps,
        command_timeout_sec=command_timeout_sec,
        environment_type=environment_type,
        n_concurrent=n_concurrent,
        task_names=task_names,
        n_tasks=len(task_names),
        n_attempts=n_attempts,
        base_url=base_url,
        working_dir=None,
        extra_args=harbor_extra_args,
    )
    return command, phase_jobs_dir, phase_job_name


def _optional_positive(value: int | None) -> int | None:
    if value is None:
        return None
    return value if value > 0 else None


def _resolve_existing_phase_job_dir(phase_jobs_dir: Path, phase_job_name: str, legacy_phase_name: str) -> Path:
    if not phase_jobs_dir.exists():
        raise FileNotFoundError(f"phase jobs dir does not exist for summarize-only: {phase_jobs_dir}")
    direct = phase_jobs_dir / phase_job_name
    if direct.exists():
        return direct
    legacy = phase_jobs_dir / legacy_phase_name
    if legacy.exists():
        return legacy
    return discover_job_dir(phase_jobs_dir, phase_job_name)


def _load_existing_skill_phase_summary(phases_dir: Path) -> dict | None:
    phase_jobs_dir = phases_dir / "weak-with-skills" / "harbor-jobs"
    if not phase_jobs_dir.exists():
        return None
    try:
        skill_job_dir = discover_job_dir(phase_jobs_dir, "weak-with-skills")
    except Exception:
        return None
    return {
        "skill_job_dir": str(skill_job_dir),
        "skill_summary": summarize_task_success(skill_job_dir),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run a Terminal-Bench strong-vs-weak transfer study with skill distillation.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--strong-model", required=True)
    parser.add_argument("--weak-model", required=True)
    parser.add_argument("--strong-agent-mode", choices=["skill-aware", "native"], default="native")
    parser.add_argument("--weak-agent-mode", choices=["skill-aware", "native"], default="native")
    parser.add_argument("--skill-agent-mode", choices=["skill-aware", "native"], default="native")
    parser.add_argument("--strong-native-agent", default="codex")
    parser.add_argument("--weak-native-agent", default="codex")
    parser.add_argument("--skill-native-agent", default="codex")
    parser.add_argument(
        "--agent-import-path",
        default="procmem2skills.integrations.harbor_terminal_agent:SkillAwareTerminalAgent",
    )
    parser.add_argument("--dataset", default="terminal-bench@2.0")
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--task-name", action="append")
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Load all tasks from the selected Harbor dataset registry entry.",
    )
    parser.add_argument(
        "--task-filter",
        default="",
        help="Optional substring filter applied after loading dataset tasks (works with --all-tasks).",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=0,
        help="Optional cap on selected task count after filtering/sharding; set <=0 for no cap.",
    )
    parser.add_argument(
        "--task-shard-count",
        type=int,
        default=1,
        help="Split selected tasks into N deterministic shards and run one shard in this process.",
    )
    parser.add_argument(
        "--task-shard-index",
        type=int,
        default=0,
        help="Zero-based shard index used with --task-shard-count.",
    )
    parser.add_argument("--strong-attempts", type=int, default=1)
    parser.add_argument("--weak-attempts", type=int, default=5)
    parser.add_argument("--skill-attempts", type=int, default=5)
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
    parser.add_argument("--environment-type", default="docker")
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--base-url")
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
    parser.add_argument(
        "--native-skill-injection",
        choices=["auto", "none", "prompt-template"],
        default="auto",
        help="How to inject distilled skills when skill phase runs in native agent mode.",
    )
    parser.add_argument(
        "--native-skill-max",
        type=int,
        default=12,
        help="Maximum number of distilled skills to inline into native prompt template; set <=0 for no cap.",
    )
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, harbor_passthrough_args = parser.parse_known_args(argv)
    requested_experiment_id = args.experiment_id
    canonical_experiment_id = normalize_experiment_name(requested_experiment_id, max_length=56)
    task_names = _resolve_task_names(
        explicit_task_names=args.task_name or [],
        all_tasks=args.all_tasks,
        dataset=args.dataset,
        task_filter=args.task_filter,
        task_limit=args.task_limit,
        task_shard_count=args.task_shard_count,
        task_shard_index=args.task_shard_index,
    )

    project_root = Path(__file__).resolve().parents[3]
    dataset_slug = dataset_storage_slug(args.dataset)
    run_root = (project_root / args.output_root / dataset_slug / canonical_experiment_id).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    phases_dir = run_root / "phases"
    phases_dir.mkdir(parents=True, exist_ok=True)
    empty_skill_repo = run_root / "empty-skills"
    empty_skill_repo.mkdir(parents=True, exist_ok=True)
    distilled_skill_repo = run_root / "distilled-skills"
    summary_path = run_root / "transfer-summary.json"
    max_steps = _optional_positive(args.max_steps)
    command_timeout_sec = _optional_positive(args.command_timeout_sec)

    strong_command, strong_jobs_dir, strong_job_name = phase_command(
        experiment_id=canonical_experiment_id,
        project_root=project_root,
        run_root=phases_dir,
        phase_name="strong-baseline",
        dataset=args.dataset,
        model=args.strong_model,
        agent_mode=args.strong_agent_mode,
        native_agent=args.strong_native_agent,
        agent_import_path=args.agent_import_path,
        skill_repository=empty_skill_repo,
        task_names=task_names,
        n_attempts=args.strong_attempts,
        top_k_skills=args.top_k_skills,
        skill_selection_mode=args.skill_selection_mode,
        skill_candidate_pool=args.skill_candidate_pool,
        agent_kwargs=None,
        max_steps=max_steps,
        command_timeout_sec=command_timeout_sec,
        environment_type=args.environment_type,
        n_concurrent=args.n_concurrent,
        base_url=args.base_url,
        harbor_extra_args=harbor_passthrough_args,
    )
    weak_command, weak_jobs_dir, weak_job_name = phase_command(
        experiment_id=canonical_experiment_id,
        project_root=project_root,
        run_root=phases_dir,
        phase_name="weak-baseline",
        dataset=args.dataset,
        model=args.weak_model,
        agent_mode=args.weak_agent_mode,
        native_agent=args.weak_native_agent,
        agent_import_path=args.agent_import_path,
        skill_repository=empty_skill_repo,
        task_names=task_names,
        n_attempts=args.weak_attempts,
        top_k_skills=args.top_k_skills,
        skill_selection_mode=args.skill_selection_mode,
        skill_candidate_pool=args.skill_candidate_pool,
        agent_kwargs=None,
        max_steps=max_steps,
        command_timeout_sec=command_timeout_sec,
        environment_type=args.environment_type,
        n_concurrent=args.n_concurrent,
        base_url=args.base_url,
        harbor_extra_args=harbor_passthrough_args,
    )

    plan = {
        "experiment_id": canonical_experiment_id,
        "requested_experiment_id": requested_experiment_id,
        "task_names": task_names,
        "task_count": len(task_names),
        "all_tasks": args.all_tasks,
        "task_filter": args.task_filter,
        "task_limit": _optional_positive(args.task_limit),
        "task_shard_count": args.task_shard_count,
        "task_shard_index": args.task_shard_index,
        "strong_model": args.strong_model,
        "weak_model": args.weak_model,
        "dataset": args.dataset,
        "dataset_slug": dataset_slug,
        "strong_agent_mode": args.strong_agent_mode,
        "weak_agent_mode": args.weak_agent_mode,
        "skill_agent_mode": args.skill_agent_mode,
        "strong_command": render_command(strong_command),
        "weak_command": render_command(weak_command),
        "phase_job_names": {
            "strong_baseline": strong_job_name,
            "weak_baseline": weak_job_name,
        },
        "skill_phase_attempts": args.skill_attempts,
        "skill_selection_mode": args.skill_selection_mode,
        "skill_candidate_pool": args.skill_candidate_pool,
        "native_skill_injection": args.native_skill_injection,
        "native_skill_max": _optional_positive(args.native_skill_max),
        "max_steps": max_steps,
        "command_timeout_sec": command_timeout_sec,
        "harbor_passthrough_args": harbor_passthrough_args,
        "summary_path": str(summary_path),
        "summarize_only": args.summarize_only,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    if args.summarize_only:
        strong_job_dir = _resolve_existing_phase_job_dir(strong_jobs_dir, strong_job_name, "strong-baseline")
        weak_job_dir = _resolve_existing_phase_job_dir(weak_jobs_dir, weak_job_name, "weak-baseline")
    else:
        ensure_openrouter_env()
        run_harbor_job(command=strong_command, project_root=project_root)
        strong_job_dir = discover_job_dir(strong_jobs_dir, strong_job_name)
        strong_alias = ensure_job_dir_alias(
            jobs_dir=strong_jobs_dir,
            alias_name=strong_job_name,
            actual_job_dir=strong_job_dir,
        )
        run_harbor_job(command=weak_command, project_root=project_root)
        weak_job_dir = discover_job_dir(weak_jobs_dir, weak_job_name)
        weak_alias = ensure_job_dir_alias(
            jobs_dir=weak_jobs_dir,
            alias_name=weak_job_name,
            actual_job_dir=weak_job_dir,
        )
        plan["phase_job_aliases"] = {
            "strong_baseline": str(strong_alias) if strong_alias is not None else None,
            "weak_baseline": str(weak_alias) if weak_alias is not None else None,
        }

    strong_summary = summarize_task_success(strong_job_dir)
    weak_summary = summarize_task_success(weak_job_dir)
    strong_records = load_trial_records(strong_job_dir)
    weak_records = load_trial_records(weak_job_dir)
    candidate_tasks, success_tasks, failure_only_tasks, transfer_plan_by_task = select_transfer_tasks(
        task_names=task_names,
        strong_summary=strong_summary,
        weak_summary=weak_summary,
    )
    analyzed_tasks = set(candidate_tasks) if candidate_tasks else set(task_names)
    failure_analysis = summarize_task_failures(
        strong_records + weak_records,
        allowed_tasks=analyzed_tasks,
    )
    failure_analysis_path = run_root / "weak-failure-analysis.json"
    dump_manifest(failure_analysis_path, failure_analysis)
    strong_success_trajectories = (
        load_success_trajectories(strong_job_dir, allowed_tasks=success_tasks)
        if success_tasks
        else []
    )
    failure_reflection_trajectories = []
    if failure_only_tasks:
        failure_reflection_trajectories.extend(
            load_failed_trajectories(strong_job_dir, allowed_tasks=failure_only_tasks)
        )
        failure_reflection_trajectories.extend(
            load_failed_trajectories(weak_job_dir, allowed_tasks=failure_only_tasks)
        )
    failed_candidate_trajectories = []
    if candidate_tasks:
        candidate_set = set(candidate_tasks)
        failed_candidate_trajectories.extend(
            load_failed_trajectories(strong_job_dir, allowed_tasks=candidate_set)
        )
        failed_candidate_trajectories.extend(
            load_failed_trajectories(weak_job_dir, allowed_tasks=candidate_set)
        )
    failure_guardrails = build_failure_guardrails(
        failed_candidate_trajectories,
        allowed_tasks=set(candidate_tasks) if candidate_tasks else None,
        failure_analysis_by_task=failure_analysis.get("by_task"),
    )

    if not candidate_tasks:
        payload = {
            **plan,
            "strong_summary": strong_summary,
            "weak_summary": weak_summary,
            "candidate_tasks": candidate_tasks,
            "transfer_plan_by_task": transfer_plan_by_task,
            "strong_job_dir": str(strong_job_dir),
            "weak_job_dir": str(weak_job_dir),
            "failure_analysis_path": str(failure_analysis_path),
            "failure_analysis": failure_analysis,
            "message": "no candidate tasks with weak zero-success",
        }
        existing_skill_phase = _load_existing_skill_phase_summary(phases_dir)
        if existing_skill_phase is not None:
            payload["existing_skill_phase"] = existing_skill_phase
            payload["note"] = (
                "existing weak-with-skills phase was found; it may target a different candidate task set than this corrected summary."
            )
        dump_manifest(summary_path, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.summarize_only:
        payload = {
            **plan,
            "strong_summary": strong_summary,
            "weak_summary": weak_summary,
            "candidate_tasks": candidate_tasks,
            "transfer_plan_by_task": transfer_plan_by_task,
            "strong_job_dir": str(strong_job_dir),
            "weak_job_dir": str(weak_job_dir),
            "failure_analysis_path": str(failure_analysis_path),
            "failure_analysis": failure_analysis,
            "failure_guardrails": failure_guardrails,
            "message": "summarize-only completed; no new weak-with-skills run launched",
        }
        existing_skill_phase = _load_existing_skill_phase_summary(phases_dir)
        if existing_skill_phase is not None:
            payload["existing_skill_phase"] = existing_skill_phase
        dump_manifest(summary_path, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    distilled_skill_repo.mkdir(parents=True, exist_ok=True)
    success_distilled_skills = distill_atomic_skills(
        strong_success_trajectories,
        cluster_backend=args.cluster_backend,
        cluster_embedding_model=args.cluster_embedding_model,
        cluster_embedding_base_url=args.cluster_embedding_base_url,
        cluster_dbscan_eps=args.cluster_dbscan_eps,
        cluster_dbscan_min_samples=args.cluster_dbscan_min_samples,
        cluster_embedding_strict=args.cluster_embedding_strict,
        workflow_aggregation_mode=args.workflow_aggregation_mode,
        per_task_skill_namespace=not args.no_per_task_skill_namespace,
    )
    failure_reflection_skills = distill_atomic_skills(
        failure_reflection_trajectories,
        cluster_backend=args.cluster_backend,
        cluster_embedding_model=args.cluster_embedding_model,
        cluster_embedding_base_url=args.cluster_embedding_base_url,
        cluster_dbscan_eps=args.cluster_dbscan_eps,
        cluster_dbscan_min_samples=args.cluster_dbscan_min_samples,
        cluster_embedding_strict=args.cluster_embedding_strict,
        workflow_aggregation_mode=args.workflow_aggregation_mode,
        per_task_skill_namespace=not args.no_per_task_skill_namespace,
    )
    distilled_skills = merge_atomic_skills([success_distilled_skills, failure_reflection_skills])
    skill_count, skill_generation = materialize_distilled_skill_repository(
        skills=distilled_skills,
        output_dir=distilled_skill_repo,
        skill_creator_model=args.skill_creator_model,
        skill_creator_base_url=args.skill_creator_base_url,
        skill_creator_agent_style=args.skill_creator_agent_style,
        skill_creator_system_prompt=args.skill_creator_system_prompt,
        failure_analysis=failure_analysis.get("global"),
        failure_analysis_by_task=failure_analysis.get("by_task"),
    )
    write_trajectories(run_root / "strong-successes.jsonl", strong_success_trajectories)
    write_trajectories(run_root / "failure-reflection.jsonl", failure_reflection_trajectories)

    skill_agent_kwargs: dict[str, str] | None = None
    native_skill_prompt: dict | None = None
    if args.skill_agent_mode == "native":
        injection_mode = _resolve_native_skill_injection(args.native_skill_injection, args.skill_native_agent)
        if injection_mode == "none":
            raise ValueError(
                "skill phase runs in native mode but no skill injection is enabled; "
                "set --native-skill-injection prompt-template or switch --skill-agent-mode to skill-aware"
            )
        if injection_mode == "prompt-template":
            prompt_template_path = run_root / "native-skill-prompt.j2"
            native_skill_prompt = build_native_skill_prompt_template(
                skill_repository=distilled_skill_repo,
                output_path=prompt_template_path,
                task_names=candidate_tasks,
                max_skills=_optional_positive(args.native_skill_max),
                failure_guardrails=failure_guardrails,
            )
            skill_agent_kwargs = {"prompt_template_path": str(prompt_template_path)}

    skill_command, skill_jobs_dir, skill_job_name = phase_command(
        experiment_id=canonical_experiment_id,
        project_root=project_root,
        run_root=phases_dir,
        phase_name="weak-with-skills",
        dataset=args.dataset,
        model=args.weak_model,
        agent_mode=args.skill_agent_mode,
        native_agent=args.skill_native_agent,
        agent_import_path=args.agent_import_path,
        skill_repository=distilled_skill_repo,
        task_names=candidate_tasks,
        n_attempts=args.skill_attempts,
        top_k_skills=args.top_k_skills,
        skill_selection_mode=args.skill_selection_mode,
        skill_candidate_pool=args.skill_candidate_pool,
        agent_kwargs=skill_agent_kwargs,
        max_steps=max_steps,
        command_timeout_sec=command_timeout_sec,
        environment_type=args.environment_type,
        n_concurrent=args.n_concurrent,
        base_url=args.base_url,
        harbor_extra_args=harbor_passthrough_args,
    )
    run_harbor_job(command=skill_command, project_root=project_root)
    skill_job_dir = discover_job_dir(skill_jobs_dir, skill_job_name)
    skill_alias = ensure_job_dir_alias(
        jobs_dir=skill_jobs_dir,
        alias_name=skill_job_name,
        actual_job_dir=skill_job_dir,
    )
    skill_summary = summarize_task_success(skill_job_dir)
    import_harbor_job(skill_job_dir, run_root / "weak-with-skills.jsonl")

    payload = {
        **plan,
        "candidate_tasks": candidate_tasks,
        "success_tasks": sorted(success_tasks),
        "failure_only_tasks": sorted(failure_only_tasks),
        "transfer_plan_by_task": transfer_plan_by_task,
        "skill_count": skill_count,
        "distilled_skill_breakdown": {
            "success_source_trajectories": len(strong_success_trajectories),
            "failure_reflection_trajectories": len(failure_reflection_trajectories),
            "success_distilled_skills": len(success_distilled_skills),
            "failure_reflection_skills": len(failure_reflection_skills),
        },
        "skill_generation": skill_generation,
        "distilled_skill_repo": str(distilled_skill_repo),
        "strong_summary": strong_summary,
        "weak_summary": weak_summary,
        "skill_command": render_command(skill_command),
        "skill_phase_job_name": skill_job_name,
        "skill_phase_job_alias": str(skill_alias) if skill_alias is not None else None,
        "skill_summary": skill_summary,
        "strong_job_dir": str(strong_job_dir),
        "weak_job_dir": str(weak_job_dir),
        "skill_job_dir": str(skill_job_dir),
        "failure_analysis_path": str(failure_analysis_path),
        "failure_analysis": failure_analysis,
        "failure_guardrails": failure_guardrails,
    }
    if native_skill_prompt:
        payload["native_skill_prompt"] = native_skill_prompt
    dump_manifest(summary_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _load_json_object(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_reward(payload: dict) -> float | None:
    verifier = payload.get("verifier_result")
    if not isinstance(verifier, dict):
        return None
    rewards = verifier.get("rewards")
    if not isinstance(rewards, dict):
        return None
    reward = rewards.get("reward")
    if reward is None:
        return None
    try:
        return float(reward)
    except (TypeError, ValueError):
        return None


def _record_is_success(record: TerminalBenchTrialRecord) -> bool:
    return record.reward is not None and record.reward >= 1.0 and not record.exception_type


def _trajectory_is_success(trajectory: Trajectory) -> bool:
    if trajectory.score is not None:
        return trajectory.score >= 1.0
    if not trajectory.completed:
        return False
    return all(event.result is None or event.result.ok for event in trajectory.events)


def _extract_trial_failure_signals(record: TerminalBenchTrialRecord) -> list[str]:
    signals: list[str] = []

    verifier_stdout = _read_optional_text(record.trial_dir / "verifier" / "test-stdout.txt")
    exception_text = _read_optional_text(record.trial_dir / "exception.txt")
    trial_log_text = _read_optional_text(record.trial_dir / "trial.log")

    for text in (verifier_stdout, exception_text, trial_log_text):
        for signal in extract_failure_signals_from_text(text):
            if signal not in signals:
                signals.append(signal)
            if len(signals) >= 6:
                return signals

    if record.exception_type and record.exception_type not in signals:
        signals.append(record.exception_type)
    return signals[:6]


def _read_optional_text(path: Path, *, max_chars: int = 24000) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(content) > max_chars:
        return content[-max_chars:]
    return content


def _resolve_native_skill_injection(requested: str, native_agent: str | None) -> str:
    mode = (requested or "auto").strip().lower()
    if mode in {"none", "prompt-template"}:
        return mode
    if mode != "auto":
        raise ValueError(f"unsupported native skill injection mode: {requested}")
    agent = (native_agent or "").strip().lower()
    if agent in {"codex", "opencode"}:
        return "prompt-template"
    return "none"


def _resolve_task_names(
    *,
    explicit_task_names: list[str],
    all_tasks: bool,
    dataset: str,
    task_filter: str,
    task_limit: int,
    task_shard_count: int,
    task_shard_index: int,
) -> list[str]:
    selected: list[str] = _dedupe_preserve_order(explicit_task_names)
    if all_tasks:
        selected = _dedupe_preserve_order(selected + _load_dataset_task_names(dataset))
    if task_filter:
        token = task_filter.strip()
        selected = [task for task in selected if token in task]
    if not selected:
        raise ValueError("no tasks selected; pass --task-name <name> or enable --all-tasks")
    if task_shard_count <= 0:
        raise ValueError("--task-shard-count must be > 0")
    if task_shard_index < 0 or task_shard_index >= task_shard_count:
        raise ValueError("--task-shard-index must be in [0, --task-shard-count)")
    if task_shard_count > 1:
        selected = [task for idx, task in enumerate(selected) if idx % task_shard_count == task_shard_index]
    if task_limit > 0:
        selected = selected[:task_limit]
    if not selected:
        raise ValueError("task selection resolved to an empty set after filters/sharding")
    return selected


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        name = item.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _load_dataset_task_names(dataset: str) -> list[str]:
    if RegistryClientFactory is None:
        raise RuntimeError("harbor registry client is unavailable; install harbor to use --all-tasks")
    dataset_name, dataset_version = _split_dataset_spec(dataset)
    client = RegistryClientFactory.create(None)
    for dataset_entry in client.get_datasets():
        payload = dataset_entry.model_dump()
        if payload.get("name") != dataset_name:
            continue
        if dataset_version is not None and str(payload.get("version")) != dataset_version:
            continue
        task_items = payload.get("tasks") or []
        names = sorted({str(item["name"]) for item in task_items if item.get("name")})
        if not names:
            raise RuntimeError(f"dataset {dataset} has no task entries")
        return names
    raise RuntimeError(f"dataset not found in harbor registry: {dataset}")


def _split_dataset_spec(dataset: str) -> tuple[str, str | None]:
    spec = dataset.strip()
    if not spec:
        raise ValueError("dataset spec cannot be empty")
    if "@" not in spec:
        return spec, None
    name, version = spec.split("@", 1)
    return name, (version or None)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - entrypoint guard
        print(f"error: {exc}", file=sys.stderr)
        raise
