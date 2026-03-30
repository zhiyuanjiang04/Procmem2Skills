from __future__ import annotations

import json
import hashlib
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import Any

from procmem2skills.inducer.workflow import induce_workflow
from procmem2skills.models import BoundaryReason, Event, Segment, Trajectory, WorkflowCandidate, WorkflowStep
from procmem2skills.segmenter.heuristics import segment_trajectory


class WorkflowAttemptStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"


class WorkflowInductionMode(str, Enum):
    RULE = "rule"
    LLM = "llm"
    HYBRID = "hybrid"


_ERROR_METADATA_KEYS = (
    "error",
    "error_message",
    "error_type",
    "exception",
    "exception_type",
    "traceback",
    "stacktrace",
    "runtime_error",
)

_ERROR_STATUS_KEYS = (
    "status",
    "run_status",
    "trial_status",
    "state",
)

_ERROR_STATUS_MARKERS = (
    "error",
    "exception",
    "crash",
    "timeout",
    "timed out",
    "aborted",
    "cancelled",
)


def classify_trajectory_status(trajectory: Trajectory) -> WorkflowAttemptStatus:
    if _is_error_trajectory(trajectory):
        return WorkflowAttemptStatus.ERROR
    if trajectory.score is not None:
        return WorkflowAttemptStatus.SUCCESS if float(trajectory.score) >= 1.0 else WorkflowAttemptStatus.FAILURE
    if not trajectory.completed:
        return WorkflowAttemptStatus.FAILURE
    has_failed_event = any(event.result is not None and not event.result.ok for event in trajectory.events)
    return WorkflowAttemptStatus.FAILURE if has_failed_event else WorkflowAttemptStatus.SUCCESS


def segment_trajectory_for_workflow_export(
    trajectory: Trajectory,
    *,
    terminal_like_max_events_per_segment: int = 6,
) -> list[Segment]:
    if _is_terminal_or_skills_bench_trace(trajectory):
        return _segment_terminal_like_trace(
            trajectory,
            max_events_per_segment=max(1, int(terminal_like_max_events_per_segment)),
        )
    return segment_trajectory(trajectory)


def induce_workflow_attempt(
    trajectory: Trajectory,
    *,
    terminal_like_max_events_per_segment: int = 6,
    induction_mode: str | WorkflowInductionMode = WorkflowInductionMode.RULE,
    llm_inducer: Any | None = None,
) -> dict[str, Any] | None:
    status = classify_trajectory_status(trajectory)
    if status == WorkflowAttemptStatus.ERROR:
        return None

    mode = _resolve_induction_mode(induction_mode)
    segments = segment_trajectory_for_workflow_export(
        trajectory,
        terminal_like_max_events_per_segment=terminal_like_max_events_per_segment,
    )
    if not segments:
        return None

    workflows = []
    for segment in segments:
        workflow = _induce_workflow_for_segment(
            segment=segment,
            trajectory=trajectory,
            status=status,
            mode=mode,
            llm_inducer=llm_inducer,
        )
        workflows.append(workflow.model_dump(mode="json"))

    if not workflows:
        return None

    return {
        "episode_id": trajectory.episode_id,
        "status": status.value,
        "benchmark": trajectory.benchmark.value,
        "harness": trajectory.harness,
        "agent": trajectory.agent,
        "task_id": trajectory.task_id,
        "instruction": trajectory.instruction,
        "completed": bool(trajectory.completed),
        "score": trajectory.score,
        "trajectory_metadata": trajectory.metadata,
        "workflows": workflows,
    }


def induce_workflows_grouped_by_task(
    trajectories: list[Trajectory],
    *,
    terminal_like_max_events_per_segment: int = 6,
    induction_mode: str | WorkflowInductionMode = WorkflowInductionMode.RULE,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_timeout_sec: int = 120,
    llm_max_retries: int = 1,
    llm_strict: bool = True,
    checkpoint_every: int = 0,
    checkpoint_output_path: Path | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    requested_mode = _resolve_induction_mode(induction_mode)
    mode = requested_mode
    llm_inducer = _build_llm_inducer(
        mode=mode,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_timeout_sec=llm_timeout_sec,
        llm_max_retries=llm_max_retries,
        llm_strict=llm_strict,
    )
    degraded_reason = ""
    if mode in (WorkflowInductionMode.LLM, WorkflowInductionMode.HYBRID) and llm_inducer is not None:
        resolved_api_key = str(getattr(llm_inducer, "api_key", "") or "").strip()
        if not resolved_api_key:
            if mode == WorkflowInductionMode.LLM:
                raise RuntimeError(
                    "LLM workflow induction mode requires OPENROUTER_API_KEY/OPENAI_API_KEY or --llm-api-key."
                )
            mode = WorkflowInductionMode.RULE
            llm_inducer = None
            degraded_reason = "hybrid requested without LLM API key; downgraded to rule induction"

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summary = {
        "input_trajectories": len(trajectories),
        "success": 0,
        "failure": 0,
        "error_discarded": 0,
        "retained_attempts": 0,
        "retained_tasks": 0,
        "induction_mode": mode.value,
        "requested_induction_mode": requested_mode.value,
    }
    checkpoint_interval = max(0, int(checkpoint_every))
    checkpoint_writes = 0
    if checkpoint_interval > 0 and checkpoint_output_path is not None:
        summary["checkpoint_every"] = checkpoint_interval
    if degraded_reason:
        summary["mode_degraded"] = True
        summary["mode_degraded_reason"] = degraded_reason

    retained_attempt_count = 0
    for trajectory in trajectories:
        status = classify_trajectory_status(trajectory)
        if status == WorkflowAttemptStatus.ERROR:
            summary["error_discarded"] += 1
            continue

        attempt = induce_workflow_attempt(
            trajectory,
            terminal_like_max_events_per_segment=terminal_like_max_events_per_segment,
            induction_mode=mode,
            llm_inducer=llm_inducer,
        )
        if attempt is None:
            summary["error_discarded"] += 1
            continue

        task_key = (trajectory.task_id or trajectory.episode_id or "unknown-task").strip() or "unknown-task"
        attempt["attempt_index"] = len(grouped[task_key]) + 1
        grouped[task_key].append(attempt)
        summary[status.value] += 1
        retained_attempt_count += 1

        if (
            checkpoint_interval > 0
            and checkpoint_output_path is not None
            and retained_attempt_count % checkpoint_interval == 0
        ):
            ordered_snapshot = _finalize_grouped_attempts(grouped)
            _write_grouped_payload(checkpoint_output_path, ordered_snapshot)
            checkpoint_writes += 1

    ordered = _finalize_grouped_attempts(grouped)

    summary["retained_attempts"] = sum(len(items) for items in ordered.values())
    summary["retained_tasks"] = len(ordered)
    if checkpoint_interval > 0 and checkpoint_output_path is not None:
        summary["checkpoint_writes"] = checkpoint_writes
    return ordered, summary


def export_grouped_workflows_json(
    trajectories: list[Trajectory],
    output_path: Path,
    *,
    terminal_like_max_events_per_segment: int = 6,
    induction_mode: str | WorkflowInductionMode = WorkflowInductionMode.RULE,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_timeout_sec: int = 120,
    llm_max_retries: int = 1,
    llm_strict: bool = True,
    checkpoint_every: int = 0,
    collection_target_k: int = 0,
    collection_metadata_output_path: Path | None = None,
) -> dict[str, Any]:
    grouped, summary = induce_workflows_grouped_by_task(
        trajectories,
        terminal_like_max_events_per_segment=terminal_like_max_events_per_segment,
        induction_mode=induction_mode,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_timeout_sec=llm_timeout_sec,
        llm_max_retries=llm_max_retries,
        llm_strict=llm_strict,
        checkpoint_every=checkpoint_every,
        checkpoint_output_path=output_path if int(checkpoint_every) > 0 else None,
    )

    effective_collection_k = max(0, int(collection_target_k))
    if effective_collection_k > 0:
        grouped, collection_meta = apply_balanced_k_collection_policy(
            grouped,
            target_k=effective_collection_k,
        )
        summary["collection_policy"] = collection_meta.get("summary", {})
        summary["collection_target_k"] = effective_collection_k

        metadata_output = collection_metadata_output_path
        if metadata_output is None:
            metadata_output = output_path.with_suffix(".collection-metadata.json")
        _write_json_payload(metadata_output, collection_meta)
        summary["collection_metadata_output"] = str(metadata_output)

    _write_grouped_payload(output_path, grouped)
    return summary


def apply_balanced_k_collection_policy(
    grouped_attempts: dict[str, list[dict[str, Any]]],
    *,
    target_k: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    k = max(1, int(target_k))

    curated: dict[str, list[dict[str, Any]]] = {}
    per_task_metadata: dict[str, dict[str, Any]] = {}
    label_counter: Counter[str] = Counter()

    total_attempts = 0
    total_selected = 0

    for task_id in sorted(grouped_attempts.keys()):
        attempts = _sort_attempts_for_collection(grouped_attempts.get(task_id, []))
        total_attempts += len(attempts)

        status_records = []
        for index, attempt in enumerate(attempts):
            status = _normalize_attempt_status_for_collection(attempt.get("status"))
            if status:
                status_records.append((index, status))

        initial_records = status_records[:k]
        initial_success = sum(1 for _, status in initial_records if status == "success")
        initial_failure = sum(1 for _, status in initial_records if status == "failure")

        selected_indices: set[int] = set()
        label = "insufficient-attempts"

        if len(initial_records) >= k:
            if initial_success == k:
                label = "all-success"
                selected_indices = {index for index, _ in initial_records}
            elif initial_failure == k:
                label = "all-failure"
                selected_indices = {index for index, _ in initial_records}
            else:
                success_taken = 0
                failure_taken = 0
                for index, status in status_records:
                    if status == "success" and success_taken < k:
                        selected_indices.add(index)
                        success_taken += 1
                    elif status == "failure" and failure_taken < k:
                        selected_indices.add(index)
                        failure_taken += 1
                    if success_taken >= k and failure_taken >= k:
                        break
                label = "mixed-balanced" if (success_taken >= k and failure_taken >= k) else "mixed-insufficient"
        else:
            if len(initial_records) > 0:
                if initial_success == len(initial_records):
                    label = "insufficient-all-success"
                elif initial_failure == len(initial_records):
                    label = "insufficient-all-failure"
                else:
                    label = "insufficient-mixed"
            selected_indices = {index for index, _ in status_records}

        selected_attempts: list[dict[str, Any]] = []
        selected_success = 0
        selected_failure = 0
        for index, attempt in enumerate(attempts):
            if index not in selected_indices:
                continue
            copied = dict(attempt)
            copied["attempt_index"] = len(selected_attempts) + 1
            selected_attempts.append(copied)
            status = _normalize_attempt_status_for_collection(copied.get("status"))
            if status == "success":
                selected_success += 1
            elif status == "failure":
                selected_failure += 1

        curated[task_id] = selected_attempts
        selected_count = len(selected_attempts)
        total_selected += selected_count
        valid_attempt_count = len(status_records)

        task_meta = {
            "task_id": task_id,
            "policy": "k-balanced-v1",
            "target_k": k,
            "label": label,
            "total_attempts": len(attempts),
            "valid_attempts": valid_attempt_count,
            "initial_window_size": len(initial_records),
            "initial_success": initial_success,
            "initial_failure": initial_failure,
            "observed_success": sum(1 for _, status in status_records if status == "success"),
            "observed_failure": sum(1 for _, status in status_records if status == "failure"),
            "selected_success": selected_success,
            "selected_failure": selected_failure,
            "selected_total": selected_count,
            "discarded_total": max(0, valid_attempt_count - selected_count),
        }
        per_task_metadata[task_id] = task_meta
        label_counter[label] += 1

    summary = {
        "policy": "k-balanced-v1",
        "target_k": k,
        "task_count": len(per_task_metadata),
        "label_counts": dict(sorted(label_counter.items())),
        "input_attempts": total_attempts,
        "selected_attempts": total_selected,
        "discarded_attempts": max(0, total_attempts - total_selected),
    }

    return curated, {"summary": summary, "tasks": per_task_metadata}

def _resolve_induction_mode(value: str | WorkflowInductionMode) -> WorkflowInductionMode:
    if isinstance(value, WorkflowInductionMode):
        return value
    normalized = str(value or "rule").strip().lower()
    try:
        return WorkflowInductionMode(normalized)
    except Exception as exc:
        raise ValueError(f"unsupported workflow induction mode: {value}") from exc


def _build_llm_inducer(
    *,
    mode: WorkflowInductionMode,
    llm_model: str | None,
    llm_base_url: str | None,
    llm_api_key: str | None,
    llm_timeout_sec: int,
    llm_max_retries: int,
    llm_strict: bool,
) -> Any | None:
    if mode == WorkflowInductionMode.RULE:
        return None
    from procmem2skills.inducer.llm_workflow import LLMWorkflowInducer

    return LLMWorkflowInducer(
        model=llm_model,
        base_url=llm_base_url,
        api_key=llm_api_key,
        timeout_sec=llm_timeout_sec,
        max_retries=llm_max_retries,
        strict=llm_strict,
    )


def _induce_workflow_for_segment(
    *,
    segment: Segment,
    trajectory: Trajectory,
    status: WorkflowAttemptStatus,
    mode: WorkflowInductionMode,
    llm_inducer: Any | None,
) -> WorkflowCandidate:
    rule_workflow = induce_workflow(segment)
    if mode == WorkflowInductionMode.RULE:
        return rule_workflow

    if llm_inducer is None:
        if mode == WorkflowInductionMode.HYBRID:
            rule_workflow.metadata = {
                **(rule_workflow.metadata if isinstance(rule_workflow.metadata, dict) else {}),
                "induction_mode": "hybrid",
                "llm_fallback": True,
                "llm_error": "llm inducer unavailable",
            }
            return rule_workflow
        raise RuntimeError("LLM workflow inducer is required for llm/hybrid mode")

    try:
        llm_workflow = llm_inducer.induce(
            segment=segment,
            trajectory=trajectory,
            attempt_status=status.value,
            base_metadata=rule_workflow.metadata,
        )
    except Exception as exc:
        if mode == WorkflowInductionMode.HYBRID:
            rule_workflow.metadata = {
                **(rule_workflow.metadata if isinstance(rule_workflow.metadata, dict) else {}),
                "induction_mode": "hybrid",
                "llm_fallback": True,
                "llm_error": str(exc)[:400],
            }
            return rule_workflow
        raise

    if mode == WorkflowInductionMode.LLM:
        llm_workflow.metadata = {
            **(llm_workflow.metadata if isinstance(llm_workflow.metadata, dict) else {}),
            "induction_mode": "llm",
        }
        return llm_workflow

    return _merge_llm_with_rule_workflow(llm_workflow=llm_workflow, rule_workflow=rule_workflow)


def _merge_llm_with_rule_workflow(*, llm_workflow: WorkflowCandidate, rule_workflow: WorkflowCandidate) -> WorkflowCandidate:
    merged = llm_workflow.model_copy(deep=True)

    if not merged.objective:
        merged.objective = rule_workflow.objective
    if not merged.trigger:
        merged.trigger = rule_workflow.trigger

    merged.preconditions = _dedupe_text((merged.preconditions or []) + (rule_workflow.preconditions or []))
    merged.verification = _dedupe_text((merged.verification or []) + (rule_workflow.verification or []))
    merged.failure_modes = _dedupe_text((merged.failure_modes or []) + (rule_workflow.failure_modes or []))

    if len(merged.steps) < len(rule_workflow.steps):
        merged.steps = _merge_steps(merged.steps, rule_workflow.steps)
    for index, step in enumerate(merged.steps, start=1):
        step.order = index

    merged.fingerprint = _fingerprint(merged.steps)
    merged.metadata = {
        **(rule_workflow.metadata if isinstance(rule_workflow.metadata, dict) else {}),
        **(merged.metadata if isinstance(merged.metadata, dict) else {}),
        "induction_mode": "hybrid",
        "hybrid_coverage": {
            "rule_step_count": len(rule_workflow.steps),
            "llm_step_count": len(llm_workflow.steps),
            "merged_step_count": len(merged.steps),
        },
    }
    return merged


def _merge_steps(llm_steps: list[WorkflowStep], rule_steps: list[WorkflowStep]) -> list[WorkflowStep]:
    merged: list[WorkflowStep] = []
    seen = set()

    for step in llm_steps + rule_steps:
        key = _step_key(step)
        if key in seen:
            continue
        seen.add(key)
        merged.append(step.model_copy(deep=True))

    return merged


def _step_key(step: WorkflowStep) -> str:
    tool = (step.tool or "").strip().lower()
    operation = (step.operation or "").strip().lower()
    intent = (step.intent or "").strip().lower()
    return f"{tool}|{operation}|{intent}"


def _fingerprint(steps: list[WorkflowStep]) -> str:
    basis = "|".join(f"{step.tool}:{step.operation}" for step in steps)
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]


def _dedupe_text(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


def _finalize_grouped_attempts(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    ordered: dict[str, list[dict[str, Any]]] = {}
    for task_key in sorted(grouped):
        attempts = _sort_attempts_for_collection(grouped[task_key])
        for index, attempt in enumerate(attempts, start=1):
            attempt["attempt_index"] = index
        ordered[task_key] = attempts
    return ordered


def _write_grouped_payload(output_path: Path, payload: dict[str, list[dict[str, Any]]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(output_path)

def _write_json_payload(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(output_path)


def _sort_attempts_for_collection(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = []
    for fallback_index, attempt in enumerate(attempts, start=1):
        index = attempt.get("attempt_index")
        try:
            numeric_index = int(index)
        except Exception:
            numeric_index = fallback_index
        indexed.append((numeric_index, fallback_index, attempt))
    indexed.sort(key=lambda item: (item[0], item[1], str(item[2].get("episode_id", ""))))
    return [item[2] for item in indexed]


def _normalize_attempt_status_for_collection(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status in {"success", "failure"}:
        return status
    return ""


def _is_error_trajectory(trajectory: Trajectory) -> bool:
    if not trajectory.events:
        return True

    metadata = trajectory.metadata if isinstance(trajectory.metadata, dict) else {}
    for key in _ERROR_METADATA_KEYS:
        text = _stringify(metadata.get(key)).lower()
        if text:
            return True

    for key in _ERROR_STATUS_KEYS:
        text = _stringify(metadata.get(key)).lower()
        if not text:
            continue
        if any(marker in text for marker in _ERROR_STATUS_MARKERS):
            return True

    if trajectory.score is None and not trajectory.completed and not any(event.action for event in trajectory.events):
        return True

    return False


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _is_terminal_or_skills_bench_trace(trajectory: Trajectory) -> bool:
    benchmark = (trajectory.benchmark.value or "").strip().lower()
    harness = (trajectory.harness or "").strip().lower()
    return benchmark == "terminal-bench" or "terminal-bench" in harness or "skills-bench" in harness


def _segment_terminal_like_trace(
    trajectory: Trajectory,
    *,
    max_events_per_segment: int,
) -> list[Segment]:
    if not trajectory.events:
        return []

    segments: list[Segment] = []
    current_events: list[Event] = []
    start_step = trajectory.events[0].step_id

    for index, event in enumerate(trajectory.events):
        current_events.append(event)
        boundary_reasons: list[BoundaryReason] = []
        next_event = trajectory.events[index + 1] if index + 1 < len(trajectory.events) else None

        if len(current_events) >= max_events_per_segment:
            boundary_reasons.append(BoundaryReason.MAX_EVENTS)
        if event.success_signal:
            boundary_reasons.append(BoundaryReason.SUCCESS_SIGNAL)
        if event.result is not None and not event.result.ok:
            boundary_reasons.append(BoundaryReason.MAX_EVENTS)
        if next_event and event.action and next_event.action and event.action.tool != next_event.action.tool:
            boundary_reasons.append(BoundaryReason.TOOL_SWITCH)

        if boundary_reasons or next_event is None:
            segment_id = f"{trajectory.episode_id}-seg-{len(segments) + 1}"
            segments.append(
                Segment(
                    segment_id=segment_id,
                    episode_id=trajectory.episode_id,
                    start_step=start_step,
                    end_step=event.step_id,
                    reasons=list(dict.fromkeys(boundary_reasons)) or [BoundaryReason.MAX_EVENTS],
                    tool_sequence=[item.action.tool for item in current_events if item.action],
                    summary_hint=_segment_summary_for_trace(current_events),
                    events=list(current_events),
                )
            )
            current_events = []
            start_step = next_event.step_id if next_event else event.step_id

    return segments


def _segment_summary_for_trace(events: list[Event]) -> str:
    for event in events:
        command = _extract_terminal_command(event)
        if command:
            return command[:140]
        if event.observation and event.observation.summary:
            return event.observation.summary[:140]
    return "terminal segment"


def _extract_terminal_command(event: Event) -> str:
    if not event.action:
        return ""
    if event.action.raw:
        return " ".join(event.action.raw.split())[:200]
    command = event.action.arguments.get("command")
    if isinstance(command, str) and command.strip():
        return " ".join(command.split())[:200]
    return event.action.name[:200]
