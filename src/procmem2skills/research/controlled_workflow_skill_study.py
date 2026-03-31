from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from procmem2skills.models import AtomicSkill, WorkflowCandidate, WorkflowStep
from procmem2skills.runtime.workflow_memory import normalize_task_key

_CONDITION_PATTERN = re.compile(r"^\s*(\d+)\s*s\s*(\d+)\s*f\s*$", re.IGNORECASE)
_PAIR_PATTERN = re.compile(r"^\s*(\d+)\s*[,/:x]\s*(\d+)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class WorkflowMixCondition:
    label: str
    success_count: int
    failure_count: int

    @property
    def total_count(self) -> int:
        return self.success_count + self.failure_count

    @property
    def is_mixed(self) -> bool:
        return self.success_count > 0 and self.failure_count > 0


@dataclass(frozen=True)
class TaskWorkflowItem:
    task_id: str
    attempt_index: int
    attempt_status: str
    workflow: WorkflowCandidate
    episode_id: str | None = None
    instruction: str | None = None


@dataclass
class TaskWorkflowPool:
    task_id: str
    instruction: str | None = None
    success_workflows: list[TaskWorkflowItem] = field(default_factory=list)
    failure_workflows: list[TaskWorkflowItem] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return len(self.success_workflows)

    @property
    def failure_count(self) -> int:
        return len(self.failure_workflows)

    @property
    def total_count(self) -> int:
        return self.success_count + self.failure_count


@dataclass(frozen=True)
class WorkflowSelection:
    task_id: str
    condition: WorkflowMixCondition
    success_workflows: list[TaskWorkflowItem]
    failure_workflows: list[TaskWorkflowItem]
    random_seed: int

    @property
    def workflows(self) -> list[TaskWorkflowItem]:
        return [*self.success_workflows, *self.failure_workflows]


@dataclass(frozen=True)
class TaskEligibility:
    task_id: str
    eligible: bool
    reasons: list[str]
    success_count: int
    failure_count: int


def parse_condition_specs(specs: Sequence[str]) -> list[WorkflowMixCondition]:
    if not specs:
        raise ValueError("at least one workflow mix condition is required")

    parsed: list[WorkflowMixCondition] = []
    seen_labels: set[str] = set()
    for spec in specs:
        raw = str(spec or "").strip()
        match = _CONDITION_PATTERN.match(raw) or _PAIR_PATTERN.match(raw)
        if not match:
            raise ValueError(
                f"invalid workflow mix condition: {spec!r} "
                "(expected format like 4s1f or m,n)"
            )

        success_count = int(match.group(1))
        failure_count = int(match.group(2))
        label = f"{success_count}s{failure_count}f"
        if success_count < 0 or failure_count < 0:
            raise ValueError(f"condition counts must be >= 0: {spec!r}")
        if success_count + failure_count <= 0:
            raise ValueError(f"condition must include at least one workflow: {spec!r}")
        if label in seen_labels:
            continue

        parsed.append(
            WorkflowMixCondition(
                label=label,
                success_count=success_count,
                failure_count=failure_count,
            )
        )
        seen_labels.add(label)

    if not parsed:
        raise ValueError("no valid workflow mix conditions parsed")
    return parsed


def collect_task_workflow_pools(grouped_payload: Mapping[str, Any]) -> dict[str, TaskWorkflowPool]:
    pools: dict[str, TaskWorkflowPool] = {}

    for task_key, attempts in grouped_payload.items():
        if not isinstance(attempts, list):
            continue

        for fallback_attempt_index, attempt in enumerate(attempts, start=1):
            if not isinstance(attempt, dict):
                continue

            status = _normalize_attempt_status(attempt.get("status"))
            if status not in {"success", "failure"}:
                continue

            raw_task_id = str(attempt.get("task_id") or task_key or "").strip()
            task_id = raw_task_id or str(task_key or "unknown-task")
            instruction = str(attempt.get("instruction") or "").strip() or None
            episode_id = str(attempt.get("episode_id") or "").strip() or None

            pool = pools.get(task_id)
            if pool is None:
                pool = TaskWorkflowPool(task_id=task_id, instruction=instruction)
                pools[task_id] = pool
            elif pool.instruction is None and instruction:
                pool.instruction = instruction

            raw_attempt_index = attempt.get("attempt_index")
            attempt_index = _coerce_attempt_index(raw_attempt_index, fallback_attempt_index)

            for workflow_index, workflow_payload in enumerate(attempt.get("workflows") or [], start=1):
                workflow = _coerce_workflow_candidate(
                    workflow_payload,
                    task_id=task_id,
                    episode_id=episode_id,
                    attempt_index=attempt_index,
                    workflow_index=workflow_index,
                )
                if workflow is None:
                    continue
                item = TaskWorkflowItem(
                    task_id=task_id,
                    attempt_index=attempt_index,
                    attempt_status=status,
                    workflow=workflow,
                    episode_id=episode_id,
                    instruction=instruction,
                )
                if status == "success":
                    pool.success_workflows.append(item)
                else:
                    pool.failure_workflows.append(item)

    for pool in pools.values():
        pool.success_workflows.sort(key=_workflow_item_sort_key)
        pool.failure_workflows.sort(key=_workflow_item_sort_key)

    return dict(sorted(pools.items(), key=lambda item: item[0]))


def select_eligible_tasks(
    pools: Mapping[str, TaskWorkflowPool],
    conditions: Sequence[WorkflowMixCondition],
    *,
    require_success: bool = True,
    require_failure_for_mixed: bool = True,
    require_counts_for_all_conditions: bool = True,
    minimum_mixed_success_count: int = 0,
    minimum_mixed_failure_count: int = 0,
    minimum_success_pool_size: int | None = None,
    minimum_failure_pool_size: int | None = None,
) -> list[str]:
    report = evaluate_task_eligibility(
        pools,
        conditions,
        require_success=require_success,
        require_failure_for_mixed=require_failure_for_mixed,
        require_counts_for_all_conditions=require_counts_for_all_conditions,
        minimum_mixed_success_count=minimum_mixed_success_count,
        minimum_mixed_failure_count=minimum_mixed_failure_count,
        minimum_success_pool_size=minimum_success_pool_size,
        minimum_failure_pool_size=minimum_failure_pool_size,
    )
    return sorted(task_id for task_id, item in report.items() if item.eligible)


def evaluate_task_eligibility(
    pools: Mapping[str, TaskWorkflowPool],
    conditions: Sequence[WorkflowMixCondition],
    *,
    require_success: bool = True,
    require_failure_for_mixed: bool = True,
    require_counts_for_all_conditions: bool = True,
    minimum_mixed_success_count: int = 0,
    minimum_mixed_failure_count: int = 0,
    minimum_success_pool_size: int | None = None,
    minimum_failure_pool_size: int | None = None,
) -> dict[str, TaskEligibility]:
    if not conditions:
        raise ValueError("at least one condition is required for task eligibility")

    has_mixed_condition = any(condition.is_mixed for condition in conditions)
    max_required_success = max(condition.success_count for condition in conditions)
    max_required_failure = max(condition.failure_count for condition in conditions)

    requested_min_success = max(0, int(minimum_success_pool_size or 0))
    requested_min_failure = max(0, int(minimum_failure_pool_size or 0))
    mixed_floor_success = max(0, int(minimum_mixed_success_count or 0)) if has_mixed_condition else 0
    mixed_floor_failure = max(0, int(minimum_mixed_failure_count or 0)) if has_mixed_condition else 0

    if require_counts_for_all_conditions:
        requested_min_success = max(requested_min_success, max_required_success)
        requested_min_failure = max(requested_min_failure, max_required_failure)

    report: dict[str, TaskEligibility] = {}
    for task_id, pool in pools.items():
        reasons: list[str] = []
        success_count = int(pool.success_count)
        failure_count = int(pool.failure_count)

        if require_success and success_count <= 0:
            reasons.append("always-failure:no-success-workflow")

        if has_mixed_condition and require_failure_for_mixed and failure_count <= 0:
            reasons.append("always-success:no-failure-workflow")

        if success_count < requested_min_success:
            reasons.append(f"insufficient-success:{success_count}<{requested_min_success}")

        if failure_count < requested_min_failure:
            reasons.append(f"insufficient-failure:{failure_count}<{requested_min_failure}")

        if has_mixed_condition and require_failure_for_mixed:
            if success_count < mixed_floor_success:
                reasons.append(f"insufficient-mixed-success:{success_count}<{mixed_floor_success}")
            if failure_count < mixed_floor_failure:
                reasons.append(f"insufficient-mixed-failure:{failure_count}<{mixed_floor_failure}")

        if require_counts_for_all_conditions:
            for condition in conditions:
                if _can_satisfy_condition(pool, condition):
                    continue
                reasons.append(
                    f"condition-unsatisfied:{condition.label}"
                )
        else:
            if not any(_can_satisfy_condition(pool, condition) for condition in conditions):
                reasons.append("no-condition-satisfied")

        report[task_id] = TaskEligibility(
            task_id=task_id,
            eligible=len(reasons) == 0,
            reasons=reasons,
            success_count=success_count,
            failure_count=failure_count,
        )

    return dict(sorted(report.items(), key=lambda item: item[0]))


def sample_workflows_for_condition(
    pool: TaskWorkflowPool,
    *,
    condition: WorkflowMixCondition,
    random_seed: int = 0,
) -> WorkflowSelection:
    if pool.success_count < condition.success_count:
        raise ValueError(
            f"task {pool.task_id!r} does not have enough success workflows for {condition.label}: "
            f"{pool.success_count}<{condition.success_count}"
        )
    if pool.failure_count < condition.failure_count:
        raise ValueError(
            f"task {pool.task_id!r} does not have enough failure workflows for {condition.label}: "
            f"{pool.failure_count}<{condition.failure_count}"
        )

    rng = random.Random(_stable_seed(pool.task_id, condition.label, random_seed))
    success_sample = _sample_items(rng, pool.success_workflows, condition.success_count)
    failure_sample = _sample_items(rng, pool.failure_workflows, condition.failure_count)

    return WorkflowSelection(
        task_id=pool.task_id,
        condition=condition,
        success_workflows=success_sample,
        failure_workflows=failure_sample,
        random_seed=random_seed,
    )


def sample_workflows_fixed_count(
    pool: TaskWorkflowPool,
    *,
    sample_count: int,
    random_seed: int = 0,
    seed_label: str | None = None,
) -> WorkflowSelection:
    total = int(sample_count)
    if total <= 0:
        raise ValueError(f"sample_count must be > 0, got {sample_count}")
    if pool.total_count < total:
        raise ValueError(
            f"task {pool.task_id!r} does not have enough workflows for fixed-count sampling: "
            f"{pool.total_count}<{total}"
        )

    combined = [*pool.success_workflows, *pool.failure_workflows]
    label = str(seed_label or f"fixed-{total}").strip() or f"fixed-{total}"
    rng = random.Random(_stable_seed(pool.task_id, label, random_seed))
    sampled = _sample_items(rng, combined, total)

    success_sample = [item for item in sampled if item.attempt_status == "success"]
    failure_sample = [item for item in sampled if item.attempt_status == "failure"]
    condition = WorkflowMixCondition(
        label=f"{len(success_sample)}s{len(failure_sample)}f",
        success_count=len(success_sample),
        failure_count=len(failure_sample),
    )
    return WorkflowSelection(
        task_id=pool.task_id,
        condition=condition,
        success_workflows=success_sample,
        failure_workflows=failure_sample,
        random_seed=random_seed,
    )


def build_atomic_skill_from_selection(
    *,
    task_id: str,
    selection: WorkflowSelection,
    skill_namespace: str = "controlled-mix",
) -> AtomicSkill:
    effective_task_id = str(task_id or selection.task_id or "unknown-task").strip() or "unknown-task"
    task_key = normalize_task_key(effective_task_id) or "unknown-task"
    namespace_key = normalize_task_key(skill_namespace) or "controlled-mix"
    skill_id = f"{namespace_key}--{task_key}--{selection.condition.label}"

    workflow_items = [*selection.success_workflows, *selection.failure_workflows]
    source_workflow_ids = _ordered_unique(
        item.workflow.workflow_id
        for item in workflow_items
        if str(item.workflow.workflow_id or "").strip()
    )

    trigger = _first_non_empty(item.workflow.trigger for item in workflow_items) or f"When solving task {effective_task_id}."

    preconditions = _ordered_unique(
        precondition
        for item in workflow_items
        for precondition in item.workflow.preconditions
        if str(precondition or "").strip()
    )

    verification = _ordered_unique(
        check
        for item in workflow_items
        for check in item.workflow.verification
        if str(check or "").strip()
    )

    failure_recovery = _ordered_unique(
        failure_mode
        for item in workflow_items
        for failure_mode in item.workflow.failure_modes
        if str(failure_mode or "").strip()
    )

    actions: list[WorkflowStep] = []
    order = 1
    for item in workflow_items:
        channel = "success" if item.attempt_status == "success" else "failure"
        for step in item.workflow.steps:
            actions.append(
                WorkflowStep(
                    order=order,
                    intent=f"[{channel}] {step.intent}".strip(),
                    tool=step.tool,
                    operation=step.operation,
                    preconditions=list(step.preconditions),
                    verification=step.verification,
                )
            )
            order += 1

    functional_name = _derive_functional_skill_name(
        workflow_items=workflow_items,
        task_key=task_key,
        fallback=f"{namespace_key}-{task_key}",
    )
    title = _humanize_skill_name(functional_name) or f"Task Workflow Skill {task_key}"

    metadata = {
        "workflow_mix": {
            "success": selection.condition.success_count,
            "failure": selection.condition.failure_count,
            "label": selection.condition.label,
        },
        "selection_seed": selection.random_seed,
        "selected_attempts": [
            {
                "workflow_id": item.workflow.workflow_id,
                "attempt_status": item.attempt_status,
                "attempt_index": item.attempt_index,
                "episode_id": item.episode_id,
            }
            for item in workflow_items
        ],
        "skill_name": functional_name,
        "output_layout": {
            "root": "created_skills",
            "condition": selection.condition.label,
            "task": task_key,
            "skill_name": functional_name,
        },
    }

    return AtomicSkill(
        skill_id=skill_id,
        title=title,
        description=(
            f"Reusable procedure distilled from {len(source_workflow_ids)} workflows for task {effective_task_id}."
        ),
        canonical_key=f"{task_key}:{selection.condition.label}",
        trigger=trigger,
        preconditions=preconditions,
        actions=actions,
        verification=verification,
        failure_recovery=failure_recovery,
        task_origins=[effective_task_id],
        source_workflow_ids=source_workflow_ids,
        support=len(source_workflow_ids),
        metadata=metadata,
    )


def _derive_functional_skill_name(
    *,
    workflow_items: list[TaskWorkflowItem],
    task_key: str,
    fallback: str,
) -> str:
    task_tokens = set(token for token in task_key.split("-") if token)
    for candidate in _derive_functional_name_candidates(workflow_items):
        slug = normalize_task_key(candidate)
        if not slug:
            continue
        tokens = [token for token in slug.split("-") if token and token not in task_tokens]
        if len(tokens) >= 2:
            return "-".join(tokens[:8])
        if tokens:
            return "-".join(tokens)
    normalized_fallback = normalize_task_key(fallback)
    return normalized_fallback or "workflow-playbook"


def _derive_functional_name_candidates(workflow_items: list[TaskWorkflowItem]) -> list[str]:
    candidates: list[str] = []
    for item in workflow_items:
        for step in item.workflow.steps[:3]:
            intent = str(step.intent or "").strip()
            if intent:
                candidates.append(intent)
            operation = str(step.operation or "").strip()
            command_match = re.search(r"command=([^\)]+)\)", operation)
            if command_match:
                candidates.append(command_match.group(1).strip())
            elif operation and "(" not in operation and ")" not in operation:
                candidates.append(operation)
        objective = str(item.workflow.objective or "").strip()
        if objective:
            candidates.append(objective)
    return candidates


def _humanize_skill_name(skill_name: str) -> str:
    tokens = [token for token in str(skill_name or "").strip().split("-") if token]
    if not tokens:
        return ""
    return " ".join(token.capitalize() for token in tokens[:8])


def _normalize_attempt_status(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return ""
    if normalized in {"success", "passed", "pass", "ok", "completed"}:
        return "success"
    if normalized in {"failure", "failed", "fail", "unsolved", "timeout", "timed_out", "timed-out"}:
        return "failure"
    return normalized


def _coerce_workflow_candidate(
    workflow_payload: Any,
    *,
    task_id: str,
    episode_id: str | None,
    attempt_index: int,
    workflow_index: int,
) -> WorkflowCandidate | None:
    if not isinstance(workflow_payload, dict):
        return None
    try:
        return WorkflowCandidate.model_validate(workflow_payload)
    except Exception:
        pass

    normalized = dict(workflow_payload)
    workflow_id = str(normalized.get("workflow_id") or "").strip()
    if not workflow_id:
        workflow_id = f"{task_id}-a{attempt_index}-w{workflow_index}"
    normalized["workflow_id"] = workflow_id

    source_segment_id = str(normalized.get("source_segment_id") or "").strip()
    if not source_segment_id:
        episode_part = str(episode_id or task_id or "episode").strip()
        source_segment_id = f"{episode_part}-a{attempt_index}-seg-{workflow_index}"
    normalized["source_segment_id"] = source_segment_id

    objective = str(normalized.get("objective") or "").strip()
    if not objective:
        objective = f"Solve task {task_id}."
    normalized["objective"] = objective

    trigger = str(normalized.get("trigger") or "").strip()
    if not trigger:
        trigger = f"When solving task {task_id}."
    normalized["trigger"] = trigger

    normalized["preconditions"] = _as_str_list(normalized.get("preconditions"))
    normalized["verification"] = _as_str_list(normalized.get("verification"))
    normalized["failure_modes"] = _as_str_list(normalized.get("failure_modes"))
    if not isinstance(normalized.get("metadata"), dict):
        normalized["metadata"] = {}

    normalized_steps: list[dict[str, Any]] = []
    for step_idx, step_payload in enumerate(normalized.get("steps") or [], start=1):
        if not isinstance(step_payload, dict):
            continue
        intent = str(step_payload.get("intent") or "").strip() or f"Step {step_idx}"
        operation = str(step_payload.get("operation") or "").strip() or f"execute: {intent}"
        normalized_steps.append(
            {
                "order": int(step_payload.get("order") or step_idx),
                "intent": intent,
                "tool": step_payload.get("tool"),
                "operation": operation,
                "preconditions": _as_str_list(step_payload.get("preconditions")),
                "verification": str(step_payload.get("verification") or "").strip() or None,
            }
        )
    if not normalized_steps:
        normalized_steps.append(
            {
                "order": 1,
                "intent": "Execute objective directly.",
                "tool": None,
                "operation": f"attempt task objective: {objective}",
                "preconditions": [],
                "verification": None,
            }
        )
    normalized["steps"] = normalized_steps

    fingerprint = str(normalized.get("fingerprint") or "").strip()
    if not fingerprint:
        fingerprint = _fingerprint_workflow_payload(normalized)
    normalized["fingerprint"] = fingerprint

    try:
        return WorkflowCandidate.model_validate(normalized)
    except Exception:
        return None


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _fingerprint_workflow_payload(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8",
        errors="ignore",
    )
    return hashlib.sha256(data).hexdigest()[:24]


def _coerce_attempt_index(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _workflow_item_sort_key(item: TaskWorkflowItem) -> tuple[int, str, str]:
    return (
        int(item.attempt_index),
        str(item.workflow.source_segment_id or ""),
        str(item.workflow.workflow_id or ""),
    )


def _can_satisfy_condition(pool: TaskWorkflowPool, condition: WorkflowMixCondition) -> bool:
    return pool.success_count >= condition.success_count and pool.failure_count >= condition.failure_count


def _sample_items(rng: random.Random, items: list[TaskWorkflowItem], count: int) -> list[TaskWorkflowItem]:
    if count <= 0:
        return []
    indices = list(range(len(items)))
    rng.shuffle(indices)
    selected = [items[index] for index in indices[:count]]
    selected.sort(key=_workflow_item_sort_key)
    return selected


def _stable_seed(task_id: str, condition_label: str, random_seed: int) -> int:
    payload = f"{task_id}::{condition_label}::{random_seed}".encode("utf-8", errors="ignore")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _ordered_unique(values) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _first_non_empty(values) -> str:
    for raw in values:
        value = str(raw or "").strip()
        if value:
            return value
    return ""


__all__ = [
    "TaskEligibility",
    "TaskWorkflowItem",
    "TaskWorkflowPool",
    "WorkflowMixCondition",
    "WorkflowSelection",
    "build_atomic_skill_from_selection",
    "collect_task_workflow_pools",
    "evaluate_task_eligibility",
    "parse_condition_specs",
    "sample_workflows_fixed_count",
    "sample_workflows_for_condition",
    "select_eligible_tasks",
]
