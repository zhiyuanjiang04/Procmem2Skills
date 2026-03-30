#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from procmem2skills.packager.materialize import materialize_skill_repository_standard_llm
from procmem2skills.research.controlled_workflow_skill_study import (
    WorkflowMixCondition,
    build_atomic_skill_from_selection,
    collect_task_workflow_pools,
    evaluate_task_eligibility,
    sample_workflows_for_condition,
)

DEFAULT_MIXES: list[tuple[int, int]] = [(5, 0), (4, 1), (3, 2), (2, 3), (1, 4), (0, 5)]
DEFAULT_COLLECTION_LABEL_CHOICES: tuple[str, ...] = (
    "all-success",
    "all-failure",
    "mixed-balanced",
    "mixed-insufficient",
    "insufficient-attempts",
    "insufficient-all-success",
    "insufficient-all-failure",
    "insufficient-mixed",
)
DEFAULT_SKILL_CREATOR_SYSTEM_PROMPT = (
    "You are generating one task-level atomic skill for a controlled workflow-memory study. "
    "Only use evidence from provided workflows, preserve actionable steps, and avoid speculative additions. "
    "Prioritize concise, reusable procedures and explicit verification checks."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled workflow->skill study pipeline: "
            "task filtering by success/failure pool constraints, fixed-mix sampling, and standardized skill generation."
        )
    )
    parser.add_argument("--workflow-input", type=Path, required=True, help="Grouped workflow JSON (task -> attempts list).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for manifest, selections, and skills.")

    parser.add_argument(
        "--mix",
        nargs=2,
        action="append",
        type=int,
        metavar=("M_SUCCESS", "N_FAILURE"),
        default=[],
        help=(
            "Repeatable m/n mix input. Example: --mix 5 0 --mix 4 1."
        ),
    )
    parser.add_argument("--success-count", type=int, help="Single-condition success count m.")
    parser.add_argument("--failure-count", type=int, help="Single-condition failure count n.")

    parser.add_argument("--task-name", action="append", default=[], help="Only include selected task IDs.")
    parser.add_argument("--exclude-task-name", action="append", default=[], help="Exclude selected task IDs.")
    parser.add_argument("--max-tasks", type=int, help="Cap eligible tasks after filtering.")
    parser.add_argument(
        "--collection-metadata",
        type=Path,
        help="Optional k-balanced collection metadata JSON for task-level label filtering.",
    )
    parser.add_argument(
        "--collection-label",
        action="append",
        choices=list(DEFAULT_COLLECTION_LABEL_CHOICES),
        default=[],
        help="Repeatable collection label filter (e.g., --collection-label mixed-balanced).",
    )
    parser.add_argument("--random-seed", type=int, default=0)

    parser.add_argument("--require-success", action="store_true", default=None)
    parser.add_argument("--allow-no-success", action="store_true", help="Disable success-workflow requirement.")
    parser.add_argument("--require-failure-for-mixed", action="store_true", default=None)
    parser.add_argument(
        "--allow-mixed-without-failure",
        action="store_true",
        help="Disable failure-workflow requirement for mixed conditions.",
    )
    parser.add_argument(
        "--require-counts-for-all-conditions",
        action="store_true",
        default=None,
        help="Require every selected task to satisfy all conditions.",
    )
    parser.add_argument(
        "--allow-partial-condition-coverage",
        action="store_true",
        help="Allow tasks that satisfy only a subset of conditions.",
    )
    parser.add_argument(
        "--minimum-mixed-success-count",
        type=int,
        default=0,
        help="Extra lower bound for success pools when any mixed condition exists.",
    )
    parser.add_argument(
        "--minimum-mixed-failure-count",
        type=int,
        default=0,
        help="Extra lower bound for failure pools when any mixed condition exists.",
    )
    parser.add_argument(
        "--minimum-success-pool-size",
        type=int,
        help="Explicit task-level minimum success pool size (e.g., 5 for 5s/5f studies).",
    )
    parser.add_argument(
        "--minimum-failure-pool-size",
        type=int,
        help="Explicit task-level minimum failure pool size (e.g., 5 for 5s/5f studies).",
    )

    parser.add_argument("--skill-namespace", default="controlled")
    parser.add_argument("--skip-materialize", action="store_true", help="Only export sampled atomic skill JSON.")
    parser.add_argument("--skill-output-dir", type=Path, help="Materialized skill repository path.")
    parser.add_argument("--skill-creator-model", default="openai/gpt-5.3-codex")
    parser.add_argument("--skill-creator-base-url")
    parser.add_argument(
        "--skill-creator-agent-style",
        choices=["codex", "claude-code", "cc", "opencode"],
        default="codex",
    )
    parser.add_argument(
        "--skill-creator-system-prompt",
        default=DEFAULT_SKILL_CREATOR_SYSTEM_PROMPT,
        help="Shared system prompt for all conditions/tasks in this controlled study.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _condition_from_pair(success_count: int, failure_count: int) -> WorkflowMixCondition:
    if success_count < 0 or failure_count < 0:
        raise ValueError(f"mix counts must be non-negative, got m={success_count}, n={failure_count}")
    if success_count + failure_count <= 0:
        raise ValueError("mix counts cannot both be zero")
    return WorkflowMixCondition(
        label=f"{success_count}s{failure_count}f",
        success_count=int(success_count),
        failure_count=int(failure_count),
    )


def _resolve_conditions(args: argparse.Namespace) -> list[WorkflowMixCondition]:
    if args.mix:
        dedup: dict[str, WorkflowMixCondition] = {}
        for pair in args.mix:
            condition = _condition_from_pair(int(pair[0]), int(pair[1]))
            dedup.setdefault(condition.label, condition)
        return list(dedup.values())

    if args.success_count is not None or args.failure_count is not None:
        if args.success_count is None or args.failure_count is None:
            raise ValueError("when using --success-count/--failure-count, both must be provided")
        return [_condition_from_pair(int(args.success_count), int(args.failure_count))]

    dedup: dict[str, WorkflowMixCondition] = {}
    for success_count, failure_count in DEFAULT_MIXES:
        condition = _condition_from_pair(success_count, failure_count)
        dedup.setdefault(condition.label, condition)
    return list(dedup.values())


def _load_grouped_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"workflow input must be a JSON object: {path}")
    return payload


def _apply_task_scope(
    task_ids: list[str],
    *,
    include: list[str],
    exclude: list[str],
    max_tasks: int | None,
) -> list[str]:
    scoped = list(task_ids)
    if include:
        include_set = set(include)
        scoped = [task_id for task_id in scoped if task_id in include_set]
    if exclude:
        exclude_set = set(exclude)
        scoped = [task_id for task_id in scoped if task_id not in exclude_set]
    if max_tasks is not None and max_tasks > 0:
        scoped = scoped[:max_tasks]
    return scoped


def _load_collection_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"collection metadata must be a JSON object: {path}")
    tasks_payload = payload.get("tasks")
    if isinstance(tasks_payload, dict):
        normalized = {
            str(task_id): meta
            for task_id, meta in tasks_payload.items()
            if isinstance(meta, dict)
        }
        return normalized
    normalized = {
        str(task_id): meta
        for task_id, meta in payload.items()
        if isinstance(meta, dict)
    }
    return normalized


def _filter_tasks_by_collection_label(
    task_ids: list[str],
    *,
    collection_metadata: dict[str, Any] | None,
    required_labels: list[str],
) -> tuple[list[str], dict[str, str], int]:
    if not required_labels:
        return task_ids, {}, 0
    if not collection_metadata:
        return [], {}, len(task_ids)

    allowed = set(required_labels)
    selected: list[str] = []
    labels_by_task: dict[str, str] = {}
    dropped = 0
    for task_id in task_ids:
        meta = collection_metadata.get(task_id)
        label = str(meta.get("label") or "") if isinstance(meta, dict) else ""
        if label in allowed:
            selected.append(task_id)
            labels_by_task[task_id] = label
        else:
            dropped += 1
    return selected, labels_by_task, dropped


def _selection_payload(selection) -> dict[str, Any]:
    def to_entry(item) -> dict[str, Any]:
        return {
            "workflow_id": item.workflow.workflow_id,
            "attempt_index": item.attempt_index,
            "attempt_status": item.attempt_status,
            "episode_id": item.episode_id,
            "objective": item.workflow.objective,
        }

    return {
        "condition": {
            "label": selection.condition.label,
            "success": selection.condition.success_count,
            "failure": selection.condition.failure_count,
        },
        "random_seed": selection.random_seed,
        "success_workflows": [to_entry(item) for item in selection.success_workflows],
        "failure_workflows": [to_entry(item) for item in selection.failure_workflows],
    }


def main() -> int:
    args = parse_args()

    conditions = _resolve_conditions(args)

    workflow_input = args.workflow_input.resolve()
    output_dir = args.output_dir.resolve()
    grouped_payload = _load_grouped_payload(workflow_input)

    collection_metadata_path = None
    collection_metadata = None
    if args.collection_metadata is not None:
        collection_metadata_path = args.collection_metadata.resolve()
    else:
        auto_metadata = workflow_input.with_suffix(".collection-metadata.json")
        if auto_metadata.is_file():
            collection_metadata_path = auto_metadata
    if collection_metadata_path is not None:
        collection_metadata = _load_collection_metadata(collection_metadata_path)
    pools = collect_task_workflow_pools(grouped_payload)

    inferred_require_success = any(condition.success_count > 0 for condition in conditions)
    inferred_require_failure_for_mixed = any(
        condition.success_count > 0 and condition.failure_count > 0 for condition in conditions
    )

    if args.allow_no_success:
        require_success = False
    elif args.require_success is True:
        require_success = True
    else:
        require_success = inferred_require_success

    if args.allow_mixed_without_failure:
        require_failure_for_mixed = False
    elif args.require_failure_for_mixed is True:
        require_failure_for_mixed = True
    else:
        require_failure_for_mixed = inferred_require_failure_for_mixed

    if args.allow_partial_condition_coverage:
        require_counts_for_all_conditions = False
    elif args.require_counts_for_all_conditions is True:
        require_counts_for_all_conditions = True
    else:
        require_counts_for_all_conditions = True

    eligibility = evaluate_task_eligibility(
        pools,
        conditions,
        require_success=require_success,
        require_failure_for_mixed=require_failure_for_mixed,
        require_counts_for_all_conditions=require_counts_for_all_conditions,
        minimum_mixed_success_count=max(0, int(args.minimum_mixed_success_count)),
        minimum_mixed_failure_count=max(0, int(args.minimum_mixed_failure_count)),
        minimum_success_pool_size=args.minimum_success_pool_size,
        minimum_failure_pool_size=args.minimum_failure_pool_size,
    )

    eligible_tasks = sorted(task_id for task_id, item in eligibility.items() if item.eligible)
    eligible_tasks, collection_labels_by_task, dropped_by_collection_label = _filter_tasks_by_collection_label(
        eligible_tasks,
        collection_metadata=collection_metadata,
        required_labels=args.collection_label,
    )
    selected_tasks = _apply_task_scope(
        eligible_tasks,
        include=args.task_name,
        exclude=args.exclude_task_name,
        max_tasks=args.max_tasks,
    )

    selections_by_task: dict[str, dict[str, dict[str, Any]]] = {}
    atomic_skills = []
    for task_id in selected_tasks:
        pool = pools[task_id]
        per_condition: dict[str, dict[str, Any]] = {}
        for condition in conditions:
            selection = sample_workflows_for_condition(
                pool,
                condition=condition,
                random_seed=args.random_seed,
            )
            per_condition[condition.label] = _selection_payload(selection)
            atomic_skills.append(
                build_atomic_skill_from_selection(
                    task_id=task_id,
                    selection=selection,
                    skill_namespace=args.skill_namespace,
                )
            )
        selections_by_task[task_id] = per_condition

    output_dir.mkdir(parents=True, exist_ok=True)
    eligibility_path = output_dir / "task_eligibility.json"
    selections_path = output_dir / "workflow_selections.json"
    atomic_skills_path = output_dir / "atomic_skills.json"
    manifest_path = output_dir / "manifest.json"

    eligibility_payload = {
        task_id: {
            "eligible": item.eligible,
            "success_count": item.success_count,
            "failure_count": item.failure_count,
            "reasons": item.reasons,
        }
        for task_id, item in eligibility.items()
    }

    atomic_payload = [skill.model_dump(mode="json") for skill in atomic_skills]

    if not args.dry_run:
        eligibility_path.write_text(json.dumps(eligibility_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        selections_path.write_text(json.dumps(selections_by_task, indent=2, ensure_ascii=False), encoding="utf-8")
        atomic_skills_path.write_text(json.dumps(atomic_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    generation_meta: dict[str, Any] = {
        "requested_mode": "llm-agent",
        "effective_mode": None,
        "llm_model": args.skill_creator_model,
        "written_skill_dirs": 0,
    }
    skill_output_dir = (args.skill_output_dir.resolve() if args.skill_output_dir else output_dir / "skills")

    if not args.skip_materialize:
        if not args.dry_run:
            _, generation_meta = materialize_skill_repository_standard_llm(
                skills=atomic_skills,
                output_dir=skill_output_dir,
                model=args.skill_creator_model,
                base_url=args.skill_creator_base_url,
                skill_creator_agent_style=args.skill_creator_agent_style,
                skill_creator_system_prompt=args.skill_creator_system_prompt,
            )
        else:
            generation_meta["effective_mode"] = "llm-agent"

    manifest = {
        "workflow_input": str(workflow_input),
        "output_dir": str(output_dir),
        "conditions": [condition.model_dump() if hasattr(condition, "model_dump") else condition.__dict__ for condition in conditions],
        "filtering": {
            "require_success": require_success,
            "require_failure_for_mixed": require_failure_for_mixed,
            "require_counts_for_all_conditions": require_counts_for_all_conditions,
            "minimum_mixed_success_count": args.minimum_mixed_success_count,
            "minimum_mixed_failure_count": args.minimum_mixed_failure_count,
            "minimum_success_pool_size": args.minimum_success_pool_size,
            "minimum_failure_pool_size": args.minimum_failure_pool_size,
            "task_include": args.task_name,
            "task_exclude": args.exclude_task_name,
            "max_tasks": args.max_tasks,
        },
        "collection_filter": {
            "collection_metadata": str(collection_metadata_path) if collection_metadata_path else None,
            "collection_labels": args.collection_label,
            "dropped_by_collection_label": dropped_by_collection_label,
        },
        "condition_input": {
            "mix": args.mix,
            "success_count": args.success_count,
            "failure_count": args.failure_count,
        },
        "task_counts": {
            "total_with_workflows": len(pools),
            "eligible_before_scope": len(eligible_tasks),
            "selected_tasks": len(selected_tasks),
            "dropped_tasks": len([task_id for task_id, item in eligibility.items() if not item.eligible]),
            "collection_label_coverage": {
                label: sum(1 for task_id in selected_tasks if collection_labels_by_task.get(task_id) == label)
                for label in sorted(set(collection_labels_by_task.values()))
            },
        },
        "outputs": {
            "eligibility": str(eligibility_path),
            "selections": str(selections_path),
            "atomic_skills": str(atomic_skills_path),
            "skill_repository": str(skill_output_dir),
        },
        "skill_generation": generation_meta,
        "dry_run": bool(args.dry_run),
    }

    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
