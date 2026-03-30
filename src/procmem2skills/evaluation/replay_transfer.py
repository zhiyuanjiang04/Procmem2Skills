from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from procmem2skills.models import Trajectory
from procmem2skills.normalization import operation_family
from procmem2skills.runtime.retrieval import SkillIndex


class DivergenceReport(BaseModel):
    step_index: int
    reference_family: str
    target_family: str
    query: str
    retrieved_skill_ids: list[str] = Field(default_factory=list)
    recoverable_top_k: bool = False


class ReplayTransferReport(BaseModel):
    reference_task_id: str
    target_task_id: str
    reference_score: float | None = None
    target_score: float | None = None
    shared_prefix_steps: int = 0
    reference_unique_families: list[str] = Field(default_factory=list)
    target_unique_families: list[str] = Field(default_factory=list)
    retrieved_reference_families: list[str] = Field(default_factory=list)
    missing_reference_families: list[str] = Field(default_factory=list)
    divergence: DivergenceReport | None = None


def evaluate_replay_transfer(
    *,
    skill_repository: Path,
    reference: Trajectory,
    target: Trajectory,
    top_k: int = 5,
) -> ReplayTransferReport:
    skill_index = SkillIndex.from_repository(skill_repository)
    reference_families = _trajectory_families(reference)
    target_families = _trajectory_families(target)
    shared_prefix = 0
    for left, right in zip(reference_families, target_families):
        if left != right:
            break
        shared_prefix += 1

    retrieved_reference_families = set()
    divergence = None
    reference_family_set = set(reference_families)
    for step_index in range(len(target_families)):
        query = _query_for_next_step(target, step_index)
        hits = skill_index.search(query, top_k=top_k)
        retrieved_ids = [hit.skill_id for hit in hits]
        retrieved_families = [_base_skill_id(skill_id) for skill_id in retrieved_ids]
        retrieved_reference_families.update(
            family for family in retrieved_families if family in reference_family_set
        )
        if step_index == shared_prefix and step_index < len(reference_families):
            reference_family = reference_families[step_index]
            target_family = target_families[step_index]
            divergence = DivergenceReport(
                step_index=step_index + 1,
                reference_family=reference_family,
                target_family=target_family,
                query=query,
                retrieved_skill_ids=retrieved_ids,
                recoverable_top_k=reference_family in retrieved_families,
            )

    reference_unique = sorted(set(reference_families))
    target_unique = sorted(set(target_families))
    retrieved_unique = sorted(retrieved_reference_families)
    return ReplayTransferReport(
        reference_task_id=reference.task_id,
        target_task_id=target.task_id,
        reference_score=reference.score,
        target_score=target.score,
        shared_prefix_steps=shared_prefix,
        reference_unique_families=reference_unique,
        target_unique_families=target_unique,
        retrieved_reference_families=retrieved_unique,
        missing_reference_families=[family for family in reference_unique if family not in retrieved_reference_families],
        divergence=divergence,
    )


def _trajectory_families(trajectory: Trajectory) -> list[str]:
    families = []
    for event in trajectory.events:
        if event.action is None:
            continue
        operation = _event_operation(event.action.tool, event.action.name, event.action.arguments)
        families.append(operation_family(event.action.tool, operation))
    return families


def _query_for_next_step(trajectory: Trajectory, step_index: int) -> str:
    parts = [trajectory.instruction, trajectory.task_id]
    if step_index < len(trajectory.events):
        current = trajectory.events[step_index]
        parts.extend(
            part
            for part in [
                current.observation.summary,
                current.observation.text or "",
            ]
            if part
        )
    if step_index > 0:
        previous = trajectory.events[step_index - 1]
        parts.extend(
            part
            for part in [
                previous.observation.summary,
                previous.observation.text or "",
                previous.result.output_text if previous.result else "",
            ]
            if part
        )
    return " ".join(part for part in parts if part)


def _event_operation(tool: str, name: str, arguments: dict) -> str:
    command = arguments.get("command")
    if command:
        return f"{name}(command={command})"
    rendered = ", ".join(f"{key}={value}" for key, value in sorted(arguments.items()) if value is not None)
    return f"{name}({rendered})" if rendered else name


def _base_skill_id(skill_id: str) -> str:
    lowered = skill_id.strip().lower()
    if lowered.endswith("--success") or lowered.endswith("--failure"):
        return skill_id.rsplit("--", 1)[0]
    return skill_id
