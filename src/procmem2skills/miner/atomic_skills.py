from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections import defaultdict

from procmem2skills.models import AtomicSkill, Trajectory, WorkflowCandidate, WorkflowCluster, WorkflowStep
from procmem2skills.normalization import compact_text, condition_key, generic_trigger_phrase, is_informative_signal, operation_family


class AtomicSkillMiner:
    def __init__(self, min_support: int = 1) -> None:
        self.min_support = min_support

    def mine(
        self,
        workflows: list[WorkflowCandidate],
        trajectories: list[Trajectory],
        clusters: list[WorkflowCluster] | None = None,
    ) -> list[AtomicSkill]:
        trajectory_by_episode = {trajectory.episode_id: trajectory for trajectory in trajectories}
        cluster_by_workflow_id = {
            workflow_id: cluster
            for cluster in clusters or []
            for workflow_id in cluster.member_workflow_ids
        }
        buckets: dict[str, list[tuple[WorkflowCandidate, WorkflowStep, Trajectory | None]]] = defaultdict(list)
        for workflow in workflows:
            episode_id = workflow.source_segment_id.split("-seg-", 1)[0]
            trajectory = trajectory_by_episode.get(episode_id)
            for step in workflow.steps:
                buckets[canonical_key(step)].append((workflow, step, trajectory))

        skills = []
        for key, members in buckets.items():
            if len(members) < self.min_support:
                continue
            exemplar_workflow, exemplar_step, _ = members[0]
            benchmarks = sorted(
                {
                    trajectory.benchmark.value
                    for _, _, trajectory in members
                    if trajectory is not None
                }
            )
            harnesses = sorted({trajectory.harness for _, _, trajectory in members if trajectory is not None})
            agents = sorted({trajectory.agent for _, _, trajectory in members if trajectory is not None})
            tasks = sorted({trajectory.task_id for _, _, trajectory in members if trajectory is not None})
            verifications = _dedupe([step.verification for _, step, _ in members if step.verification])
            preconditions = _rank_text_signals(
                condition
                for workflow, step, _ in members
                for condition in _candidate_preconditions(workflow, step)
            )
            failure_recovery = _rank_text_signals(
                failure
                for workflow, _, _ in members
                for failure in workflow.failure_modes
            )
            verifications = _rank_text_signals(verifications, max_items=3)
            support = len(members)
            title = _title_from_key(key)
            cluster_ids = sorted(
                {
                    cluster_by_workflow_id[workflow.workflow_id].cluster_id
                    for workflow, _, _ in members
                    if workflow.workflow_id in cluster_by_workflow_id
                }
            )
            skills.append(
                AtomicSkill(
                    skill_id=key,
                    title=title,
                    description=_description(exemplar_step, support),
                    canonical_key=key,
                    trigger=_trigger_from_step(exemplar_step),
                    preconditions=preconditions,
                    actions=[
                        WorkflowStep(
                            order=1,
                            intent=exemplar_step.intent,
                            tool=exemplar_step.tool,
                            operation=exemplar_step.operation,
                            preconditions=list(exemplar_step.preconditions),
                            verification=exemplar_step.verification,
                        )
                    ],
                    verification=verifications,
                    failure_recovery=failure_recovery,
                    benchmark_origins=benchmarks,
                    harness_origins=harnesses,
                    agent_origins=agents,
                    task_origins=tasks,
                    source_workflow_ids=sorted({workflow.workflow_id for workflow, _, _ in members}),
                    support=support,
                    metadata={
                        "objective": exemplar_workflow.objective,
                        "benchmark_count": len(benchmarks),
                        "harness_count": len(harnesses),
                        "agent_count": len(agents),
                        "cluster_ids": cluster_ids,
                    },
                )
            )
        return sorted(skills, key=lambda skill: (-skill.support, skill.skill_id))


def canonical_key(step: WorkflowStep) -> str:
    base = operation_family(step.tool, step.operation)
    base = re.sub(r"-{2,}", "-", base)
    if len(base) <= 64:
        return base
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    return f"{base[:53].rstrip('-')}-{digest}"


def _title_from_key(key: str) -> str:
    return key.replace("-", " ").title()


def _description(step: WorkflowStep, support: int) -> str:
    tool = step.tool or "tool"
    family = operation_family(step.tool, step.operation).split("-", 1)[-1]
    return f"Reusable atomic skill for {tool} `{family}` actions derived from {support} matching workflow steps."


def _trigger_from_step(step: WorkflowStep) -> str:
    return generic_trigger_phrase(step.tool, step.operation)


def _rank_text_signals(items, max_items: int = 4) -> list[str]:
    counts = Counter()
    exemplars: dict[str, str] = {}
    for item in items:
        if not item:
            continue
        text = compact_text(item, limit=120)
        if not is_informative_signal(text):
            continue
        key = condition_key(text)
        if not key:
            continue
        counts[key] += 1
        exemplar = exemplars.get(key)
        if exemplar is None or len(text) < len(exemplar):
            exemplars[key] = text

    ranked = sorted(
        counts.items(),
        key=lambda entry: (-entry[1], len(exemplars[entry[0]]), exemplars[entry[0]]),
    )
    return [exemplars[key] for key, _ in ranked[:max_items]]


def _dedupe(items) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if not item or item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


def _candidate_preconditions(workflow: WorkflowCandidate, step: WorkflowStep) -> list[str]:
    if step.preconditions:
        return step.preconditions
    if len(workflow.steps) == 1:
        return workflow.preconditions
    return []
