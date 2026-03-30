from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from procmem2skills.inducer.workflow import induce_workflow
from procmem2skills.miner.atomic_skills import AtomicSkillMiner
from procmem2skills.miner.clustering import WorkflowClusterer
from procmem2skills.models import AtomicSkill, ExecutionMode, Segment, Trajectory, WorkflowCandidate, WorkflowCluster
from procmem2skills.segmenter.heuristics import segment_trajectory


class DistillationResult(BaseModel):
    trajectories: list[Trajectory] = Field(default_factory=list)
    successful_trajectories: list[Trajectory] = Field(default_factory=list)
    failed_trajectories: list[Trajectory] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    workflows: list[WorkflowCandidate] = Field(default_factory=list)
    clusters: list[WorkflowCluster] = Field(default_factory=list)
    skills: list[AtomicSkill] = Field(default_factory=list)


class SkillDistillationPipeline:
    def __init__(
        self,
        min_support: int = 1,
        similarity_threshold: float = 0.34,
        structure_threshold: float = 0.5,
        cluster_backend: str = "auto",
        cluster_embedding_model: str | None = None,
        cluster_embedding_base_url: str | None = None,
        cluster_dbscan_eps: float = 0.35,
        cluster_dbscan_min_samples: int = 2,
        workflow_aggregation_mode: str = "global",
        per_task_skill_namespace: bool = True,
        cluster_embedding_strict: bool = False,
    ) -> None:
        self.workflow_aggregation_mode = _resolve_workflow_aggregation_mode(workflow_aggregation_mode)
        self.per_task_skill_namespace = bool(per_task_skill_namespace)

        effective_cluster_backend = cluster_backend
        effective_embedding_model = cluster_embedding_model
        effective_embedding_strict = bool(cluster_embedding_strict)
        if self.workflow_aggregation_mode == "global-dbscan-qwen":
            effective_cluster_backend = "embedding-dbscan"
            if not effective_embedding_model:
                effective_embedding_model = "Qwen/Qwen3-Embedding-0.6B"
            effective_embedding_strict = True

        self.miner = AtomicSkillMiner(min_support=min_support)
        self.clusterer = WorkflowClusterer(
            similarity_threshold=similarity_threshold,
            structure_threshold=structure_threshold,
            cluster_backend=effective_cluster_backend,
            embedding_model=effective_embedding_model,
            embedding_base_url=cluster_embedding_base_url,
            dbscan_eps=cluster_dbscan_eps,
            dbscan_min_samples=cluster_dbscan_min_samples,
            embedding_strict=effective_embedding_strict,
        )

    def distill(self, trajectories: list[Trajectory]) -> DistillationResult:
        successful_trajectories = [trajectory for trajectory in trajectories if _trajectory_is_success(trajectory)]
        success_ids = {id(trajectory) for trajectory in successful_trajectories}
        failed_trajectories = [trajectory for trajectory in trajectories if id(trajectory) not in success_ids]
        source_trajectories = successful_trajectories or trajectories
        segments = []
        workflows = []
        for trajectory in source_trajectories:
            segments.extend(segment_trajectory(trajectory))
        for segment in segments:
            workflows.append(induce_workflow(segment))

        if self.workflow_aggregation_mode == "per-task":
            deduped_workflows, clusters, skills = self._distill_per_task(workflows, source_trajectories)
        else:
            clusters = self.clusterer.cluster(workflows, source_trajectories)
            deduped_workflows = self.clusterer.dedupe_workflows(workflows, clusters)
            skills = self.miner.mine(deduped_workflows, source_trajectories, clusters=clusters)
            skills = [_annotate_skill(skill, aggregation_scope=self.workflow_aggregation_mode) for skill in skills]

        return DistillationResult(
            trajectories=trajectories,
            successful_trajectories=successful_trajectories,
            failed_trajectories=failed_trajectories,
            segments=segments,
            workflows=deduped_workflows,
            clusters=clusters,
            skills=skills,
        )

    def _distill_per_task(
        self,
        workflows: list[WorkflowCandidate],
        source_trajectories: list[Trajectory],
    ) -> tuple[list[WorkflowCandidate], list[WorkflowCluster], list[AtomicSkill]]:
        if not workflows:
            return [], [], []

        trajectory_by_episode = {trajectory.episode_id: trajectory for trajectory in source_trajectories}
        trajectories_by_task: dict[str, list[Trajectory]] = defaultdict(list)
        for trajectory in source_trajectories:
            task_key = (trajectory.task_id or "unknown-task").strip() or "unknown-task"
            trajectories_by_task[task_key].append(trajectory)

        workflows_by_task: dict[str, list[WorkflowCandidate]] = defaultdict(list)
        for workflow in workflows:
            episode_id = workflow.source_segment_id.split("-seg-", 1)[0]
            trajectory = trajectory_by_episode.get(episode_id)
            task_key = "unknown-task"
            if trajectory is not None:
                task_key = (trajectory.task_id or "unknown-task").strip() or "unknown-task"
            workflows_by_task[task_key].append(workflow)

        all_clusters: list[WorkflowCluster] = []
        all_deduped_workflows: list[WorkflowCandidate] = []
        all_skills: list[AtomicSkill] = []

        for task_key in sorted(workflows_by_task.keys()):
            task_workflows = workflows_by_task[task_key]
            task_trajectories = trajectories_by_task.get(task_key, [])
            task_clusters = self.clusterer.cluster(task_workflows, task_trajectories)
            task_slug = _slugify_task(task_key)
            namespaced_clusters = [_namespace_cluster(cluster, task_slug=task_slug, task_key=task_key) for cluster in task_clusters]
            task_deduped = self.clusterer.dedupe_workflows(task_workflows, namespaced_clusters)
            task_skills = self.miner.mine(task_deduped, task_trajectories, clusters=namespaced_clusters)

            for skill in task_skills:
                if self.per_task_skill_namespace:
                    skill = _namespace_task_skill(skill, task_slug=task_slug)
                all_skills.append(
                    _annotate_skill(
                        skill,
                        aggregation_scope="per-task",
                        source_task=task_key,
                    )
                )

            all_clusters.extend(namespaced_clusters)
            all_deduped_workflows.extend(task_deduped)

        return all_deduped_workflows, all_clusters, all_skills


class EvaluationPlan(BaseModel):
    mode: ExecutionMode
    trajectory_path: Path
    skill_repository: Path
    min_support: int = 1


def _trajectory_is_success(trajectory: Trajectory) -> bool:
    if trajectory.score is not None:
        return trajectory.score >= 1.0
    if trajectory.completed:
        return True
    return False


def _resolve_workflow_aggregation_mode(value: str) -> str:
    normalized = (value or "global").strip().lower()
    aliases = {
        "global-dbscan": "global",
        "global_dbscan": "global",
        "global-dbscan-qwen": "global-dbscan-qwen",
        "global_dbscan_qwen": "global-dbscan-qwen",
        "per-task": "per-task",
        "per_task": "per-task",
        "task": "per-task",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in {"global", "per-task", "global-dbscan-qwen"}:
        raise ValueError(
            "unsupported workflow aggregation mode: "
            f"{value} (expected one of: global, per-task, global-dbscan-qwen)"
        )
    return resolved


def _slugify_task(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return text or "task"


def _namespace_cluster(cluster: WorkflowCluster, *, task_slug: str, task_key: str) -> WorkflowCluster:
    namespaced = cluster.model_copy(deep=True)
    namespaced.cluster_id = f"task-{task_slug}-{cluster.cluster_id}"
    namespaced.metadata = {
        **(namespaced.metadata or {}),
        "aggregation_scope": "per-task",
        "source_task": task_key,
    }
    return namespaced


def _namespace_task_skill(skill: AtomicSkill, *, task_slug: str) -> AtomicSkill:
    return skill.model_copy(update={"skill_id": f"{task_slug}--{skill.skill_id}"})


def _annotate_skill(skill: AtomicSkill, *, aggregation_scope: str, source_task: str | None = None) -> AtomicSkill:
    metadata = dict(skill.metadata or {})
    metadata["aggregation_scope"] = aggregation_scope
    if source_task:
        metadata["source_task"] = source_task
    return skill.model_copy(update={"metadata": metadata})
