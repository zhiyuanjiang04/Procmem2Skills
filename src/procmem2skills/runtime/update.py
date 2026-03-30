from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from procmem2skills.analysis.failure import build_failure_analysis_from_trajectories
from procmem2skills.evaluation.pipeline import SkillDistillationPipeline
from procmem2skills.models import Trajectory
from procmem2skills.packager.materialize import materialize_skill_repository_standard_llm
from procmem2skills.recorder.jsonl import append_trajectory, load_trajectories


class OnlineUpdateConfig(BaseModel):
    archive_path: Path
    repository_dir: Path
    min_support: int = 1
    min_score: float = 0.0
    similarity_threshold: float = 0.34
    structure_threshold: float = 0.5
    cluster_backend: str = "auto"
    cluster_embedding_model: str | None = None
    cluster_embedding_base_url: str | None = None
    cluster_dbscan_eps: float = 0.35
    cluster_dbscan_min_samples: int = 2
    cluster_embedding_strict: bool = False
    workflow_aggregation_mode: str = "global"
    per_task_skill_namespace: bool = True
    skill_creator_model: str | None = None
    skill_creator_base_url: str | None = None
    skill_creator_agent_style: str = "codex"
    skill_creator_system_prompt: str | None = None


class OnlineSkillUpdater:
    def __init__(self, config: OnlineUpdateConfig) -> None:
        self.config = config
        self.pipeline = SkillDistillationPipeline(
            min_support=config.min_support,
            similarity_threshold=config.similarity_threshold,
            structure_threshold=config.structure_threshold,
            cluster_backend=config.cluster_backend,
            cluster_embedding_model=config.cluster_embedding_model,
            cluster_embedding_base_url=config.cluster_embedding_base_url,
            cluster_dbscan_eps=config.cluster_dbscan_eps,
            cluster_dbscan_min_samples=config.cluster_dbscan_min_samples,
            cluster_embedding_strict=config.cluster_embedding_strict,
            workflow_aggregation_mode=config.workflow_aggregation_mode,
            per_task_skill_namespace=config.per_task_skill_namespace,
        )

    def ingest(self, trajectory: Trajectory) -> list[Path]:
        if not trajectory.completed:
            return []
        if trajectory.score is not None and trajectory.score < self.config.min_score:
            return []
        append_trajectory(self.config.archive_path, trajectory)
        all_trajectories = load_trajectories(self.config.archive_path)
        failure_analysis = build_failure_analysis_from_trajectories(all_trajectories)
        result = self.pipeline.distill(all_trajectories)
        written, _ = materialize_skill_repository_standard_llm(
            skills=result.skills,
            output_dir=self.config.repository_dir,
            model=self.config.skill_creator_model,
            base_url=self.config.skill_creator_base_url,
            skill_creator_agent_style=self.config.skill_creator_agent_style,
            skill_creator_system_prompt=self.config.skill_creator_system_prompt,
            failure_analysis=failure_analysis.get("global"),
            failure_analysis_by_task=failure_analysis.get("by_task"),
        )
        return written
