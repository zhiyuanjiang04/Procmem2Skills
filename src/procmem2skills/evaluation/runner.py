from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from procmem2skills.adapters.base import BenchmarkAdapter, TaskDescriptor
from procmem2skills.models import Action, Observation, Trajectory
from procmem2skills.recorder.live import LiveTrajectoryRecorder
from procmem2skills.runtime.retrieval import SkillBundle, SkillIndex, SkillSearchHit


class PolicyDecision(BaseModel):
    action: Action
    thought: str | None = None
    retrieved_skills: list[str] = Field(default_factory=list)


class ActionPolicy(Protocol):
    def choose_action(
        self,
        *,
        task: TaskDescriptor,
        observation: Observation,
        retrieved_skills: list[SkillBundle],
        step_index: int,
        trajectory_so_far: list,
    ) -> PolicyDecision:
        ...


@dataclass
class LiveRunResult:
    trajectory: Trajectory
    retrieved_hits: list[list[SkillSearchHit]] = field(default_factory=list)


class LiveRunner:
    def __init__(self, skill_repository: Path | None = None, top_k_skills: int = 3, max_steps: int = 20) -> None:
        self.skill_repository = skill_repository
        self.top_k_skills = top_k_skills
        self.max_steps = max_steps
        self.skill_index = SkillIndex.from_repository(skill_repository) if skill_repository else SkillIndex({})

    def run(self, adapter: BenchmarkAdapter, policy: ActionPolicy, episode_id: str | None = None) -> LiveRunResult:
        task = adapter.task()
        initial_observation = adapter.reset(task.task_id)
        recorder = LiveTrajectoryRecorder(
            benchmark=adapter.profile.benchmark.value,
            harness=adapter.profile.harness,
            agent=policy.__class__.__name__,
            task=task,
            initial_observation=initial_observation,
            metadata={"adapter_mode": adapter.profile.mode.value},
        )

        observation = initial_observation
        retrieval_trace: list[list[SkillSearchHit]] = []
        completed = False
        for step_index in range(1, self.max_steps + 1):
            bundles, hits = self._retrieve(task, observation)
            retrieval_trace.append(hits)
            decision = policy.choose_action(
                task=task,
                observation=observation,
                retrieved_skills=bundles,
                step_index=step_index,
                trajectory_so_far=recorder.events,
            )
            result = adapter.step(decision.action)
            recorder.record_step(decision.action, result, thought=decision.thought)
            observation = result.observation
            if result.done or adapter.is_done():
                completed = True
                break

        score_payload = adapter.score()
        score = _extract_score(score_payload, completed)
        trajectory = recorder.finalize(
            episode_id=episode_id or f"{task.task_id}-{uuid4().hex[:8]}",
            completed=completed,
            score=score,
            metadata={"score_payload": score_payload, "exported_artifacts": adapter.export_artifacts()},
        )
        return LiveRunResult(trajectory=trajectory, retrieved_hits=retrieval_trace)

    def _retrieve(self, task: TaskDescriptor, observation: Observation) -> tuple[list[SkillBundle], list[SkillSearchHit]]:
        if not self.skill_index.records:
            return [], []
        query = " ".join(part for part in [task.instruction, observation.summary, observation.text or ""] if part)
        hits = self.skill_index.search(query, top_k=self.top_k_skills)
        bundles = [self.skill_index.load_bundle(hit.skill_id) for hit in hits]
        return bundles, hits


def _extract_score(payload: dict, completed: bool) -> float | None:
    if "score" in payload and isinstance(payload["score"], (int, float)):
        return float(payload["score"])
    if "reward" in payload and isinstance(payload["reward"], (int, float)):
        return float(payload["reward"])
    return 1.0 if completed else 0.0
