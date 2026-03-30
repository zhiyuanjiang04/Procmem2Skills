from __future__ import annotations

from dataclasses import dataclass, field

from procmem2skills.adapters.base import StepResult, TaskDescriptor
from procmem2skills.models import Action, Event, ExecutionMode, ExecutionResult, Observation, Trajectory


@dataclass
class LiveTrajectoryRecorder:
    benchmark: str
    harness: str
    agent: str
    task: TaskDescriptor
    initial_observation: Observation
    metadata: dict = field(default_factory=dict)
    _events: list[Event] = field(default_factory=list, init=False)
    _step_id: int = field(default=0, init=False)

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def record_step(self, action: Action, result: StepResult, thought: str | None = None) -> None:
        self._step_id += 1
        self._events.append(
            Event(
                step_id=self._step_id,
                observation=result.observation,
                thought=thought,
                action=action,
                result=ExecutionResult(
                    ok=not bool(result.info.get("error")),
                    output_text=result.info.get("message"),
                    exit_code=result.info.get("exit_code"),
                    metadata=dict(result.info),
                ),
                state_delta=result.info.get("state_delta") or {},
                artifacts=list(result.artifacts),
                success_signal=result.success_signal,
            )
        )

    def finalize(self, episode_id: str, completed: bool, score: float | None = None, metadata: dict | None = None) -> Trajectory:
        combined_metadata = {**self.metadata, **(metadata or {})}
        benchmark_value = self.benchmark
        return Trajectory(
            episode_id=episode_id,
            benchmark=benchmark_value,
            harness=self.harness,
            agent=self.agent,
            task_id=self.task.task_id,
            instruction=self.task.instruction,
            mode=ExecutionMode.ONLINE_INCREMENTAL,
            metadata=combined_metadata,
            events=list(self._events),
            completed=completed,
            score=score,
        )
