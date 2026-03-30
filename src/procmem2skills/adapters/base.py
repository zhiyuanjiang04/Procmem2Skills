from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, Field

from procmem2skills.models import Action, ArtifactRef, BenchmarkKind, JsonDict, Observation


class AdapterMode(str, Enum):
    INTERACTIVE = "interactive"
    REPLAY = "replay"


class AdapterCapability(str, Enum):
    HTML = "html"
    TERMINAL = "terminal"
    TEXT_WORLD = "text-world"
    FILE_DIFF = "file-diff"
    FUNCTIONAL_EVAL = "functional-eval"
    STEP_LABELS = "step-labels"


class TaskDescriptor(BaseModel):
    task_id: str
    instruction: str
    metadata: JsonDict = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: Observation = Field(default_factory=Observation)
    reward: float | None = None
    done: bool = False
    info: JsonDict = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    success_signal: str | None = None


class AdapterProfile(BaseModel):
    benchmark: BenchmarkKind
    harness: str
    mode: AdapterMode
    observation_modalities: list[str] = Field(default_factory=list)
    action_modalities: list[str] = Field(default_factory=list)
    evaluation_style: str
    notes: str


class BenchmarkAdapter(ABC):
    profile: AdapterProfile
    capabilities: frozenset[AdapterCapability]

    @abstractmethod
    def task(self) -> TaskDescriptor:
        raise NotImplementedError

    @abstractmethod
    def reset(self, task_id: str | None = None) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        raise NotImplementedError

    def is_done(self) -> bool:
        return False

    def score(self) -> JsonDict:
        return {}

    def export_artifacts(self) -> JsonDict:
        return {}
