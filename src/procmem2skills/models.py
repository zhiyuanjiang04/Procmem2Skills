from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

JsonDict = dict[str, Any]


class BenchmarkKind(str, Enum):
    WEB_ARENA = "webarena"
    MIND2WEB = "mind2web"
    ALFWORLD = "alfworld"
    TERMINAL_BENCH = "terminal-bench"
    UNKNOWN = "unknown"


class ExecutionMode(str, Enum):
    OFFLINE_BOOTSTRAP = "offline-bootstrap"
    ONLINE_INCREMENTAL = "online-incremental"


class BoundaryReason(str, Enum):
    TOOL_SWITCH = "tool-switch"
    SUCCESS_SIGNAL = "success-signal"
    FILESET_CHANGE = "fileset-change"
    MAX_EVENTS = "max-events"


class Observation(BaseModel):
    summary: str = ""
    text: str | None = None
    structured: JsonDict = Field(default_factory=dict)


class Action(BaseModel):
    tool: str
    name: str
    arguments: JsonDict = Field(default_factory=dict)
    raw: str | None = None


class ExecutionResult(BaseModel):
    ok: bool = True
    output_text: str | None = None
    exit_code: int | None = None
    metadata: JsonDict = Field(default_factory=dict)


class ArtifactRef(BaseModel):
    kind: str
    path: str | None = None
    description: str | None = None
    metadata: JsonDict = Field(default_factory=dict)


class Event(BaseModel):
    step_id: int
    timestamp: str | None = None
    observation: Observation = Field(default_factory=Observation)
    thought: str | None = None
    action: Action | None = None
    result: ExecutionResult | None = None
    state_delta: JsonDict = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    success_signal: str | None = None


class Trajectory(BaseModel):
    episode_id: str
    benchmark: BenchmarkKind = BenchmarkKind.UNKNOWN
    harness: str
    agent: str
    task_id: str
    instruction: str
    mode: ExecutionMode = ExecutionMode.OFFLINE_BOOTSTRAP
    metadata: JsonDict = Field(default_factory=dict)
    events: list[Event] = Field(default_factory=list)
    completed: bool = False
    score: float | None = None


class Segment(BaseModel):
    segment_id: str
    episode_id: str
    start_step: int
    end_step: int
    reasons: list[BoundaryReason] = Field(default_factory=list)
    tool_sequence: list[str] = Field(default_factory=list)
    summary_hint: str | None = None
    events: list[Event] = Field(default_factory=list)


class WorkflowStep(BaseModel):
    order: int
    intent: str
    tool: str | None = None
    operation: str
    preconditions: list[str] = Field(default_factory=list)
    verification: str | None = None


class WorkflowCandidate(BaseModel):
    workflow_id: str
    source_segment_id: str
    objective: str
    trigger: str
    preconditions: list[str] = Field(default_factory=list)
    steps: list[WorkflowStep] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    fingerprint: str
    metadata: JsonDict = Field(default_factory=dict)


class WorkflowCluster(BaseModel):
    cluster_id: str
    member_workflow_ids: list[str] = Field(default_factory=list)
    canonical_workflow_ids: list[str] = Field(default_factory=list)
    support: int = 0
    benchmark_origins: list[str] = Field(default_factory=list)
    harness_origins: list[str] = Field(default_factory=list)
    agent_origins: list[str] = Field(default_factory=list)
    task_origins: list[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)


class AtomicSkill(BaseModel):
    skill_id: str
    title: str
    description: str
    canonical_key: str
    trigger: str
    preconditions: list[str] = Field(default_factory=list)
    actions: list[WorkflowStep] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    failure_recovery: list[str] = Field(default_factory=list)
    benchmark_origins: list[str] = Field(default_factory=list)
    harness_origins: list[str] = Field(default_factory=list)
    agent_origins: list[str] = Field(default_factory=list)
    task_origins: list[str] = Field(default_factory=list)
    source_workflow_ids: list[str] = Field(default_factory=list)
    support: int = 1
    metadata: JsonDict = Field(default_factory=dict)
