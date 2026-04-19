from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Strategy = Literal["random", "easy_neg", "hard_neg_semantic", "hard_neg_functional", "adversarial"]
Representation = Literal["card", "name_only", "desc_only", "full", "compressed_full"]
Probe = Literal["awareness", "selection"]
FormatStatus = Literal["clean", "warning", "fail"]


class PoolSpec(BaseModel):
    task_id: str
    strategy: Strategy
    n: int = Field(ge=1)
    seed: int = Field(ge=0)
    representation: Representation = "card"

    @property
    def pool_id(self) -> str:
        return f"{self.task_id}__{self.strategy}__n{self.n}__s{self.seed}__{self.representation}"


class TrialRecord(BaseModel):
    pool_id: str
    probe: Probe
    model: str
    raw_response: str
    extracted_ids: list[str]
    format_status: FormatStatus
    flags: dict = Field(default_factory=dict)
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None


class RunConfig(BaseModel):
    label: str
    model: str = "claude-sonnet-4-6"
    task_ids: list[str]
    strategies: list[Strategy]
    pool_sizes: list[int]
    representation: Representation = "card"
    seeds: list[int] = [0, 1, 2]
    probes: list[Probe] = ["awareness", "selection"]
    temperature: float = 0.0
    max_concurrency: int = 8
    context_safety_margin: int = 1000
