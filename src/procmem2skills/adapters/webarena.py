from __future__ import annotations

from procmem2skills.adapters.base import AdapterMode, AdapterProfile
from procmem2skills.models import Action, ArtifactRef, BenchmarkKind, Event, ExecutionResult, Observation

PROFILE = AdapterProfile(
    benchmark=BenchmarkKind.WEB_ARENA,
    harness="browsergym/webarena",
    mode=AdapterMode.INTERACTIVE,
    observation_modalities=["dom", "a11y-tree", "url", "screenshot"],
    action_modalities=["click", "type", "select", "navigate"],
    evaluation_style="functional task completion",
    notes="Best suited for online web-agent evaluation and direct comparison against workflow-memory baselines.",
)


def normalize_webarena_step(raw_step: dict, step_id: int) -> Event:
    action_payload = raw_step.get("action") or {}
    observation_payload = raw_step.get("observation") or {}
    info = raw_step.get("info") or {}
    artifacts = []
    screenshot = observation_payload.get("screenshot")
    if screenshot:
        artifacts.append(ArtifactRef(kind="screenshot", path=str(screenshot)))
    return Event(
        step_id=step_id,
        timestamp=raw_step.get("timestamp"),
        observation=Observation(
            summary=observation_payload.get("summary", ""),
            text=observation_payload.get("url"),
            structured=observation_payload,
        ),
        action=Action(
            tool="browser",
            name=action_payload.get("name", "web-action"),
            arguments=action_payload.get("arguments", {}),
            raw=action_payload.get("raw"),
        ),
        result=ExecutionResult(
            ok=bool(info.get("ok", True)),
            output_text=info.get("message"),
            metadata=info,
        ),
        state_delta=raw_step.get("state_delta") or {},
        artifacts=artifacts,
        success_signal=raw_step.get("success_signal"),
    )
