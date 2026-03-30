from __future__ import annotations

from procmem2skills.adapters.base import AdapterMode, AdapterProfile
from procmem2skills.models import Action, BenchmarkKind, Event, ExecutionResult, Observation

PROFILE = AdapterProfile(
    benchmark=BenchmarkKind.ALFWORLD,
    harness="alfworld/textworld",
    mode=AdapterMode.INTERACTIVE,
    observation_modalities=["text-description", "inventory"],
    action_modalities=["text-command"],
    evaluation_style="episode success with intermediate rewards",
    notes="Best suited for developing atomic-skill segmentation because subgoals are structured and low-noise.",
)


def normalize_alfworld_step(raw_step: dict, step_id: int) -> Event:
    action_text = raw_step.get("action") or "look"
    return Event(
        step_id=step_id,
        observation=Observation(
            summary=raw_step.get("summary", ""),
            text=raw_step.get("observation"),
            structured={"inventory": raw_step.get("inventory")},
        ),
        action=Action(
            tool="text-world",
            name=action_text.split(" ", 1)[0],
            arguments={"command": action_text},
            raw=action_text,
        ),
        result=ExecutionResult(
            ok=bool(raw_step.get("ok", True)),
            output_text=raw_step.get("feedback"),
            metadata={"reward": raw_step.get("reward")},
        ),
        success_signal=raw_step.get("success_signal"),
    )
