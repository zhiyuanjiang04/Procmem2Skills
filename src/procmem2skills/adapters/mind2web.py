from __future__ import annotations

import json

from procmem2skills.adapters.base import AdapterMode, AdapterProfile
from procmem2skills.models import Action, ArtifactRef, BenchmarkKind, Event, ExecutionResult, Observation

PROFILE = AdapterProfile(
    benchmark=BenchmarkKind.MIND2WEB,
    harness="mind2web/replay",
    mode=AdapterMode.REPLAY,
    observation_modalities=["html", "candidate-elements", "trace", "snapshot"],
    action_modalities=["click", "type", "select", "navigate"],
    evaluation_style="step labels and task-grounded replay analysis",
    notes="Best suited for offline workflow induction and cross-website generalization studies.",
)


def normalize_mind2web_step(raw_step: dict, step_id: int) -> Event:
    operation = raw_step.get("operation") or {}
    if isinstance(operation, str):
        action_name = operation
        arguments = {}
    else:
        action_name = operation.get("op") or operation.get("operation") or "web-action"
        arguments = {
            "value": operation.get("value"),
            "element": _extract_element_reference(raw_step),
        }
    artifacts = []
    for key in ("screenshot", "snapshot", "trace"):
        value = raw_step.get(key)
        if value:
            artifacts.append(ArtifactRef(kind=key, path=str(value)))
    return Event(
        step_id=step_id,
        observation=Observation(
            summary=raw_step.get("action_repr", ""),
            text=raw_step.get("cleaned_html"),
            structured={
                "raw_html": raw_step.get("raw_html"),
                "positive_candidates": raw_step.get("pos_candidates"),
                "negative_candidates": raw_step.get("neg_candidates"),
            },
        ),
        action=Action(
            tool="browser",
            name=action_name,
            arguments={key: value for key, value in arguments.items() if value is not None},
            raw=raw_step.get("action_uid"),
        ),
        result=ExecutionResult(
            ok=True,
            output_text=raw_step.get("annotation_id"),
            metadata={"website": raw_step.get("website")},
        ),
        artifacts=artifacts,
    )


def _extract_element_reference(raw_step: dict):
    target = raw_step.get("target")
    if target is not None:
        return _summarize_target(target)
    candidates = raw_step.get("pos_candidates")
    if isinstance(candidates, list) and candidates:
        return _summarize_target(candidates[0])
    if isinstance(candidates, dict):
        return _summarize_target(candidates)
    return None


def _summarize_target(target):
    if isinstance(target, str):
        return _clip_text(target)
    if isinstance(target, list):
        return [_summarize_target(item) for item in target[:3]]
    if not isinstance(target, dict):
        return target
    summary = {}
    for key in ("selector", "backend_node_id", "node_id", "tag", "role"):
        value = target.get(key)
        if value is not None:
            summary[key] = value
    text_value = target.get("text") or target.get("value") or target.get("inner_text") or target.get("label")
    if text_value:
        summary["text"] = _clip_text(str(text_value))
    if not summary:
        compact = json.dumps(target, sort_keys=True, ensure_ascii=True)
        return _clip_text(compact)
    return summary


def _clip_text(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
