from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from json import JSONDecodeError, JSONDecoder
from typing import Any

from procmem2skills.models import Segment, Trajectory, WorkflowCandidate, WorkflowStep

_SYSTEM_PROMPT = """You are a workflow induction engine.

Given one trajectory segment, output exactly one JSON object with these keys:
- objective
- trigger
- preconditions
- steps
- verification
- failure_modes

`steps` must be a non-empty list when the segment contains actionable events.
Each step must be an object with keys:
- order
- intent
- tool
- operation
- preconditions
- verification

Rules:
- Cover all actionable events in the input segment.
- Do not invent tools/commands/actions not grounded in the input.
- Keep wording concise and operational.
- Output strict JSON only; no markdown.
"""


class LLMWorkflowInducer:
    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_sec: int = 120,
        max_retries: int = 1,
        strict: bool = True,
    ) -> None:
        self.model = (
            model
            or os.environ.get("PROCMEM_WORKFLOW_INDUCER_MODEL")
            or os.environ.get("OPENROUTER_MODEL")
            or "openai/gpt-5.3-codex"
        )
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self.timeout_sec = max(1, int(timeout_sec))
        self.max_retries = max(0, int(max_retries))
        self.strict = bool(strict)

    def induce(
        self,
        *,
        segment: Segment,
        trajectory: Trajectory | None = None,
        attempt_status: str | None = None,
        base_metadata: dict[str, Any] | None = None,
    ) -> WorkflowCandidate:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY or OPENAI_API_KEY is required for LLM workflow induction")

        user_prompt = _build_user_prompt(segment=segment, trajectory=trajectory, attempt_status=attempt_status)
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            raw_response = self._chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            try:
                payload = _parse_json_payload(raw_response)
                workflow = _workflow_from_payload(
                    payload,
                    segment=segment,
                    strict=self.strict,
                )
            except Exception as exc:
                last_error = exc
                continue

            metadata: dict[str, Any] = {}
            if isinstance(base_metadata, dict):
                metadata.update(base_metadata)
            metadata["llm_induction"] = {
                "enabled": True,
                "model": self.model,
                "strict": self.strict,
                "segment_id": segment.segment_id,
                "attempt_status": attempt_status,
                "raw_response": raw_response,
            }
            workflow.metadata = metadata
            return workflow

        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM workflow induction failed without a concrete error")

    def _chat(self, *, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM workflow induction request failed with status {exc.code}: {detail}") from exc

        message = ((body.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        if isinstance(message, list):
            return "".join(part.get("text", "") for part in message if isinstance(part, dict))
        return str(message or "")


def _build_user_prompt(*, segment: Segment, trajectory: Trajectory | None, attempt_status: str | None) -> str:
    trajectory_payload = {}
    if trajectory is not None:
        trajectory_payload = {
            "episode_id": trajectory.episode_id,
            "task_id": trajectory.task_id,
            "instruction": trajectory.instruction,
            "benchmark": trajectory.benchmark.value,
            "harness": trajectory.harness,
            "agent": trajectory.agent,
            "score": trajectory.score,
            "completed": trajectory.completed,
            "status": attempt_status,
        }
    segment_payload = segment.model_dump(mode="json")

    return (
        "Induce one reusable workflow for the segment below.\n\n"
        "Trajectory context:\n"
        + json.dumps(trajectory_payload, indent=2, ensure_ascii=False)
        + "\n\nSegment JSON:\n"
        + json.dumps(segment_payload, indent=2, ensure_ascii=False)
    )


def _workflow_from_payload(payload: dict[str, Any], *, segment: Segment, strict: bool) -> WorkflowCandidate:
    root = payload.get("workflow") if isinstance(payload.get("workflow"), dict) else payload
    if not isinstance(root, dict):
        raise ValueError("LLM workflow payload must be a JSON object")

    objective = str(root.get("objective") or "").strip()
    trigger = str(root.get("trigger") or "").strip()
    preconditions = _to_string_list(root.get("preconditions"))
    verification = _to_string_list(root.get("verification"))
    failure_modes = _to_string_list(root.get("failure_modes"))

    raw_steps = root.get("steps") if isinstance(root.get("steps"), list) else []
    steps: list[WorkflowStep] = []
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            continue
        order_value = raw_step.get("order")
        try:
            order = int(order_value)
        except Exception:
            order = index
        intent = str(raw_step.get("intent") or "").strip()
        tool = raw_step.get("tool")
        tool_text = str(tool).strip() if tool is not None else None
        operation = str(raw_step.get("operation") or "").strip()
        step_preconditions = _to_string_list(raw_step.get("preconditions"))
        step_verification_raw = raw_step.get("verification")
        step_verification = str(step_verification_raw).strip() if step_verification_raw is not None else None

        if not operation:
            continue
        steps.append(
            WorkflowStep(
                order=max(1, order),
                intent=intent or operation,
                tool=tool_text or None,
                operation=operation,
                preconditions=step_preconditions,
                verification=step_verification or None,
            )
        )

    steps = sorted(steps, key=lambda item: (item.order, item.operation))
    for index, step in enumerate(steps, start=1):
        step.order = index

    action_event_count = sum(1 for event in segment.events if event.action is not None)
    if strict and action_event_count > 0 and len(steps) < action_event_count:
        raise ValueError(
            f"LLM workflow did not cover enough action events: expected>={action_event_count}, got={len(steps)}"
        )

    if strict and not objective:
        raise ValueError("LLM workflow payload missing objective")
    if strict and action_event_count > 0 and not steps:
        raise ValueError("LLM workflow payload missing steps for actionable segment")

    if not objective:
        objective = segment.summary_hint or "complete subtask"
    if not trigger:
        trigger = f"When the agent needs to {objective}."

    return WorkflowCandidate(
        workflow_id=f"{segment.segment_id}-wf",
        source_segment_id=segment.segment_id,
        objective=objective,
        trigger=trigger,
        preconditions=preconditions,
        steps=steps,
        verification=verification,
        failure_modes=failure_modes,
        fingerprint=_fingerprint(steps),
        metadata={},
    )


def _to_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return _dedupe(items)
    text = str(value).strip()
    return [text] if text else []


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def _fingerprint(steps: list[WorkflowStep]) -> str:
    basis = "|".join(f"{step.tool}:{step.operation}" for step in steps)
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]


def _parse_json_payload(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    candidates = [cleaned]

    fenced = _strip_code_fence(cleaned)
    if fenced != cleaned:
        candidates.append(fenced)

    extracted = _extract_first_json_object(cleaned)
    if extracted is not None:
        candidates.append(extracted)

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    raise ValueError(f"could not parse LLM workflow response as JSON: {text}")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_first_json_object(text: str) -> str | None:
    decoder = JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, end = decoder.raw_decode(text[index:])
        except JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return text[index : index + end]
    return None
