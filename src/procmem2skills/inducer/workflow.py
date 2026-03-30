from __future__ import annotations

import hashlib
import json
from typing import Any

from procmem2skills.models import Event, Segment, WorkflowCandidate, WorkflowStep
from procmem2skills.normalization import compact_text, normalize_text, operation_family, summarize_observation, trigger_phrase

_CONTEXT_TEXT_LIMIT = 240
_CONTEXT_VALUE_LIMIT = 200


def induce_workflow(segment: Segment) -> WorkflowCandidate:
    steps: list[WorkflowStep] = []
    step_context_records: list[dict[str, Any]] = []
    actionable_order = 1
    for event in segment.events:
        if not event.action:
            continue
        operation = _render_operation(event.action.name, event.action.arguments)
        intent = _step_intent(event, operation)
        verification = event.success_signal or _verification_hint(event)
        steps.append(
            WorkflowStep(
                order=actionable_order,
                intent=intent,
                tool=event.action.tool,
                operation=operation,
                preconditions=_step_preconditions(event),
                verification=verification,
            )
        )
        step_context_records.append(
            _build_step_context_record(
                order=actionable_order,
                event=event,
                operation=operation,
                intent=intent,
                verification=verification,
            )
        )
        actionable_order += 1

    objective = _objective(segment, steps)
    trigger = trigger_phrase(
        WorkflowStep(
            order=1,
            intent=objective,
            tool=steps[0].tool if steps else None,
            operation=steps[0].operation if steps else "step",
        )
    )
    preconditions = _preconditions(segment)
    verification_points = _dedupe_normalized([step.verification for step in steps if step.verification])
    failure_modes = _failure_modes(segment)
    fingerprint = _fingerprint(steps)

    event_trace = [_build_event_trace_entry(event, compact=True) for event in segment.events]
    event_trace_full = [_build_event_trace_entry(event, compact=False) for event in segment.events]
    context_payload = {
        "objective": objective,
        "trigger": trigger,
        "preconditions": preconditions,
        "steps": step_context_records,
        "timeline": _build_timeline_records(segment.events),
        "verification": verification_points,
        "failure_modes": failure_modes,
    }

    return WorkflowCandidate(
        workflow_id=f"{segment.segment_id}-wf",
        source_segment_id=segment.segment_id,
        objective=objective,
        trigger=trigger,
        preconditions=preconditions,
        steps=steps,
        verification=verification_points,
        failure_modes=failure_modes,
        fingerprint=fingerprint,
        metadata={
            "tool_sequence": segment.tool_sequence,
            "event_trace": event_trace,
            "event_trace_full": event_trace_full,
            "context_payload": context_payload,
            "information_coverage": _information_coverage(segment.events),
            "cluster_reservation": _cluster_reservation_features(
                segment=segment,
                steps=steps,
                objective=objective,
                preconditions=preconditions,
                verification_points=verification_points,
                failure_modes=failure_modes,
            ),
        },
    )


def render_workflow_context(workflow: WorkflowCandidate, *, max_steps: int | None = None) -> str:
    payload = None
    if isinstance(workflow.metadata, dict):
        raw_payload = workflow.metadata.get("context_payload")
        if isinstance(raw_payload, dict):
            payload = dict(raw_payload)

    if payload is None:
        payload = {
            "objective": workflow.objective,
            "trigger": workflow.trigger,
            "preconditions": list(workflow.preconditions),
            "steps": [
                {
                    "index": step.order,
                    "tool": step.tool,
                    "operation": step.operation,
                    "intent": step.intent,
                    "verification": step.verification,
                }
                for step in workflow.steps
            ],
            "verification": list(workflow.verification),
            "failure_modes": list(workflow.failure_modes),
        }

    if "timeline" not in payload and isinstance(workflow.metadata, dict):
        trace = workflow.metadata.get("event_trace")
        if isinstance(trace, list):
            payload["timeline"] = _build_timeline_from_event_trace(trace)

    if max_steps is not None and max_steps > 0:
        payload["steps"] = list(payload.get("steps") or [])[:max_steps]
        payload["timeline"] = list(payload.get("timeline") or [])[: max(max_steps, 1)]
    return _render_workflow_context_payload(payload)


def _render_operation(name: str, arguments: dict) -> str:
    if not arguments:
        return name
    normalized = ", ".join(
        f"{key}={_render_argument_value(value)}"
        for key, value in sorted(arguments.items())
        if value is not None
    )
    return f"{name}({normalized})"


def _operation_token(operation: str) -> str:
    raw = (operation or "").strip()
    if not raw:
        return "operation"
    head = raw.split("(", 1)[0].strip()
    if not head:
        return "operation"
    normalized = normalize_text(head)
    return normalized or "operation"


def _objective(segment: Segment, steps: list[WorkflowStep]) -> str:
    for event in reversed(segment.events):
        if event.success_signal and event.observation.summary:
            return compact_text(event.observation.summary, limit=120)
    for event in reversed(segment.events):
        if event.observation.summary:
            return compact_text(event.observation.summary, limit=120)
        if event.thought:
            return compact_text(event.thought, limit=120)
    if steps:
        return compact_text(steps[-1].intent, limit=120)
    return compact_text(segment.summary_hint, limit=120) or "complete subtask"


def _step_intent(event: Event, operation: str) -> str:
    if event.observation.summary:
        return compact_text(event.observation.summary, limit=120)
    if event.thought:
        return compact_text(event.thought, limit=120)
    return operation


def _verification_hint(event: Event) -> str | None:
    if not event.result:
        return None

    exit_code = event.result.exit_code
    if exit_code is not None:
        try:
            if int(exit_code) != 0:
                return f"exit_code == {exit_code}"
        except Exception:
            return f"exit_code == {exit_code}"

    stderr = _compact_or_none((event.result.metadata or {}).get("stderr"), limit=120)
    if stderr:
        return f"stderr: {stderr}"

    output = _summarize_result_output(event.result.output_text, limit=120)
    if output:
        return f"output: {output}"

    if exit_code is not None:
        return f"exit_code == {exit_code}"
    return None


def _preconditions(segment: Segment) -> list[str]:
    if not segment.events:
        return []
    conditions = []
    for event in segment.events:
        conditions.extend(_step_preconditions(event))
        cwd = _compact_or_none(event.observation.text, limit=120)
        if cwd:
            conditions.append(f"Working directory hint: {cwd}")
    return _dedupe_normalized(conditions)


def _step_preconditions(event: Event) -> list[str]:
    tool = event.action.tool if event.action else None
    return summarize_observation(tool, event.observation.summary, event.observation.text)


def _build_step_context_record(
    *,
    order: int,
    event: Event,
    operation: str,
    intent: str,
    verification: str | None,
) -> dict[str, Any]:
    command = _extract_command((event.action.arguments or {}) if event.action else {})
    action_raw = _compact_or_none(event.action.raw if event.action else None, limit=_CONTEXT_TEXT_LIMIT)
    result_output = _summarize_result_output(event.result.output_text if event.result else None, limit=_CONTEXT_TEXT_LIMIT)

    record: dict[str, Any] = {
        "index": order,
        "step_id": event.step_id,
        "intent": intent,
        "tool": event.action.tool if event.action else None,
        "operation": operation,
        "command": command,
        "thought": _compact_or_none(event.thought, limit=_CONTEXT_TEXT_LIMIT),
        "observation": _compact_or_none(event.observation.summary, limit=_CONTEXT_TEXT_LIMIT),
        "cwd": _compact_or_none(event.observation.text, limit=_CONTEXT_VALUE_LIMIT),
        "stderr": _compact_or_none((event.result.metadata or {}).get("stderr") if event.result else None, limit=_CONTEXT_TEXT_LIMIT),
        "exit_code": event.result.exit_code if event.result else None,
        "verification": verification,
        "success_signal": _compact_or_none(event.success_signal, limit=_CONTEXT_TEXT_LIMIT),
    }
    if action_raw and (not command or normalize_text(action_raw) != normalize_text(command)):
        record["raw_action"] = action_raw
    if result_output:
        normalized_output = normalize_text(result_output)
        normalized_verification = normalize_text(verification) if verification else ""
        if not verification or (normalized_output and normalized_output not in normalized_verification):
            record["result_output"] = result_output
    state_delta = _compact_mapping(event.state_delta)
    if state_delta:
        record["state_delta"] = state_delta
    artifacts = _artifact_entries(event)
    if artifacts:
        record["artifacts"] = artifacts
    return record


def _build_event_trace_entry(event: Event, *, compact: bool = True) -> dict[str, Any]:
    if not compact:
        payload = event.model_dump(mode="python")
        action = payload.get("action")
        if isinstance(action, dict):
            action["command"] = _extract_command(action.get("arguments") or {})
        result = payload.get("result")
        if isinstance(result, dict):
            metadata = result.get("metadata") or {}
            if isinstance(metadata, dict) and metadata.get("stderr") is not None:
                result["stderr"] = str(metadata.get("stderr"))
        return payload

    action = None
    if event.action:
        action = {
            "tool": event.action.tool,
            "name": event.action.name,
            "raw": _compact_or_none(event.action.raw, limit=_CONTEXT_TEXT_LIMIT),
            "arguments": _compact_mapping(event.action.arguments),
            "command": _extract_command(event.action.arguments or {}),
        }

    result = None
    if event.result:
        result = {
            "ok": bool(event.result.ok),
            "exit_code": event.result.exit_code,
            "output_text": _summarize_result_output(event.result.output_text, limit=_CONTEXT_TEXT_LIMIT),
            "stderr": _compact_or_none((event.result.metadata or {}).get("stderr"), limit=_CONTEXT_TEXT_LIMIT),
        }

    trace: dict[str, Any] = {
        "step_id": event.step_id,
        "thought": _compact_or_none(event.thought, limit=_CONTEXT_TEXT_LIMIT),
        "observation": {
            "summary": _compact_or_none(event.observation.summary, limit=_CONTEXT_TEXT_LIMIT),
            "text": _compact_or_none(event.observation.text, limit=_CONTEXT_VALUE_LIMIT),
            "structured": _compact_mapping(event.observation.structured),
        },
        "action": action,
        "result": result,
        "success_signal": _compact_or_none(event.success_signal, limit=_CONTEXT_TEXT_LIMIT),
    }
    state_delta = _compact_mapping(event.state_delta)
    if state_delta:
        trace["state_delta"] = state_delta
    artifacts = _artifact_entries(event)
    if artifacts:
        trace["artifacts"] = artifacts
    return trace


def _information_coverage(events: list[Event]) -> dict[str, int]:
    action_event_count = sum(1 for event in events if event.action is not None)
    return {
        "total_event_count": len(events),
        "action_event_count": action_event_count,
        "non_action_event_count": len(events) - action_event_count,
        "thought_event_count": sum(1 for event in events if bool(_compact_or_none(event.thought, limit=80))),
        "result_event_count": sum(1 for event in events if event.result is not None),
        "success_signal_count": sum(1 for event in events if bool(_compact_or_none(event.success_signal, limit=80))),
        "state_delta_count": sum(1 for event in events if bool(event.state_delta)),
        "artifact_event_count": sum(1 for event in events if bool(event.artifacts)),
    }


def _build_timeline_records(events: list[Event]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for index, event in enumerate(events, start=1):
        operation = _render_operation(event.action.name, event.action.arguments) if event.action else None
        command = _extract_command((event.action.arguments or {}) if event.action else {})
        summary = (
            _compact_or_none(event.observation.summary, limit=_CONTEXT_TEXT_LIMIT)
            or _compact_or_none(event.thought, limit=_CONTEXT_TEXT_LIMIT)
            or _summarize_result_output(event.result.output_text if event.result else None, limit=_CONTEXT_TEXT_LIMIT)
            or _compact_or_none(event.success_signal, limit=_CONTEXT_TEXT_LIMIT)
            or "event"
        )
        item: dict[str, Any] = {
            "index": index,
            "step_id": event.step_id,
            "summary": summary,
        }
        if event.action:
            item["tool"] = event.action.tool
        if operation:
            item["operation"] = operation
        if command:
            item["command"] = command
        if event.result and event.result.exit_code is not None:
            item["exit_code"] = event.result.exit_code
        success_signal = _compact_or_none(event.success_signal, limit=_CONTEXT_TEXT_LIMIT)
        if success_signal:
            item["success_signal"] = success_signal
        timeline.append(item)
    return timeline


def _build_timeline_from_event_trace(event_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for index, item in enumerate(event_trace, start=1):
        if not isinstance(item, dict):
            continue
        action = item.get("action") if isinstance(item.get("action"), dict) else {}
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        observation = item.get("observation") if isinstance(item.get("observation"), dict) else {}
        summary = (
            _compact_or_none(observation.get("summary"), limit=_CONTEXT_TEXT_LIMIT)
            or _compact_or_none(item.get("thought"), limit=_CONTEXT_TEXT_LIMIT)
            or _compact_or_none(result.get("output_text"), limit=_CONTEXT_TEXT_LIMIT)
            or _compact_or_none(item.get("success_signal"), limit=_CONTEXT_TEXT_LIMIT)
            or "event"
        )
        row: dict[str, Any] = {
            "index": index,
            "step_id": item.get("step_id", index),
            "summary": summary,
        }
        tool = _compact_or_none(action.get("tool"), limit=40)
        operation = _compact_or_none(action.get("name") or action.get("operation"), limit=_CONTEXT_TEXT_LIMIT)
        command = _compact_or_none(action.get("command"), limit=_CONTEXT_TEXT_LIMIT)
        if tool:
            row["tool"] = tool
        if operation:
            row["operation"] = operation
        if command:
            row["command"] = command
        if result.get("exit_code") is not None:
            row["exit_code"] = result.get("exit_code")
        timeline.append(row)
    return timeline


def _render_workflow_context_payload(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    objective = _compact_or_none(payload.get("objective"), limit=_CONTEXT_TEXT_LIMIT) or "complete subtask"
    trigger = _compact_or_none(payload.get("trigger"), limit=_CONTEXT_TEXT_LIMIT) or "When relevant."

    lines.append("Objective: " + objective)
    lines.append("Trigger: " + trigger)

    lines.append("Preconditions:")
    preconditions = [str(item).strip() for item in (payload.get("preconditions") or []) if str(item).strip()]
    if preconditions:
        lines.extend(f"- {compact_text(item, limit=_CONTEXT_TEXT_LIMIT)}" for item in preconditions)
    else:
        lines.append("- none")

    lines.append("Steps:")
    steps = payload.get("steps") or []
    if steps:
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                lines.append(f"{index}. {compact_text(str(step), limit=_CONTEXT_TEXT_LIMIT)}")
                continue
            tool = _compact_or_none(step.get("tool"), limit=40) or "tool"
            operation = _compact_or_none(step.get("operation"), limit=_CONTEXT_TEXT_LIMIT) or "step"
            intent = _compact_or_none(step.get("intent"), limit=_CONTEXT_TEXT_LIMIT)
            command = _compact_or_none(step.get("command"), limit=_CONTEXT_TEXT_LIMIT)
            cwd = _compact_or_none(step.get("cwd"), limit=_CONTEXT_VALUE_LIMIT)
            verification = _compact_or_none(step.get("verification"), limit=_CONTEXT_TEXT_LIMIT)
            result_output = _compact_or_none(step.get("result_output"), limit=_CONTEXT_TEXT_LIMIT)
            line = f"{index}. [{tool}] {operation}"
            if intent:
                line += f" | intent={intent}"
            lines.append(line)
            if command:
                lines.append(f"   command: {command}")
            if cwd:
                lines.append(f"   cwd: {cwd}")
            if verification:
                lines.append(f"   verify: {verification}")
            if result_output:
                normalized_output = normalize_text(result_output)
                normalized_verification = normalize_text(verification) if verification else ""
                if not verification or (normalized_output and normalized_output not in normalized_verification):
                    lines.append(f"   evidence: {result_output}")
    else:
        lines.append("- none")

    lines.append("Timeline:")
    timeline = payload.get("timeline") or []
    if timeline:
        for index, event in enumerate(timeline, start=1):
            if not isinstance(event, dict):
                lines.append(f"{index}. {compact_text(str(event), limit=_CONTEXT_TEXT_LIMIT)}")
                continue
            step_id = event.get("step_id", index)
            summary = _compact_or_none(event.get("summary"), limit=_CONTEXT_TEXT_LIMIT) or "event"
            tool = _compact_or_none(event.get("tool"), limit=40)
            operation = _compact_or_none(event.get("operation"), limit=_CONTEXT_TEXT_LIMIT)
            line = f"{index}. step={step_id}"
            if tool and operation:
                line += f" | [{tool}] {operation}"
            line += f" | {summary}"
            lines.append(line)
            command = _compact_or_none(event.get("command"), limit=_CONTEXT_TEXT_LIMIT)
            if command:
                lines.append(f"   command: {command}")
            if event.get("exit_code") is not None:
                lines.append(f"   exit_code: {event.get('exit_code')}")
            success_signal = _compact_or_none(event.get("success_signal"), limit=_CONTEXT_TEXT_LIMIT)
            if success_signal:
                lines.append(f"   success: {success_signal}")
    else:
        lines.append("- none")

    lines.append("Verification:")
    verification = [str(item).strip() for item in (payload.get("verification") or []) if str(item).strip()]
    if verification:
        lines.extend(f"- {compact_text(item, limit=_CONTEXT_TEXT_LIMIT)}" for item in verification)
    else:
        lines.append("- none")

    lines.append("Failure Signals:")
    failure_modes = [str(item).strip() for item in (payload.get("failure_modes") or []) if str(item).strip()]
    if failure_modes:
        lines.extend(f"- {compact_text(item, limit=_CONTEXT_TEXT_LIMIT)}" for item in failure_modes)
    else:
        lines.append("- none")

    return "\n".join(lines)


def _extract_command(arguments: dict[str, Any]) -> str | None:
    for key in ("command", "cmd", "keystrokes", "script"):
        value = arguments.get(key)
        text = _summarize_command(value)
        if text:
            return text
    return None


def _compact_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compacted: dict[str, Any] = {}
    for key in sorted(value.keys()):
        raw = value[key]
        if raw is None:
            continue
        if isinstance(raw, dict):
            nested = _compact_mapping(raw)
            if nested:
                compacted[str(key)] = nested
            continue
        if isinstance(raw, list):
            rendered = [_compact_or_none(item, limit=80) for item in raw[:5]]
            rendered = [item for item in rendered if item]
            if rendered:
                compacted[str(key)] = rendered
            continue
        if str(key).lower() in {"stdout", "stderr", "output", "output_text"}:
            output_summary = _summarize_result_output(raw, limit=_CONTEXT_VALUE_LIMIT)
            if output_summary:
                compacted[str(key)] = output_summary
            continue
        text = _compact_or_none(raw, limit=_CONTEXT_VALUE_LIMIT)
        if text:
            compacted[str(key)] = text
    return compacted


def _artifact_entries(event: Event) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for artifact in event.artifacts:
        entry: dict[str, str] = {"kind": artifact.kind}
        if artifact.path:
            entry["path"] = compact_text(artifact.path, limit=_CONTEXT_VALUE_LIMIT)
        if artifact.description:
            entry["description"] = compact_text(artifact.description, limit=_CONTEXT_VALUE_LIMIT)
        entries.append(entry)
    return entries


def _compact_or_none(value: Any, *, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return compact_text(text, limit=limit)


def _render_argument_value(value) -> str:
    if isinstance(value, dict):
        preview = ", ".join(
            f"{key}={_render_argument_value(item)}"
            for key, item in list(sorted(value.items()))[:4]
            if item is not None
        )
        return "{" + preview + "}"
    if isinstance(value, list):
        preview = ", ".join(_render_argument_value(item) for item in value[:3])
        suffix = ", ..." if len(value) > 3 else ""
        return "[" + preview + suffix + "]"
    if isinstance(value, str):
        if "\n" in value or "\r" in value:
            compact = "\\n".join(line.strip() for line in value.splitlines() if line.strip())
        else:
            compact = " ".join(value.split())
        return compact if len(compact) <= 120 else compact[:117] + "..."
    compact = json.dumps(value, sort_keys=True, ensure_ascii=True)
    return compact if len(compact) <= 120 else compact[:117] + "..."


def _failure_modes(segment: Segment) -> list[str]:
    failures = []
    for event in segment.events:
        if not event.result or event.result.ok:
            continue
        output_summary = _summarize_result_output(event.result.output_text, limit=160)
        if output_summary:
            failures.append(f"output: {output_summary}")
        stderr = _compact_or_none((event.result.metadata or {}).get("stderr"), limit=160)
        if stderr:
            failures.append(f"stderr: {stderr}")
        if event.result.exit_code is not None:
            failures.append(f"exit_code == {event.result.exit_code}")
        if event.success_signal:
            failures.append(str(event.success_signal))
        if output_summary is None and stderr is None and event.result.exit_code is None:
            failures.append("command failed")
    return _dedupe_normalized([compact_text(item, limit=160) for item in failures if item])


def _fingerprint(steps: list[WorkflowStep]) -> str:
    basis = "|".join(f"{step.tool}:{step.operation}" for step in steps)
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


def _dedupe_normalized(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        normalized = normalize_text(item)
        if normalized in seen:
            continue
        ordered.append(item)
        seen.add(normalized)
    return ordered


def _summarize_command(value: Any) -> str | None:
    text = _compact_or_none(value, limit=1000)
    if not text:
        return None
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
    if not lines:
        return None
    if len(lines) == 1:
        return compact_text(lines[0], limit=_CONTEXT_TEXT_LIMIT)
    return compact_text(f"{lines[0]} ...", limit=_CONTEXT_TEXT_LIMIT)


def _summarize_result_output(value: Any, *, limit: int) -> str | None:
    text = _compact_or_none(value, limit=4000)
    if not text:
        return None
    if "Output:" in text:
        candidate_tail = text.rsplit("Output:", 1)[-1].strip()
        if candidate_tail:
            text = candidate_tail
    lowered = text.lower()
    if "exec_command failed for" in lowered:
        first_line = text.splitlines()[0].strip() if text.splitlines() else text
        return compact_text(first_line, limit=limit)

    lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
    if not lines:
        return None

    filtered: list[str] = []
    for line in lines:
        if line.startswith("Chunk ID:"):
            continue
        if line.startswith("Wall time:"):
            continue
        if line.startswith("Original token count:"):
            continue
        if line == "Output:":
            continue
        if line.startswith("Command: /bin/"):
            continue
        filtered.append(line)

    if not filtered:
        filtered = lines

    candidate = filtered[0]
    if candidate.startswith("Command: /bin/") and len(filtered) > 1:
        candidate = filtered[1]
    if candidate.startswith("Process running with session ID") and len(filtered) > 1:
        candidate = filtered[1]
    if candidate.startswith("Process exited with code") and len(filtered) > 1:
        candidate = f"{candidate}; {filtered[1]}"
    return compact_text(candidate, limit=limit)


def _cluster_reservation_features(
    *,
    segment: Segment,
    steps: list[WorkflowStep],
    objective: str,
    preconditions: list[str],
    verification_points: list[str],
    failure_modes: list[str],
) -> dict[str, Any]:
    action_families = [operation_family(step.tool, step.operation) for step in steps]
    step_tokens = [normalize_text(f"{step.tool}:{_operation_token(step.operation)}") for step in steps]
    step_operation_signatures = [compact_text(normalize_text(step.operation), limit=120) for step in steps]
    step_bigrams = [f"{step_tokens[index]} -> {step_tokens[index + 1]}" for index in range(max(0, len(step_tokens) - 1))]
    tool_signature = " ".join((step.tool or "tool") for step in steps)
    return {
        "event_count": len(segment.events),
        "step_count": len(steps),
        "action_families": action_families,
        "step_tokens": step_tokens,
        "step_operation_signatures": step_operation_signatures,
        "step_bigrams": step_bigrams,
        "tool_signature": tool_signature,
        "objective_token": normalize_text(objective),
        "precondition_tokens": [normalize_text(item) for item in _dedupe_normalized(preconditions)],
        "verification_tokens": [normalize_text(item) for item in _dedupe_normalized(verification_points)],
        "failure_tokens": [normalize_text(item) for item in _dedupe_normalized(failure_modes)],
        "boundary_reasons": [reason.value for reason in segment.reasons],
    }
