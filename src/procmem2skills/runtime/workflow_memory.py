from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from procmem2skills.inducer.workflow import render_workflow_context
from procmem2skills.models import WorkflowCandidate


@dataclass(frozen=True)
class RenderedWorkflowMemory:
    requested_task_key: str | None
    resolved_task_key: str | None
    attempt_count: int
    workflow_count: int
    text: str


class WorkflowMemoryIndex:
    def __init__(self, grouped_attempts: dict[str, list[dict[str, Any]]]) -> None:
        self._grouped_attempts = grouped_attempts
        self._alias_to_task_key: dict[str, str] = {}

        for task_key, attempts in grouped_attempts.items():
            normalized_key = normalize_task_key(task_key)
            if not normalized_key:
                continue
            self._alias_to_task_key.setdefault(normalized_key, task_key)
            for attempt in attempts:
                task_id = str((attempt or {}).get("task_id") or "").strip()
                if not task_id:
                    continue
                self._alias_to_task_key.setdefault(normalize_task_key(task_id), task_key)

    @classmethod
    def from_grouped_attempts(cls, payload: dict[str, Any]) -> "WorkflowMemoryIndex":
        grouped: dict[str, list[dict[str, Any]]] = {}
        if isinstance(payload, dict):
            for task_key, attempts in payload.items():
                normalized_task_key = str(task_key or "").strip() or "unknown-task"
                if not isinstance(attempts, list):
                    continue
                grouped[normalized_task_key] = [attempt for attempt in attempts if isinstance(attempt, dict)]
        return cls(grouped)

    @classmethod
    def from_path(cls, path: Path) -> "WorkflowMemoryIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"workflow memory payload must be an object: {path}")
        return cls.from_grouped_attempts(payload)

    def resolve_task_key(self, task_hint: str | None) -> str | None:
        normalized_hint = normalize_task_key(task_hint)
        if not normalized_hint:
            return None
        if normalized_hint in self._alias_to_task_key:
            return self._alias_to_task_key[normalized_hint]
        return None

    def render_task_memory(
        self,
        task_hint: str | None,
        *,
        max_attempts: int | None = None,
        max_workflows_per_attempt: int | None = None,
        max_steps_per_workflow: int | None = None,
    ) -> RenderedWorkflowMemory:
        resolved_task_key = self.resolve_task_key(task_hint)
        if resolved_task_key is None:
            return RenderedWorkflowMemory(
                requested_task_key=task_hint,
                resolved_task_key=None,
                attempt_count=0,
                workflow_count=0,
                text="<none>",
            )

        raw_attempts = list(self._grouped_attempts.get(resolved_task_key) or [])
        sorted_attempts = _sort_attempts(raw_attempts)
        max_attempts_value = _as_optional_positive_int(max_attempts)
        if max_attempts_value is not None:
            sorted_attempts = sorted_attempts[:max_attempts_value]

        max_workflows_value = _as_optional_positive_int(max_workflows_per_attempt)
        max_steps_value = _as_optional_positive_int(max_steps_per_workflow)
        total_workflows = 0
        lines = [
            "Workflow memory below was induced from previous trajectories of the same task.",
            "Use it as procedural guidance. Adapt paths and commands to the current environment before execution.",
            f"Task: {resolved_task_key}",
            f"Attempt Count: {len(sorted_attempts)}",
        ]

        for index, attempt in enumerate(sorted_attempts, start=1):
            status = str((attempt or {}).get("status") or "unknown")
            episode_id = str((attempt or {}).get("episode_id") or "")
            task_id = str((attempt or {}).get("task_id") or resolved_task_key)
            workflows = [item for item in ((attempt or {}).get("workflows") or []) if isinstance(item, dict)]
            if max_workflows_value is not None:
                workflows = workflows[:max_workflows_value]

            lines.append("")
            lines.append(f"Attempt {index}: status={status} task_id={task_id}" + (f" episode_id={episode_id}" if episode_id else ""))
            lines.append(f"Workflow Count: {len(workflows)}")

            if not workflows:
                lines.append("- No workflow extracted for this attempt.")
                continue

            for workflow_index, raw_workflow in enumerate(workflows, start=1):
                try:
                    workflow = WorkflowCandidate.model_validate(raw_workflow)
                except Exception:
                    lines.append(f"- Workflow {workflow_index}: invalid payload skipped.")
                    continue
                total_workflows += 1
                lines.append(f"- Workflow {workflow_index}:")
                rendered = render_workflow_context(workflow, max_steps=max_steps_value)
                lines.extend(f"  {line}" if line else "  " for line in rendered.splitlines())

        text = "\n".join(lines).strip() or "<none>"
        return RenderedWorkflowMemory(
            requested_task_key=task_hint,
            resolved_task_key=resolved_task_key,
            attempt_count=len(sorted_attempts),
            workflow_count=total_workflows,
            text=text,
        )


def normalize_task_key(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    if "__" in raw:
        raw = raw.split("__", 1)[0]
    if "/" in raw:
        raw = raw.split("/")[-1]
    raw = raw.strip()
    if not raw:
        return ""
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", raw)).strip("-")


def _sort_attempts(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sortable: list[tuple[int, int, dict[str, Any]]] = []
    fallback_index = 0
    for attempt in attempts:
        fallback_index += 1
        raw_index = (attempt or {}).get("attempt_index")
        try:
            parsed = int(raw_index)
            sortable.append((0, parsed, attempt))
        except Exception:
            sortable.append((1, fallback_index, attempt))
    sortable.sort(key=lambda row: (row[0], row[1]))
    return [row[2] for row in sortable]


def _as_optional_positive_int(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed <= 0:
        return None
    return parsed

