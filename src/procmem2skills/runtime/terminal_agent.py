from __future__ import annotations

import html
import json
import re
from json import JSONDecodeError, JSONDecoder
from typing import Iterable

from pydantic import BaseModel, Field

from procmem2skills.runtime.retrieval import SkillBundle

_SYSTEM_PROMPT = """You are a terminal agent solving a software task inside Terminal-Bench.

You must respond with a single JSON object and nothing else:
{"thought": "...", "command": "...", "done": false}

Rules:
- The shell does not preserve state between commands. If you need a directory, include `cd <path> && ...` in the command itself.
- Prefer short, non-interactive shell commands.
- Do not open editors that require a TTY.
- Reuse retrieved skills when they match the current failure mode or task stage.
- Workflow memories (when provided) are historical trajectories for the same task. Treat them as guidance, not rigid scripts.
- When workflow memories disagree with current observations, trust current observations and re-verify with commands.
- Do not emit XML tags, <function_calls>, markdown code fences, or tool-call markup.
- If the task is complete or you are blocked, return {"thought": "...", "command": "", "done": true}.
"""

_XML_PARAMETER_RE = re.compile(
    r"<parameter\s+name=['\"](?P<name>[^'\"]+)['\"]\s*>(?P<value>.*?)</parameter>",
    re.IGNORECASE | re.DOTALL,
)


class TerminalAgentDecision(BaseModel):
    thought: str = ""
    command: str = ""
    done: bool = False


class TerminalExecutionSnapshot(BaseModel):
    instruction: str
    cwd: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    recent_history: list[str] = Field(default_factory=list)


def build_terminal_query(snapshot: TerminalExecutionSnapshot) -> str:
    parts = [snapshot.instruction]
    if snapshot.cwd:
        parts.append(f"cwd={snapshot.cwd}")
    if snapshot.stdout:
        parts.append(snapshot.stdout)
    if snapshot.stderr:
        parts.append(snapshot.stderr)
    parts.extend(snapshot.recent_history[-3:])
    return "\n".join(part for part in parts if part)


def build_terminal_agent_messages(
    *,
    snapshot: TerminalExecutionSnapshot,
    retrieved_skills: Iterable[SkillBundle],
    workflow_memories_context: str | None = None,
) -> list[dict[str, str]]:
    skill_context = render_skill_context(retrieved_skills)
    workflow_context = (workflow_memories_context or "").strip() or "<none>"
    history = "\n".join(f"- {item}" for item in snapshot.recent_history[-5:]) or "- <none>"
    observation = "\n".join(
        part
        for part in [
            f"Current working directory: {snapshot.cwd}" if snapshot.cwd else None,
            "Stdout:\n" + _clip_text(snapshot.stdout) if snapshot.stdout else None,
            "Stderr:\n" + _clip_text(snapshot.stderr) if snapshot.stderr else None,
        ]
        if part
    )
    if not observation:
        observation = "No command has been executed yet."
    user_prompt = "\n\n".join(
        [
            "Task Instruction:\n" + snapshot.instruction,
            "Recent Command History:\n" + history,
            "Current Observation:\n" + observation,
            "Retrieved Skills:\n" + skill_context,
            "Workflow Memories:\n" + workflow_context,
        ]
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def render_skill_context(bundles: Iterable[SkillBundle]) -> str:
    rendered = []
    for bundle in bundles:
        references = ", ".join(sorted(bundle.references)) if bundle.references else "<none>"
        scripts = ", ".join(bundle.scripts) if bundle.scripts else "<none>"
        body = _clip_text(bundle.body, limit=1200)
        rendered.append(
            "\n".join(
                [
                    f"Skill ID: {bundle.skill_id}",
                    f"Scripts: {scripts}",
                    f"References: {references}",
                    "Body:",
                    body,
                ]
            )
        )
    return "\n\n".join(rendered) if rendered else "<none>"


def parse_terminal_agent_response(text: str) -> TerminalAgentDecision:
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
            return TerminalAgentDecision(
                thought=str(payload.get("thought") or ""),
                command=str(payload.get("command") or ""),
                done=bool(payload.get("done", False)),
            )

    xml_decision = _parse_xmlish_tool_response(cleaned)
    if xml_decision is not None:
        return xml_decision

    raise ValueError(f"could not parse terminal agent response as JSON or XML tool markup: {text}")


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


def _parse_xmlish_tool_response(text: str) -> TerminalAgentDecision | None:
    matches = list(_XML_PARAMETER_RE.finditer(text))
    if not matches and "<function_calls>" not in text and "<parameter" not in text:
        return None

    commands: list[str] = []
    thoughts: list[str] = []
    for match in matches:
        name = match.group("name").strip().lower()
        value = html.unescape(match.group("value")).strip()
        if not value:
            continue
        if name == "command":
            commands.append(value)
        elif name == "thought":
            thoughts.append(value)

    leading_text = text
    for marker in ("<function_calls>", "<parameter", "<invoke"):
        if marker in leading_text:
            leading_text = leading_text.split(marker, 1)[0]
    leading_text = leading_text.strip()

    if not commands and not thoughts and not leading_text:
        return None

    command = "\n".join(commands).strip()
    thought = (thoughts[-1] if thoughts else leading_text).strip()
    return TerminalAgentDecision(
        thought=thought,
        command=command,
        done=not bool(command),
    )


def _clip_text(text: str | None, limit: int = 1600) -> str:
    if not text:
        return ""
    compact = text.strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
