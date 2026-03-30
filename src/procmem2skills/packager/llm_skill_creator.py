from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from json import JSONDecodeError, JSONDecoder
from typing import Any

from procmem2skills.models import AtomicSkill

_INTEGRATOR_SYSTEM_PROMPT = """You are a trajectory-integration agent.

Integrate success and failure evidence into a compact plan for downstream skill creation.
Return one JSON object with exactly these top-level keys:
- success_strategy
- failure_strategy
- transfer_hypothesis

Requirements:
- success_strategy must include: objective, ordered_steps, preconditions, verify_checks.
- failure_strategy must include: failure_signatures, diagnosis_steps, recovery_steps.
- transfer_hypothesis must be one concise sentence.
- Keep values short and operational.
- Output strict JSON only; no markdown code fences.
"""

_SYSTEM_PROMPT = """You are a skill-creator agent.

Create dual-channel, execution-ready skills for coding agents.
Follow skill-creator style: concise, procedural, progressive disclosure, no auxiliary docs.
Return one JSON object with exactly these top-level keys:
- success
- failure

Each channel object must contain:
- skill_md
- provenance_md
- verify_script

The success channel must also contain:
- apply_script

The failure channel must also contain:
- recover_script

Requirements:
- Both skill_md values must be complete SKILL.md files with YAML frontmatter.
- Keep SKILL.md concise and practical; avoid long prose.
- Include sections in SKILL.md:
  - Use This Skill When
  - Preconditions (if needed)
  - Steps
  - Verify
  - References
- In References, point only to references/provenance.md and scripts/*.sh.
- Every script must be valid bash and start with:
  #!/usr/bin/env bash
  set -euo pipefail
- Do not wrap output in markdown code fences.
"""

_AGENT_STYLE_PROMPTS = {
    "codex": (
        "Execution style: Codex.\n"
        "- Prefer deterministic terminal-first plans.\n"
        "- Keep each step action-oriented and verifiable.\n"
        "- Avoid speculative guidance; focus on commands and checks."
    ),
    "claude-code": (
        "Execution style: Claude Code.\n"
        "- Use progressive disclosure: start from minimal safe action, then expand only when needed.\n"
        "- Keep failure recovery explicit and branch-based.\n"
        "- Preserve concise operational language."
    ),
    "opencode": (
        "Execution style: OpenCode.\n"
        "- Prioritize short edit-run-verify loops.\n"
        "- Make tool usage and verification checkpoints explicit.\n"
        "- Keep scripts minimal and composable."
    ),
}


@dataclass(frozen=True)
class GeneratedSkillArtifact:
    base_skill_id: str
    skill_id: str
    channel: str
    skill_md: str
    provenance_md: str
    apply_script: str | None
    recover_script: str | None
    verify_script: str
    raw_response: str
    integration_payload: dict[str, Any] | None = None


class LLMSkillCreator:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        timeout_sec: int = 120,
        max_retries: int = 1,
        agent_style: str = "codex",
        custom_system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_retries = max(0, max_retries)
        self.agent_style = _resolve_agent_style(agent_style)
        self.system_prompt = _compose_skill_creator_system_prompt(
            agent_style=self.agent_style,
            custom_system_prompt=custom_system_prompt,
        )

    def compose_skill(self, skill: AtomicSkill, *, failure_context: dict | None = None) -> GeneratedSkillArtifact:
        variants = self.compose_skill_variants(skill, failure_context=failure_context)
        for artifact in variants:
            if artifact.channel == "success":
                return artifact
        if variants:
            return variants[0]
        raise RuntimeError("LLM skill creator returned no variants")

    def compose_skill_variants(self, skill: AtomicSkill, *, failure_context: dict | None = None) -> list[GeneratedSkillArtifact]:
        integration_payload = self._integrate_evidence(skill, failure_context=failure_context)
        prompt = _build_user_prompt(
            skill,
            failure_context=failure_context,
            integration_payload=integration_payload,
        )
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            raw_response = self._chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
            try:
                payload = _parse_json_payload(raw_response)
                artifacts = _artifacts_from_payload(
                    skill,
                    payload,
                    raw_response,
                    integration_payload=integration_payload,
                )
            except Exception as exc:
                last_error = exc
                continue
            if artifacts:
                return artifacts
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM skill creation failed without a concrete error")

    def _integrate_evidence(self, skill: AtomicSkill, *, failure_context: dict | None = None) -> dict[str, Any]:
        prompt = _build_integrator_prompt(skill, failure_context=failure_context)
        try:
            raw = self._chat(
                messages=[
                    {"role": "system", "content": _INTEGRATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )
            payload = _parse_json_payload(raw)
        except Exception:
            return _heuristic_integration_payload(skill, failure_context=failure_context)
        if not isinstance(payload, dict):
            return _heuristic_integration_payload(skill, failure_context=failure_context)
        required = {"success_strategy", "failure_strategy", "transfer_hypothesis"}
        if required.issubset(payload.keys()):
            return payload
        return _heuristic_integration_payload(skill, failure_context=failure_context)

    def _chat(self, messages: list[dict[str, str]]) -> str:
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY or OPENAI_API_KEY is required for LLM skill creation")
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
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM skill creation request failed with status {exc.code}: {detail}") from exc
        message = ((body.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        if isinstance(message, list):
            return "".join(part.get("text", "") for part in message if isinstance(part, dict))
        return str(message or "")


def _resolve_agent_style(value: str) -> str:
    normalized = (value or "codex").strip().lower()
    aliases = {
        "cc": "claude-code",
        "claude_code": "claude-code",
        "claudecode": "claude-code",
        "open-code": "opencode",
        "open_code": "opencode",
        "openai-codex": "codex",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in _AGENT_STYLE_PROMPTS:
        allowed = ", ".join(sorted(_AGENT_STYLE_PROMPTS.keys()))
        raise ValueError(f"unsupported skill creator agent style: {value} (expected one of: {allowed})")
    return resolved


def _compose_skill_creator_system_prompt(*, agent_style: str, custom_system_prompt: str | None) -> str:
    style_key = _resolve_agent_style(agent_style)
    style_prompt = _AGENT_STYLE_PROMPTS.get(style_key, "")
    custom = str(custom_system_prompt or "").strip()
    parts = [_SYSTEM_PROMPT.strip()]
    if style_prompt:
        parts.append(style_prompt.strip())
    if custom:
        parts.append("Additional system constraints:")
        parts.append(custom)
    return "\n\n".join(part for part in parts if part)


def _build_dual_channel_prompt(skill: AtomicSkill) -> str:
    source_payload = _skill_payload(skill)
    prompt = (
        "Generate one dual-channel skill package from this mined candidate.\n\n"
        "Output only a JSON object with these keys:\n"
        "- success (object)\n"
        "- failure (object)\n\n"
        "success object must include:\n"
        "- skill_md (SKILL.md)\n"
        "- provenance_md (references/provenance.md)\n"
        "- apply_script (scripts/apply.sh)\n"
        "- verify_script (scripts/verify.sh)\n\n"
        "failure object must include:\n"
        "- skill_md (SKILL.md)\n"
        "- provenance_md (references/provenance.md)\n"
        "- recover_script (scripts/recover.sh)\n"
        "- verify_script (scripts/verify.sh)\n\n"
        "Do not invent extra files (no README/CHANGELOG/installation guides).\n\n"
        "Candidate JSON:\n"
        + json.dumps(source_payload, indent=2, ensure_ascii=False)
    )
    return prompt


def _build_integrator_prompt(skill: AtomicSkill, *, failure_context: dict | None = None) -> str:
    source_payload = _skill_payload(skill)
    prompt = (
        "Integrate trajectory evidence for transfer learning.\n\n"
        "Return JSON with success_strategy, failure_strategy, transfer_hypothesis.\n\n"
        "Candidate JSON:\n"
        + json.dumps(source_payload, indent=2, ensure_ascii=False)
    )
    if failure_context:
        prompt += (
            "\n\nWeak-model failure analysis:\n"
            + json.dumps(failure_context, indent=2, ensure_ascii=False)
        )
    return prompt


def _skill_payload(skill: AtomicSkill) -> dict[str, Any]:
    source_payload = {
        "skill_id": skill.skill_id,
        "title": skill.title,
        "description": skill.description,
        "trigger": skill.trigger,
        "preconditions": skill.preconditions,
        "actions": [action.model_dump() for action in skill.actions],
        "verification": skill.verification,
        "failure_recovery": skill.failure_recovery,
        "benchmark_origins": skill.benchmark_origins,
        "harness_origins": skill.harness_origins,
        "agent_origins": skill.agent_origins,
        "task_origins": skill.task_origins,
        "source_workflow_ids": skill.source_workflow_ids,
        "support": skill.support,
        "metadata": skill.metadata,
    }
    return source_payload


def _build_user_prompt(
    skill: AtomicSkill,
    *,
    failure_context: dict | None = None,
    integration_payload: dict[str, Any] | None = None,
) -> str:
    prompt = _build_dual_channel_prompt(skill)
    if integration_payload:
        prompt += (
            "\n\nTrajectory integration summary (generated before skill creation):\n"
            + json.dumps(integration_payload, indent=2, ensure_ascii=False)
        )
    if failure_context:
        prompt += (
            "\n\nWeak-model failure analysis (prioritize failure-targeted recovery steps when relevant):\n"
            + json.dumps(failure_context, indent=2, ensure_ascii=False)
        )
    return prompt


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
    raise ValueError(f"could not parse LLM skill creator response as JSON: {text}")


def _artifacts_from_payload(
    skill: AtomicSkill,
    payload: dict[str, Any],
    raw_response: str,
    *,
    integration_payload: dict[str, Any] | None = None,
) -> list[GeneratedSkillArtifact]:
    success_payload = _select_channel_payload(payload, "success")
    failure_payload = _select_channel_payload(payload, "failure")

    if success_payload is None and failure_payload is None:
        return [_legacy_artifact_from_payload(skill, payload, raw_response, integration_payload=integration_payload)]

    artifacts: list[GeneratedSkillArtifact] = []
    if success_payload is not None:
        artifacts.append(
            _build_channel_artifact(
                skill,
                channel="success",
                channel_payload=success_payload,
                raw_response=raw_response,
                integration_payload=integration_payload,
            )
        )
    if failure_payload is not None:
        artifacts.append(
            _build_channel_artifact(
                skill,
                channel="failure",
                channel_payload=failure_payload,
                raw_response=raw_response,
                integration_payload=integration_payload,
            )
        )
    return artifacts


def _legacy_artifact_from_payload(
    skill: AtomicSkill,
    payload: dict[str, Any],
    raw_response: str,
    *,
    integration_payload: dict[str, Any] | None = None,
) -> GeneratedSkillArtifact:
    skill_md = str(payload.get("skill_md") or "").strip()
    provenance_md = str(payload.get("provenance_md") or "").strip()
    verify_script = _normalize_script(payload.get("verify_script"), default=_default_verify_script(skill, channel="success"))
    apply_script = _normalize_script(payload.get("apply_script"), default=_default_apply_script(skill))
    if not skill_md:
        raise ValueError("LLM skill creator response missing `skill_md`")
    if not provenance_md:
        raise ValueError("LLM skill creator response missing `provenance_md`")
    skill_id = _variant_skill_id(skill.skill_id, "success")
    return GeneratedSkillArtifact(
        base_skill_id=skill.skill_id,
        skill_id=skill_id,
        channel="success",
        skill_md=_ensure_skill_frontmatter(skill_md, skill_id, channel="success", base_description=skill.description),
        provenance_md=provenance_md,
        apply_script=apply_script,
        recover_script=None,
        verify_script=verify_script,
        raw_response=raw_response,
        integration_payload=integration_payload,
    )


def _select_channel_payload(payload: dict[str, Any], channel: str) -> dict[str, Any] | None:
    channel_payload = payload.get(channel)
    if isinstance(channel_payload, dict):
        return channel_payload
    alias = payload.get(f"{channel}_skill")
    if isinstance(alias, dict):
        return alias
    flattened = {key: value for key, value in payload.items() if key.startswith(f"{channel}_")}
    if not flattened:
        return None
    remapped = {key[len(channel) + 1 :]: value for key, value in flattened.items()}
    return remapped if remapped else None


def _build_channel_artifact(
    skill: AtomicSkill,
    *,
    channel: str,
    channel_payload: dict[str, Any],
    raw_response: str,
    integration_payload: dict[str, Any] | None = None,
) -> GeneratedSkillArtifact:
    skill_id = _variant_skill_id(skill.skill_id, channel)
    skill_md = str(channel_payload.get("skill_md") or "").strip()
    provenance_md = str(channel_payload.get("provenance_md") or "").strip()
    if not skill_md:
        raise ValueError(f"LLM skill creator response missing `{channel}.skill_md`")
    if not provenance_md:
        raise ValueError(f"LLM skill creator response missing `{channel}.provenance_md`")

    verify_script = _normalize_script(
        channel_payload.get("verify_script"),
        default=_default_verify_script(skill, channel=channel),
    )
    apply_script: str | None = None
    recover_script: str | None = None
    if channel == "success":
        apply_script = _normalize_script(
            channel_payload.get("apply_script") or channel_payload.get("execute_script"),
            default=_default_apply_script(skill),
        )
    else:
        recover_script = _normalize_script(
            channel_payload.get("recover_script") or channel_payload.get("diagnose_script"),
            default=_default_recover_script(skill),
        )

    return GeneratedSkillArtifact(
        base_skill_id=skill.skill_id,
        skill_id=skill_id,
        channel=channel,
        skill_md=_ensure_skill_frontmatter(skill_md, skill_id, channel=channel, base_description=skill.description),
        provenance_md=provenance_md,
        apply_script=apply_script,
        recover_script=recover_script,
        verify_script=verify_script,
        raw_response=raw_response,
        integration_payload=integration_payload,
    )


def _variant_skill_id(skill_id: str, channel: str) -> str:
    suffix = "success" if channel == "success" else "failure"
    return f"{skill_id}--{suffix}"


def _ensure_skill_frontmatter(markdown: str, skill_id: str, *, channel: str, base_description: str) -> str:
    text = markdown.replace("\r\n", "\n").strip()
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        description = _channel_description(channel, base_description)
        body = text if text else f"# {skill_id}\n\n## Steps\n\n- Fill in deterministic steps."
        return "\n".join(
            [
                "---",
                f"name: {skill_id}",
                f"description: {description}",
                "---",
                "",
                body,
            ]
        ).strip()

    out = []
    in_frontmatter = True
    name_written = False
    description_written = False
    for line in lines:
        if in_frontmatter and line.strip().startswith("name:"):
            out.append(f"name: {skill_id}")
            name_written = True
            continue
        if in_frontmatter and line.strip().startswith("description:"):
            out.append(f"description: {_channel_description(channel, base_description)}")
            description_written = True
            continue
        out.append(line)
        if in_frontmatter and line.strip() == "---" and len(out) > 1:
            if not name_written:
                out.insert(1, f"name: {skill_id}")
                name_written = True
            if not description_written:
                out.insert(2 if name_written else 1, f"description: {_channel_description(channel, base_description)}")
            in_frontmatter = False
    return "\n".join(out).strip()


def _channel_description(channel: str, base_description: str) -> str:
    label = "success playbook" if channel == "success" else "failure diagnosis and recovery"
    return f"{label} for {base_description}".strip()


def _normalize_script(raw_script: Any, *, default: str) -> str:
    script = str(raw_script or "").strip()
    if not script:
        script = default.strip()
    if not script.startswith("#!/usr/bin/env bash"):
        script = "#!/usr/bin/env bash\nset -euo pipefail\n\n" + script
    if "set -euo pipefail" not in script:
        script = script.replace("#!/usr/bin/env bash", "#!/usr/bin/env bash\nset -euo pipefail", 1)
    return script


def _default_apply_script(skill: AtomicSkill) -> str:
    commands = _extract_candidate_commands(skill)
    if not commands:
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                f"echo 'No deterministic command extracted for {skill.skill_id}.' >&2",
                "echo 'Fill scripts/apply.sh with environment-specific commands before execution.' >&2",
                "exit 2",
            ]
        )
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Success playbook for {skill.skill_id}",
    ]
    lines.extend(commands)
    return "\n".join(lines)


def _default_recover_script(skill: AtomicSkill) -> str:
    recovery_lines = [line for line in skill.failure_recovery if line]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Failure recovery for {skill.skill_id}",
    ]
    if not recovery_lines:
        lines.extend(
            [
                "echo 'No explicit failure recovery commands were mined.' >&2",
                "echo 'Inspect verifier output and update scripts/recover.sh deterministically.' >&2",
                "exit 2",
            ]
        )
        return "\n".join(lines)
    for item in recovery_lines:
        lines.append(f"echo '[recovery] {item}'")
    return "\n".join(lines)


def _default_verify_script(skill: AtomicSkill, *, channel: str) -> str:
    checklist = skill.verification or ["Review environment state and rerun verifier checks."]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"echo 'Verification checklist ({channel}) for {skill.skill_id}:'",
    ]
    lines.extend([f"echo '- {item}'" for item in checklist])
    return "\n".join(lines)


def _extract_candidate_commands(skill: AtomicSkill) -> list[str]:
    commands: list[str] = []
    for step in skill.actions:
        command = _extract_command_from_operation(step.operation)
        if command and command not in commands:
            commands.append(command)
    return commands


def _extract_command_from_operation(operation: str) -> str | None:
    text = operation.strip()
    match = re.search(r"command=(.*)\)$", text)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            return candidate
    if text and "(" not in text and ")" not in text:
        return text
    return None


def _heuristic_integration_payload(skill: AtomicSkill, *, failure_context: dict | None = None) -> dict[str, Any]:
    failure_signatures = []
    task_failure = (failure_context or {}).get("task_failure_analysis") if isinstance(failure_context, dict) else None
    if isinstance(task_failure, dict):
        for payload in task_failure.values():
            if not isinstance(payload, dict):
                continue
            for signal in payload.get("failure_signals") or []:
                if isinstance(signal, dict):
                    signature = str(signal.get("signature") or "").strip()
                else:
                    signature = str(signal).strip()
                if signature and signature not in failure_signatures:
                    failure_signatures.append(signature)
    return {
        "success_strategy": {
            "objective": skill.title,
            "ordered_steps": [action.operation for action in skill.actions],
            "preconditions": list(skill.preconditions),
            "verify_checks": list(skill.verification),
        },
        "failure_strategy": {
            "failure_signatures": failure_signatures[:8],
            "diagnosis_steps": list(skill.failure_recovery[:4]),
            "recovery_steps": list(skill.failure_recovery[:4]),
        },
        "transfer_hypothesis": "Reuse successful deterministic steps first; apply failure recovery only when matching failure signatures appear.",
    }


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
