from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping, Sequence

from procmem2skills.models import AtomicSkill
from procmem2skills.packager.llm_skill_creator import GeneratedSkillArtifact


class SkillWriter:
    def __init__(self, include_empty_assets_dir: bool = True) -> None:
        self.include_empty_assets_dir = include_empty_assets_dir

    def write_repository(
        self,
        skills: list[AtomicSkill],
        output_dir: Path,
        generated_artifacts: Mapping[str, GeneratedSkillArtifact | Sequence[GeneratedSkillArtifact]] | None = None,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for skill in skills:
            payload = generated_artifacts.get(skill.skill_id) if generated_artifacts else None
            artifacts = _normalize_artifacts(skill, payload)
            for artifact in artifacts:
                written.append(self.write_skill_variant(skill, output_dir, generated_artifact=artifact))
        return written

    def write_skill(
        self,
        skill: AtomicSkill,
        output_dir: Path,
        generated_artifact: GeneratedSkillArtifact | None = None,
    ) -> Path:
        artifact = generated_artifact or _default_success_artifact(skill)
        return self.write_skill_variant(skill, output_dir, generated_artifact=artifact)

    def write_skill_variant(
        self,
        skill: AtomicSkill,
        output_dir: Path,
        generated_artifact: GeneratedSkillArtifact,
    ) -> Path:
        skill_dir = output_dir / generated_artifact.skill_id
        references_dir = skill_dir / "references"
        scripts_dir = skill_dir / "scripts"
        assets_dir = skill_dir / "assets"
        references_dir.mkdir(parents=True, exist_ok=True)
        scripts_dir.mkdir(parents=True, exist_ok=True)
        if self.include_empty_assets_dir:
            assets_dir.mkdir(parents=True, exist_ok=True)
            (assets_dir / ".gitkeep").write_text("", encoding="utf-8")

        (skill_dir / "SKILL.md").write_text(_ensure_trailing_newline(generated_artifact.skill_md), encoding="utf-8")
        (references_dir / "provenance.md").write_text(
            _ensure_trailing_newline(generated_artifact.provenance_md),
            encoding="utf-8",
        )
        if generated_artifact.apply_script:
            apply_script = scripts_dir / "apply.sh"
            apply_script.write_text(_ensure_trailing_newline(generated_artifact.apply_script), encoding="utf-8")
            os.chmod(apply_script, 0o755)
        if generated_artifact.recover_script:
            recover_script = scripts_dir / "recover.sh"
            recover_script.write_text(_ensure_trailing_newline(generated_artifact.recover_script), encoding="utf-8")
            os.chmod(recover_script, 0o755)
        verify_script = scripts_dir / "verify.sh"
        verify_script.write_text(_ensure_trailing_newline(generated_artifact.verify_script), encoding="utf-8")
        os.chmod(verify_script, 0o755)
        return skill_dir


def _normalize_artifacts(
    skill: AtomicSkill,
    payload: GeneratedSkillArtifact | Sequence[GeneratedSkillArtifact] | None,
) -> list[GeneratedSkillArtifact]:
    if payload is None:
        return _default_artifacts(skill)
    if isinstance(payload, GeneratedSkillArtifact):
        return [payload]
    variants = [item for item in payload if isinstance(item, GeneratedSkillArtifact)]
    if variants:
        return variants
    return _default_artifacts(skill)


def _default_artifacts(skill: AtomicSkill) -> list[GeneratedSkillArtifact]:
    return [_default_success_artifact(skill), _default_failure_artifact(skill)]


def _default_success_artifact(skill: AtomicSkill) -> GeneratedSkillArtifact:
    skill_id = f"{skill.skill_id}--success"
    return GeneratedSkillArtifact(
        base_skill_id=skill.skill_id,
        skill_id=skill_id,
        channel="success",
        skill_md=_render_skill_md(skill, channel="success"),
        provenance_md=_render_provenance(skill, channel="success"),
        apply_script=_render_apply_script(skill),
        recover_script=None,
        verify_script=_render_verify_script(skill, channel="success"),
        raw_response="",
        integration_payload=None,
    )


def _default_failure_artifact(skill: AtomicSkill) -> GeneratedSkillArtifact:
    skill_id = f"{skill.skill_id}--failure"
    return GeneratedSkillArtifact(
        base_skill_id=skill.skill_id,
        skill_id=skill_id,
        channel="failure",
        skill_md=_render_skill_md(skill, channel="failure"),
        provenance_md=_render_provenance(skill, channel="failure"),
        apply_script=None,
        recover_script=_render_recover_script(skill),
        verify_script=_render_verify_script(skill, channel="failure"),
        raw_response="",
        integration_payload=None,
    )


def _render_skill_md(skill: AtomicSkill, *, channel: str) -> str:
    channel_title = "Success Playbook" if channel == "success" else "Failure Recovery"
    channel_use = (
        skill.trigger
        if channel == "success"
        else "When execution fails, verifier checks fail, or current output matches known failure signatures."
    )
    lines = [
        "---",
        f"name: {skill.skill_id}--{channel}",
        f"description: {_channel_description(skill, channel=channel)}",
        "---",
        "",
        f"# {skill.title} ({channel_title})",
        "",
        "## Use This Skill When",
        "",
        channel_use,
        "",
    ]
    if skill.preconditions:
        lines.extend(["## Preconditions", ""])
        lines.extend(f"- {item}" for item in skill.preconditions[:6])
        lines.append("")
    lines.extend(["## Steps", ""])
    if channel == "success":
        lines.extend(_render_success_step_lines(skill))
    else:
        lines.extend(_render_failure_step_lines(skill))
    lines.append("")
    lines.extend(["## Verify", ""])
    verification = skill.verification or ["Run scripts/verify.sh and confirm checks pass."]
    lines.extend(f"- {item}" for item in verification[:6])
    lines.append("")
    lines.extend(
        [
            "## References",
            "",
            "- Read `references/provenance.md` for source benchmarks, harnesses, and tasks.",
            "- Run `scripts/verify.sh` for deterministic verification checks.",
            "- Execute the strict script in `scripts/` (`apply.sh` for success or `recover.sh` for failure).",
            "",
        ]
    )
    return "\n".join(lines)


def _render_success_step_lines(skill: AtomicSkill) -> list[str]:
    steps = []
    commands = _extract_commands(skill)
    if commands:
        steps.extend(f"{index}. Run `{command}`." for index, command in enumerate(commands, start=1))
    else:
        steps.extend(
            [
                "1. Inspect `scripts/apply.sh` and parameterize paths as needed.",
                "2. Execute `scripts/apply.sh` in the target workspace.",
            ]
        )
    return steps


def _render_failure_step_lines(skill: AtomicSkill) -> list[str]:
    items = skill.failure_recovery or ["Inspect verifier logs and map the error to a deterministic fix."]
    return [f"{index}. {item}" for index, item in enumerate(items[:6], start=1)]


def _channel_description(skill: AtomicSkill, *, channel: str) -> str:
    if channel == "success":
        return f"Deterministic success playbook distilled from {skill.support} supporting trajectories."
    return "Deterministic diagnosis and recovery steps for repeated failure signatures."


def _render_provenance(skill: AtomicSkill, *, channel: str) -> str:
    lines = [
        f"# Provenance for {skill.skill_id} ({channel})",
        "",
        f"- Support: {skill.support}",
        f"- Benchmarks: {', '.join(skill.benchmark_origins) or 'n/a'}",
        f"- Harnesses: {', '.join(skill.harness_origins) or 'n/a'}",
        f"- Agents: {', '.join(skill.agent_origins) or 'n/a'}",
        f"- Tasks: {', '.join(skill.task_origins) or 'n/a'}",
        f"- Workflows: {', '.join(skill.source_workflow_ids) or 'n/a'}",
    ]
    if channel == "failure" and skill.failure_recovery:
        lines.append("- Failure recovery hints:")
        lines.extend(f"  - {item}" for item in skill.failure_recovery[:8])
    return "\n".join(lines) + "\n"


def _render_apply_script(skill: AtomicSkill) -> str:
    commands = _extract_commands(skill)
    if not commands:
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                f"echo 'No deterministic commands extracted for {skill.skill_id}.' >&2",
                "echo 'Update this script before execution.' >&2",
                "exit 2",
                "",
            ]
        )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"# Success playbook for {skill.skill_id}",
            *commands,
            "",
        ]
    )


def _render_recover_script(skill: AtomicSkill) -> str:
    if not skill.failure_recovery:
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                f"echo 'No recovery commands available for {skill.skill_id}.' >&2",
                "echo 'Inspect verifier output and add deterministic recovery commands.' >&2",
                "exit 2",
                "",
            ]
        )
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Failure recovery for {skill.skill_id}",
    ]
    lines.extend(f"echo '[recovery] {item}'" for item in skill.failure_recovery[:10])
    lines.append("")
    return "\n".join(lines)


def _render_verify_script(skill: AtomicSkill, *, channel: str) -> str:
    checklist = "\n".join(
        f"echo '- {item}'"
        for item in (skill.verification or ["Review the target environment state manually."])
    )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"echo 'Verification checklist ({channel}) for {skill.skill_id}:'",
            checklist,
            "",
        ]
    )


def _extract_commands(skill: AtomicSkill) -> list[str]:
    commands: list[str] = []
    for step in skill.actions:
        command = _extract_command(step.operation)
        if command and command not in commands:
            commands.append(command)
    return commands


def _extract_command(operation: str) -> str | None:
    match = re.search(r"command=(.*)\)$", operation.strip())
    if not match:
        return None
    candidate = match.group(1).strip()
    return candidate or None


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"
