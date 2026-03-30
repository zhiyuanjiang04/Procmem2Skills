from __future__ import annotations

from pathlib import Path
from typing import Any

from procmem2skills.runtime.retrieval import SkillIndex


def build_native_skill_prompt_template(
    *,
    skill_repository: Path,
    output_path: Path,
    task_names: list[str],
    max_skills: int | None = 12,
    failure_guardrails: dict | None = None,
) -> dict[str, Any]:
    skill_index = SkillIndex.from_repository(skill_repository)
    all_cards = skill_index.cards()
    skill_ids = _select_skill_ids(
        skill_index=skill_index,
        task_names=task_names,
        max_skills=max_skills,
    )
    card_by_id = {card.skill_id: card for card in all_cards}
    success_sections: list[str] = []
    failure_sections: list[str] = []
    other_sections: list[str] = []
    for skill_id in skill_ids:
        card = card_by_id.get(skill_id)
        if card is None:
            continue
        bundle = skill_index.load_bundle(skill_id)
        channel = _skill_channel(skill_id)
        body = bundle.body.strip() or "(empty)"
        lines = [
            f"### {card.name} (`{skill_id}`)",
            f"Description: {card.description}",
            f"Channel: {channel}",
            "",
            "```markdown",
            body,
            "```",
        ]
        if bundle.references:
            lines.append(f"References: {', '.join(sorted(bundle.references.keys()))}")
        if bundle.scripts:
            lines.append(f"Scripts: {', '.join(bundle.scripts)}")
        section = "\n".join(lines)
        if channel == "success":
            success_sections.append(section)
        elif channel == "failure":
            failure_sections.append(section)
        else:
            other_sections.append(section)
    sections = success_sections + failure_sections + other_sections
    skill_block = "\n\n".join(sections) if sections else "No distilled skills are available."
    guardrails_block = _render_failure_guardrails(failure_guardrails, task_names=task_names)
    template = "\n".join(
        [
            "You are solving a terminal task.",
            "",
            "Task Instruction:",
            "{{ instruction }}",
            "",
            "Distilled Skills:",
            _wrap_in_raw_block(skill_block),
            "",
            "Failure Guardrails:",
            _wrap_in_raw_block(guardrails_block),
            "",
            "Skill usage policy:",
            "1. Prefer success-channel skills first to build the execution plan.",
            "2. Trigger failure-channel skills only when command output or verifier output matches failure signatures.",
            "3. Follow script paths (`scripts/apply.sh`, `scripts/recover.sh`, `scripts/verify.sh`) exactly when present.",
            "4. If no skill applies, continue with normal debugging/execution.",
            "5. Do not fabricate commands or file paths that are not in the environment.",
            "6. Do not repeat an unchanged failed command sequence under the same failure signature.",
            "7. If a retry is needed, change diagnosis and command plan first, then explain what changed.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template, encoding="utf-8")
    return {
        "prompt_template_path": str(output_path),
        "max_skills": max_skills,
        "selected_skill_count": len(skill_ids),
        "selected_skill_ids": skill_ids,
        "has_failure_guardrails": bool(isinstance(failure_guardrails, dict) and failure_guardrails.get("by_task")),
    }


def _select_skill_ids(
    *,
    skill_index: SkillIndex,
    task_names: list[str],
    max_skills: int | None,
) -> list[str]:
    cards = skill_index.cards()
    if not cards:
        return []
    ordered_ids = [card.skill_id for card in cards]
    if max_skills is None or max_skills <= 0:
        return ordered_ids
    limit = min(max_skills, len(ordered_ids))
    if limit == len(ordered_ids):
        return ordered_ids
    query = " ".join(task_names).strip()
    selected: list[str] = []
    if query:
        hits = skill_index.search(query, top_k=limit, scope="metadata")
        selected = [hit.skill_id for hit in hits]
    seen = set(selected)
    for skill_id in ordered_ids:
        if len(selected) >= limit:
            break
        if skill_id in seen:
            continue
        selected.append(skill_id)
        seen.add(skill_id)
    return selected


def _skill_channel(skill_id: str) -> str:
    normalized = skill_id.strip().lower()
    if normalized.endswith("--success"):
        return "success"
    if normalized.endswith("--failure"):
        return "failure"
    return "general"


def _wrap_in_raw_block(text: str) -> str:
    marker = "{% endraw %}"
    escaped = text.replace(marker, marker + "{% raw %}")
    return "{% raw %}\n" + escaped + "\n{% endraw %}"


def _render_failure_guardrails(failure_guardrails: dict | None, *, task_names: list[str]) -> str:
    if not isinstance(failure_guardrails, dict):
        return "No explicit failure guardrails were generated."
    by_task = failure_guardrails.get("by_task")
    if not isinstance(by_task, dict) or not by_task:
        return "No explicit failure guardrails were generated."

    lines: list[str] = []
    for task_name in task_names:
        payload = by_task.get(task_name)
        if not isinstance(payload, dict):
            continue
        lines.append(f"- Task: {task_name}")
        signatures = payload.get("failure_signatures") or []
        if signatures:
            formatted = []
            for entry in signatures[:4]:
                if isinstance(entry, dict):
                    signature = str(entry.get("signature") or "").strip()
                else:
                    signature = str(entry).strip()
                if signature:
                    formatted.append(signature)
            if formatted:
                lines.append("  Failure signatures: " + "; ".join(formatted))
        sequences = payload.get("failed_command_sequences") or []
        if sequences:
            cleaned = [str(item).strip() for item in sequences[:2] if str(item).strip()]
            if cleaned:
                lines.append("  Avoid repeating failed command chain:")
                lines.extend([f"  - {item}" for item in cleaned])
    if lines:
        return "\n".join(lines)

    global_signatures = failure_guardrails.get("global_failure_signatures") or []
    rendered_global = []
    for entry in global_signatures[:5]:
        if isinstance(entry, dict):
            signature = str(entry.get("signature") or "").strip()
        else:
            signature = str(entry).strip()
        if signature:
            rendered_global.append(signature)
    if rendered_global:
        return "Top failure signatures: " + "; ".join(rendered_global)
    return "No explicit failure guardrails were generated."
