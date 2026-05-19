"""Build harbor-terminus-style skill-selection prompts.

Mirrors `skillsbench_repo/libs/terminus_agent/agents/terminus_2/`:
- `harbor_terminus_2_skills.py::_build_skill_prompt_xml` (available_skills block)
- `prompt-templates/terminus-xml-plain.txt` (skill-system instructions)

We strip the terminal-execution scaffolding because this harness only
evaluates the FIRST agent turn — the skill-selection decision — not
end-to-end task execution.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillCandidate:
    name: str
    description: str


SKILL_SYSTEM_BLOCK = """\
SKILL SYSTEM:
If skills are available (shown in <available_skills>), you MUST check if any \
skill matches the task BEFORE executing commands.

To load a skill, respond with ONLY this (no other tags or text):
<tool_call name="skill">
<name>skill-name</name>
</tool_call>

If multiple skills are relevant you may emit multiple <tool_call name="skill"> \
blocks, one per skill, in order of relevance. Use the exact `name` shown in \
<available_skills>. If none of the skills is relevant, respond with:
<tool_call name="skill">
<name>NONE</name>
</tool_call>
"""


DESC_MAX_CHARS = 240  # ~60 tokens; mirrors pool_builder/embed_gt truncation


def _trunc(text: str, n: int = DESC_MAX_CHARS) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:n] or "No description provided."


def build_available_skills_xml(candidates: list[SkillCandidate]) -> str:
    inner = "\n".join(
        f'  <skill name="{c.name}">{_trunc(c.description)}</skill>' for c in candidates
    )
    return f"<available_skills>\n{inner}\n</available_skills>"


def build_prompt(task_description: str, candidates: list[SkillCandidate]) -> str:
    skills_xml = build_available_skills_xml(candidates)
    return (
        f"{SKILL_SYSTEM_BLOCK}\n"
        f"Skill context:\n{skills_xml}\n\n"
        f"Task Description:\n{task_description}\n"
    )
