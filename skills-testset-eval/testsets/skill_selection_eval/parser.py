"""Parse skill selections from model output.

Uses the same regex as harbor terminus_2:
`harbor_terminus_2_skills.py::_handle_skill_tool_calls_xml`
"""
from __future__ import annotations

import re

_SKILL_TOOL_CALL_RE = re.compile(
    r'<tool_call\s+name="skill">\s*<name>([^<]+)</name>\s*</tool_call>',
    re.DOTALL,
)


def parse_selected_skills(response: str) -> list[str]:
    """Return the ordered list of skill names the agent emitted.

    `NONE` is treated as an explicit no-selection sentinel and stripped.
    Duplicates are removed while preserving first-occurrence order.
    """
    matches = _SKILL_TOOL_CALL_RE.findall(response or "")
    seen: dict[str, None] = {}
    for raw in matches:
        name = raw.strip()
        if not name or name.upper() == "NONE":
            continue
        if name not in seen:
            seen[name] = None
    return list(seen.keys())
