"""Assemble the prefill-context system prompt: all candidate SKILL.md contents
concatenated, with GT real content and noise stubs from corpus description.

Output system prompt has roughly this shape:

  You are an autonomous agent solving the task below. You have shell access.

  The following skills are available — their full SKILL.md content is prefilled.
  Use any skill whose instructions match the task; ignore irrelevant ones.

  <skill name="...">
  <full SKILL.md content>
  </skill>
  <skill name="...">
  ...
  </skill>

  Task workspace: /workdir
  When you finish, save outputs to the paths specified in the task.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


SYSTEM_PREAMBLE = """You are an autonomous agent. You have a Bash tool and a working directory at /workdir (the current directory). Solve the task below by running shell commands. Write outputs to the paths the task specifies. Reply concisely; the user does not need narration."""


@lru_cache(maxsize=1)
def _corpus_by_slug(corpus_path: str) -> dict:
    out = {}
    for line in Path(corpus_path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["slug"]] = r
            if r.get("name") and r["name"] not in out:
                out[r["name"]] = r
    return out


def _stub_skill_md(slug: str, desc: str) -> str:
    name = slug.split("/")[-1] if "/" in slug else slug
    body = (desc or f"Skill {slug}").strip()
    return f"""---
name: {name}
description: {body[:200]!r}
---

# {name}

{body}

## When to Use

Use this skill if your task aligns with the description above. (Note: this
skill's full content was not available; only the description is shown.)
"""


def load_skill_md(*, slug: str, sb_task_dir: Path | None, corpus_path: Path) -> str:
    """Resolve a skill slug → SKILL.md text.

    Lookup order:
      1. {sb_task_dir}/environment/skills/{name}/SKILL.md          (real GT content)
      2. corpus description → synthesized stub                     (noise content)
    """
    name = slug.split("/")[-1] if "/" in slug else slug
    if sb_task_dir is not None:
        for cand in (sb_task_dir / "environment" / "skills").glob("*"):
            if cand.is_dir() and (cand.name == name or cand.name == slug.replace("/", "-")):
                md = cand / "SKILL.md"
                if md.exists():
                    return md.read_text()
    by_slug = _corpus_by_slug(str(corpus_path))
    rec = by_slug.get(slug) or by_slug.get(name)
    desc = rec.get("description") if rec else ""
    return _stub_skill_md(slug, desc or "")


def build_system_prompt(
    *,
    candidate_slugs: list[str],
    task_instruction: str,
    sb_task_dir: Path | None,
    corpus_path: Path,
) -> str:
    blocks = []
    for slug in candidate_slugs:
        md = load_skill_md(slug=slug, sb_task_dir=sb_task_dir, corpus_path=corpus_path)
        name = slug.split("/")[-1] if "/" in slug else slug
        blocks.append(f'<skill name="{name}">\n{md.strip()}\n</skill>')

    return (
        SYSTEM_PREAMBLE
        + "\n\nThe following skills are prefilled. Use any whose instructions match the task.\n\n"
        + "\n\n".join(blocks)
        + "\n\n---\n\nTask:\n"
        + task_instruction.strip()
    )
