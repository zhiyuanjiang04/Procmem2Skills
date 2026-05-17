"""Materialize a pool spec into an on-disk skills/ directory the
terminus-2-skills agent can read at /root/.terminus/skills.

For each candidate in the pool:
  - GT skill: copy the real <task>/environment/skills/<slug>/ tree
  - Noise skill: synthesize a stub SKILL.md from the corpus description

The agent walks the dir, reads each SKILL.md frontmatter, presents
<available_skills> with name+description, and (if the model picks one) loads
the SKILL.md body content. So noise picks land on a stub body — there's no
real procedural content to mislead with. This is the cleanest "Step 1: GT in
pool + Step 2: distractors" setup without scraping full noise SKILL.md from
the public hub.

Usage:
    pool_dir = build_skill_dir(
        task_dir="/path/to/skillsbench_repo/tasks/3d-scan-calc",
        gt_slugs=["mesh-analysis"],
        noise_slugs=["foo/bar", ...],
        out_dir="/tmp/eval/<trial>/skills/",
    )
"""
from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _load_corpus_by_slug(corpus_path: str) -> dict[str, dict]:
    out = {}
    for line in Path(corpus_path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["slug"]] = r
            # also index by name for slug-less fallback
            if r.get("name") and r["name"] not in out:
                out[r["name"]] = r
    return out


def _stub_skill_md(slug: str, description: str) -> str:
    name = slug.split("/")[-1] if "/" in slug else slug
    body = description or "No description available."
    return f"""---
name: {name}
description: "{body[:200].replace('"', '')}"
---

# {name}

{body}

## When to Use

Use this skill if your task aligns with the description above.
"""


def build_skill_dir(
    *,
    task_dir: Path,
    gt_slugs: list[str],
    noise_slugs: list[str],
    out_dir: Path,
    corpus_path: Path,
) -> Path:
    """Create a fresh skills/ tree at out_dir with GT (real) + noise (stub)."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # GT skills: copy from task's environment/skills/<gt>/
    src_skills_root = Path(task_dir) / "environment" / "skills"
    for slug in gt_slugs:
        name = slug.split("/")[-1] if "/" in slug else slug
        src = src_skills_root / name
        if not src.exists():
            # Try slug-matched dir (some tasks use the full slug as dirname)
            for cand in src_skills_root.iterdir():
                if cand.name == name or cand.name == slug.replace("/", "-"):
                    src = cand
                    break
        if not src.exists():
            raise FileNotFoundError(f"GT skill source missing: {src}")
        shutil.copytree(src, out_dir / name)

    # Noise skills: stub SKILL.md
    corpus_by_slug = _load_corpus_by_slug(str(corpus_path))
    for slug in noise_slugs:
        rec = corpus_by_slug.get(slug)
        if rec is None:
            # unknown slug — make a minimal stub
            rec = {"description": f"Skill {slug}."}
        name = slug.split("/")[-1] if "/" in slug else slug
        target = out_dir / name
        target.mkdir(exist_ok=True)
        (target / "SKILL.md").write_text(_stub_skill_md(slug, rec.get("description", "")))

    return out_dir
