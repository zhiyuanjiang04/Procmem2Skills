"""Embed SkillsBench GT skills into Qwen-3-Embedding-0.6B space.

For each task in skillsbench_repo/tasks/<task>/environment/skills/<gt>/SKILL.md:
  - Extract frontmatter description (fallback: first 60 tokens of body)
  - Embed with the same encoder used for ClawHub corpus
  - Detect ClawHub alias: cosine_max > 0.9 against any ClawHub skill embedding
  - Store everything in testsets/embeddings/

Output:
  testsets/embeddings/skillsbench_gt_embeddings.npy   (N x 1024 fp32)
  testsets/embeddings/skillsbench_gt_metadata.jsonl   (per row: task_id, gt_name, description, clawhub_alias_id, alias_sim)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .retrieve import REPO_ROOT, _load_corpus, _load_encoder

TASKS_DIR = Path("/anvil/projects/x-cis260386/william/procmem2skills/skillsbench_repo/tasks")
OUT_DIR = REPO_ROOT / "testsets" / "embeddings"
OUT_VEC = OUT_DIR / "skillsbench_gt_embeddings.npy"
OUT_META = OUT_DIR / "skillsbench_gt_metadata.jsonl"

# alias threshold; user-specified
ALIAS_SIM_THRESHOLD = 0.9
DESC_MAX_CHARS = 240  # ~60 tokens, mirrors prompt-side truncation


def parse_frontmatter_description(text: str) -> str | None:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    desc_lines: list[str] = []
    in_desc = False
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            break
        m = re.match(r"description\s*:\s*(.*)", line, re.IGNORECASE)
        if m:
            in_desc = True
            val = m.group(1).strip().strip('"').strip("'")
            if val:
                desc_lines.append(val)
        elif in_desc:
            # multi-line YAML continuation: indented or starts with >|
            if line.startswith((" ", "\t")) or stripped.startswith((">", "|")):
                desc_lines.append(stripped)
            else:
                break
    return " ".join(desc_lines).strip() or None


def fallback_body_snippet(text: str, max_chars: int = DESC_MAX_CHARS) -> str:
    # Strip frontmatter
    if text.startswith("---"):
        parts = text.split("---", 2)
        body = parts[2] if len(parts) >= 3 else text
    else:
        body = text
    # Take first non-empty paragraph
    snippet = " ".join(line.strip() for line in body.splitlines() if line.strip())
    return snippet[:max_chars]


def collect_gt_records() -> list[dict]:
    records: list[dict] = []
    for task_dir in sorted(TASKS_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        skills_dir = task_dir / "environment" / "skills"
        if not skills_dir.exists():
            continue
        for gt_dir in sorted(skills_dir.iterdir()):
            if not gt_dir.is_dir():
                continue
            skill_md = gt_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            text = skill_md.read_text(encoding="utf-8", errors="replace")
            desc = parse_frontmatter_description(text) or fallback_body_snippet(text)
            desc = desc[:DESC_MAX_CHARS]
            records.append({
                "task_id": task_dir.name,
                "gt_name": gt_dir.name,
                "description": desc,
            })
    return records


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = collect_gt_records()
    print(f"GT records: {len(records)}")
    if not records:
        print("No GT records found; aborting.")
        return 1

    enc = _load_encoder()
    descriptions = [r["description"] or r["gt_name"] for r in records]
    embs = enc.encode(descriptions, normalize_embeddings=True, convert_to_numpy=True, batch_size=16, show_progress_bar=True)
    embs = embs.astype(np.float32)

    # Alias detection vs ClawHub
    clawhub_corpus = _load_corpus()
    clawhub_emb = np.load(REPO_ROOT / "data" / "embeddings" / "skill_embeddings.npy", mmap_mode="r")
    # Cosine = dot since both are already normalized (FAISS index is IP).
    sims = embs @ clawhub_emb.T  # (n_gt, 44787)
    best_idx = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(records)), best_idx]

    for i, rec in enumerate(records):
        sim = float(best_sim[i])
        rec["alias_sim"] = sim
        if sim >= ALIAS_SIM_THRESHOLD:
            ch = clawhub_corpus[int(best_idx[i])]
            rec["clawhub_alias_id"] = ch.id
            rec["clawhub_alias_slug"] = ch.slug
        else:
            rec["clawhub_alias_id"] = None
            rec["clawhub_alias_slug"] = None

    np.save(OUT_VEC, embs)
    with OUT_META.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    aliased = sum(1 for r in records if r["clawhub_alias_id"])
    print(f"Embeddings -> {OUT_VEC}")
    print(f"Metadata   -> {OUT_META}")
    print(f"Aliased to ClawHub (sim>={ALIAS_SIM_THRESHOLD}): {aliased}/{len(records)}")
    print(f"Mean alias_sim: {float(best_sim.mean()):.3f}, max: {float(best_sim.max()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
