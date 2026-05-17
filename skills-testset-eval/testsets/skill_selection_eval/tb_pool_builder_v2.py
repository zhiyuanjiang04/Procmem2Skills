"""TB pool builder V2: uses validated GT skills directly (not FAISS proxy-GT).

Pool composition (per task):
    pool = gt_skills ∪ noise(N - |gt_skills|, mode)

Step 1: GT skills (from terminal_bench_validated.jsonl gt_skills[]) are placed in the pool.
        Each GT has a skill_id for corpus lookup and a slug as the canonical name.
Step 2: Distractors fill remaining slots.

noise_mode:
    "random": uniform random ClawHub skills (excluding GT skill_ids)
    "hard":   nearest ClawHub skills to task description embedding (excl GT) — semantically close
    "easy":   farthest ClawHub skills from task description embedding (excl GT) — semantically far
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache

from .pool_builder import _stable_seed, _truncate, _sample_random_noise, _sample_neighbour_noise
from .prompt import SkillCandidate
from .retrieve import _load_corpus


@dataclass
class TBPoolV2:
    candidates: list[SkillCandidate]
    gt_positions: list[int]   # indices into candidates of GT skills (after shuffle)
    gt_names: list[str]       # GT slugs placed in pool (canonical identifiers for Hit@1)


@lru_cache(maxsize=1)
def _corpus_by_id() -> dict[str, object]:
    return {r.id: r for r in _load_corpus()}


@lru_cache(maxsize=1)
def _corpus_by_slug() -> dict[str, object]:
    return {r.slug: r for r in _load_corpus()}


def _gt_to_candidate(gt_skill: dict) -> tuple[SkillCandidate, str, str]:
    """Returns (candidate, slug, skill_id) from a validated GT entry.

    Looks up description by skill_id first, then falls back to slug-based lookup
    (needed when corpus was rebuilt with new sequential IDs from all_skills_comprehensive.json).
    """
    skill_id = gt_skill["skill_id"]
    slug = gt_skill["slug"]
    by_id = _corpus_by_id()
    rec = by_id.get(skill_id)
    if rec is None:
        # Corpus was rebuilt; try matching by slug (author/name).
        rec = _corpus_by_slug().get(slug)
    desc = _truncate(rec.description) if rec else "No description provided."
    # Resolve the actual corpus ID for exclusion from distractors.
    actual_id = rec.id if rec else skill_id
    return SkillCandidate(name=slug, description=desc), slug, actual_id


def build_tb_pool_v2(
    *,
    task_id: str,
    task_description: str,
    gt_skills: list[dict],
    pool_size: int,
    noise_mode: str,
    seed: int = 0,
) -> TBPoolV2:
    """Build a candidate pool for a terminal-bench task.

    gt_skills: list of dicts from terminal_bench_validated.jsonl, each with
               {"skill_id": ..., "slug": ..., "name": ..., ...}
    pool_size: total number of candidates (GT + distractors)
    noise_mode: "random" | "hard" | "easy"
    seed: for reproducibility (each (task_id, pool_size, noise_mode, seed) is deterministic)
    """
    rng = random.Random(_stable_seed("tb_v2", task_id, pool_size, noise_mode, seed))

    gt_triples = [_gt_to_candidate(g) for g in gt_skills]
    gt_cands = [c for c, _, _ in gt_triples]
    gt_slugs = [s for _, s, _ in gt_triples]
    gt_ids = {sid for _, _, sid in gt_triples}

    n_noise = max(0, pool_size - len(gt_cands))

    if noise_mode == "random":
        noise = _sample_random_noise(n_noise, gt_ids, rng)
    elif noise_mode in ("hard", "easy"):
        from .retrieve import encode_query
        q_emb = encode_query(task_description)
        noise = _sample_neighbour_noise(q_emb, n_noise, gt_ids, mode=noise_mode)
    else:
        raise ValueError(f"Unknown noise_mode: {noise_mode!r} — must be random|hard|easy")

    pool_list = gt_cands + noise
    rng.shuffle(pool_list)
    name_to_idx = {c.name: i for i, c in enumerate(pool_list)}
    gt_positions = [name_to_idx[s] for s in gt_slugs if s in name_to_idx]
    return TBPoolV2(candidates=pool_list, gt_positions=gt_positions, gt_names=gt_slugs)
