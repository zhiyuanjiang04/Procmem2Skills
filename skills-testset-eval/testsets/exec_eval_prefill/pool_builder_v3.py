"""Pool builder for prefill-context execution eval (user spec 2026-05-13).

Differences from skill_selection_eval/pool_builder.py:
  - Pool sizes are EXACT totals (5, 10, 20, 50), not |GT| + n_noise.
  - All GT kept; if |GT| > pool_size we drop the trial (no truncation).
  - Returns (gt_slugs, noise_slugs) — caller materializes the full content.

Noise modes: random | hard (emb-near) | easy (emb-far)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
import sys

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from skill_selection_eval.pool_builder import (  # noqa: E402
    _stable_seed, _load_clawhub_embeddings,
)
from skill_selection_eval.retrieve import _load_corpus  # noqa: E402


@dataclass
class PoolSpec:
    task_id: str
    pool_size: int          # total candidates (including GT)
    noise_mode: str
    seed: int
    gt_slugs: list[str]     # always preserved in pool
    noise_slugs: list[str]  # distractors
    candidates: list[str]   # shuffled (gt + noise)
    gt_positions: list[int]


def _resolve_gt_slugs(dataset: str, task: dict) -> list[str]:
    if dataset == "sb":
        return list(task.get("gt_skills") or [])
    # TB: gt_skills is list of {"slug": "...", "skill_id": "...", ...}
    return [g["slug"] for g in task.get("gt_skills") or []]


def _corpus_ids_for_gt(gt_slugs: list[str]) -> set[str]:
    corpus = _load_corpus()
    by_slug = {r.slug: r for r in corpus}
    by_name = {r.name: r for r in corpus if r.name}
    ids = set()
    for s in gt_slugs:
        r = by_slug.get(s) or by_name.get(s)
        if r:
            ids.add(r.id)
    return ids


def _sample_random_noise(n: int, exclude_ids: set[str], rng: random.Random) -> list[str]:
    if n <= 0:
        return []
    corpus = _load_corpus()
    candidate_idx = [i for i, r in enumerate(corpus) if r.id not in exclude_ids]
    chosen = rng.sample(candidate_idx, min(n, len(candidate_idx)))
    return [corpus[i].slug or corpus[i].name for i in chosen]


def _sample_neighbour_noise(task_desc_emb, n: int, exclude_ids: set[str], mode: str) -> list[str]:
    if n <= 0:
        return []
    import numpy as np
    corpus = _load_corpus()
    emb = _load_clawhub_embeddings()
    sims = (task_desc_emb @ emb.T).reshape(-1)
    order = np.argsort(sims)
    if mode == "hard":
        order = order[::-1]
    elif mode != "easy":
        raise ValueError(mode)
    out = []
    for idx in order:
        rec = corpus[int(idx)]
        if rec.id in exclude_ids:
            continue
        out.append(rec.slug or rec.name)
        if len(out) >= n:
            break
    return out


def build_pool(
    *, dataset: str, task: dict, pool_size: int, noise_mode: str, seed: int = 0,
) -> PoolSpec | None:
    """Build a pool of exactly pool_size candidates with GT preserved.

    Returns None if |GT| > pool_size (task incompatible with this pool size).
    """
    gt_slugs = _resolve_gt_slugs(dataset, task)
    if not gt_slugs:
        return None

    # Baseline modes bypass pool_size constraints
    if noise_mode in ("noskill", "gtonly", "emptyframe"):
        pass  # handled below
    elif noise_mode == "noiseonly":
        pass  # uses pool_size for noise count, no GT
    elif len(gt_slugs) > pool_size:
        return None

    rng = random.Random(_stable_seed("prefill_v3", task["task_id"], pool_size, noise_mode, seed))
    exclude_ids = _corpus_ids_for_gt(gt_slugs)
    n_noise = max(0, pool_size - len(gt_slugs))

    if noise_mode == "noskill":
        # Baseline: no skills at all
        return PoolSpec(
            task_id=task["task_id"], pool_size=0, noise_mode="noskill", seed=seed,
            gt_slugs=gt_slugs, noise_slugs=[], candidates=[], gt_positions=[],
        )
    elif noise_mode == "emptyframe":
        # Control: skill prompt framing but zero skill blocks
        return PoolSpec(
            task_id=task["task_id"], pool_size=0, noise_mode="emptyframe", seed=seed,
            gt_slugs=gt_slugs, noise_slugs=[], candidates=[], gt_positions=[],
        )
    elif noise_mode == "gtonly":
        # Baseline: GT skills only, no noise
        return PoolSpec(
            task_id=task["task_id"], pool_size=len(gt_slugs), noise_mode="gtonly", seed=seed,
            gt_slugs=gt_slugs, noise_slugs=[], candidates=list(gt_slugs),
            gt_positions=list(range(len(gt_slugs))),
        )
    elif noise_mode == "noiseonly":
        # Control: noise skills only, NO GT
        noise_slugs = _sample_random_noise(pool_size, exclude_ids, rng)
        return PoolSpec(
            task_id=task["task_id"], pool_size=pool_size, noise_mode="noiseonly", seed=seed,
            gt_slugs=gt_slugs, noise_slugs=noise_slugs, candidates=list(noise_slugs),
            gt_positions=[],
        )
    elif noise_mode == "random":
        noise_slugs = _sample_random_noise(n_noise, exclude_ids, rng)
    elif noise_mode in ("hard", "easy"):
        from skill_selection_eval.retrieve import encode_query
        q_emb = encode_query(task["task_description"])
        noise_slugs = _sample_neighbour_noise(q_emb, n_noise, exclude_ids, mode=noise_mode)
    else:
        raise ValueError(noise_mode)

    candidates = list(gt_slugs) + list(noise_slugs)
    rng.shuffle(candidates)
    gt_positions = [i for i, s in enumerate(candidates) if s in set(gt_slugs)]

    return PoolSpec(
        task_id=task["task_id"],
        pool_size=pool_size,
        noise_mode=noise_mode,
        seed=seed,
        gt_slugs=gt_slugs,
        noise_slugs=noise_slugs,
        candidates=candidates,
        gt_positions=gt_positions,
    )
