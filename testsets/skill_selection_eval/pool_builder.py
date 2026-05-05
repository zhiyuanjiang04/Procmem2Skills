"""Build candidate pools for the controlled study.

Pool composition (per task):
    pool = gt_subsampled ∪ noise(N - |gt_subsampled|, mode)

Variables:
    pool_size  ∈ {5, 50, 500, 1500}
    noise_mode ∈ {"random", "hard", "easy", "none"}   # "none" => no-GT condition: pure noise

Returns SkillCandidate list (shuffled), plus a record of GT positions in the
shuffled pool (post-shuffle index) for downstream position-bias analysis.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _stable_seed(*parts: object) -> int:
    """Deterministic int seed from hashable inputs (unlike Python's hash())."""
    h = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(h[:8], "big")

import numpy as np

from .prompt import SkillCandidate
from .retrieve import REPO_ROOT, _load_corpus

DESC_MAX_CHARS = 240  # ~60 tokens cap per skill description


@dataclass
class Pool:
    candidates: list[SkillCandidate]
    gt_positions: list[int]   # indices into `candidates` of ground-truth skills (empty if no_gt)
    gt_names: list[str]       # actual GT names placed in pool (post-subsampling/aliasing)


@lru_cache(maxsize=1)
def _load_gt_meta() -> list[dict]:
    path = REPO_ROOT / "testsets" / "embeddings" / "skillsbench_gt_metadata.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@lru_cache(maxsize=1)
def _load_gt_embeddings() -> np.ndarray:
    path = REPO_ROOT / "testsets" / "embeddings" / "skillsbench_gt_embeddings.npy"
    return np.load(path)


@lru_cache(maxsize=1)
def _load_clawhub_embeddings() -> np.ndarray:
    return np.load(REPO_ROOT / "data" / "embeddings" / "skill_embeddings.npy", mmap_mode="r")


def _truncate(text: str) -> str:
    return (text or "").strip()[:DESC_MAX_CHARS] or "No description provided."


def _gt_records_for_task(task_id: str) -> list[dict]:
    return [r for r in _load_gt_meta() if r["task_id"] == task_id]


def _subsample_gt(gt_records: list[dict], max_gt: int, rng: random.Random) -> list[dict]:
    if max_gt <= 0 or len(gt_records) <= max_gt:
        return list(gt_records)
    return rng.sample(gt_records, max_gt)


def _gt_to_candidate(rec: dict) -> tuple[SkillCandidate, str]:
    """Materialise a GT record as a pool candidate.

    Prefer the ClawHub alias slug as the pool-side `name` when available, so
    the model sees only ClawHub-style names (no leakage of internal SkillsBench
    naming convention). Returns (candidate, gt_name_used) where gt_name_used
    is what we count as the GT identifier post-shuffle (alias_slug or gt_name).
    """
    desc = _truncate(rec.get("description") or "")
    if rec.get("clawhub_alias_slug"):
        name = rec["clawhub_alias_slug"]
    else:
        name = rec["gt_name"]
    return SkillCandidate(name=name, description=desc), name


def _sample_random_noise(n: int, exclude_clawhub_ids: set[str], rng: random.Random) -> list[SkillCandidate]:
    corpus = _load_corpus()
    if n <= 0:
        return []
    pool_idx = [i for i, rec in enumerate(corpus) if rec.id not in exclude_clawhub_ids]
    chosen = rng.sample(pool_idx, min(n, len(pool_idx)))
    out = []
    for i in chosen:
        rec = corpus[i]
        out.append(SkillCandidate(
            name=rec.slug or rec.name or rec.id,
            description=_truncate(rec.description),
        ))
    return out


def _sample_neighbour_noise(
    task_description_emb: np.ndarray,
    n: int,
    exclude_clawhub_ids: set[str],
    *,
    mode: str,
) -> list[SkillCandidate]:
    """Hard / easy noise sampler.

    hard: take ClawHub skills with HIGHEST cosine to the task description (most
          confusable distractors).
    easy: take LOWEST cosine (semantically far; should be trivially rejectable).
    """
    if n <= 0:
        return []
    corpus = _load_corpus()
    emb = _load_clawhub_embeddings()
    sims = (task_description_emb @ emb.T).reshape(-1)  # (44787,)
    order = np.argsort(sims)
    if mode == "hard":
        order = order[::-1]   # descending
    elif mode != "easy":
        raise ValueError(f"Unknown neighbour mode: {mode}")
    out: list[SkillCandidate] = []
    for idx in order:
        rec = corpus[int(idx)]
        if rec.id in exclude_clawhub_ids:
            continue
        out.append(SkillCandidate(
            name=rec.slug or rec.name or rec.id,
            description=_truncate(rec.description),
        ))
        if len(out) >= n:
            break
    return out


def build_pool(
    *,
    task_id: str,
    task_description: str,
    pool_size: int,
    noise_mode: str,        # "random" | "hard" | "easy" | "none"
    max_gt: int = 1,
    seed: int = 0,
) -> Pool:
    rng = random.Random(_stable_seed(task_id, pool_size, noise_mode, seed))

    gt_records = _gt_records_for_task(task_id)

    # Build set of ClawHub IDs to exclude from noise (any aliased GT, to prevent leakage).
    exclude = {r["clawhub_alias_id"] for r in gt_records if r.get("clawhub_alias_id")}

    if noise_mode == "none":
        gt_used: list[dict] = []
        gt_cands: list[SkillCandidate] = []
        gt_names_in_pool: list[str] = []
    else:
        gt_used = _subsample_gt(gt_records, max_gt, rng)
        gt_cands_named = [_gt_to_candidate(r) for r in gt_used]
        gt_cands = [c for c, _ in gt_cands_named]
        gt_names_in_pool = [n for _, n in gt_cands_named]

    n_noise = max(0, pool_size - len(gt_cands))
    if noise_mode in ("none", "random"):
        noise = _sample_random_noise(n_noise, exclude, rng)
    elif noise_mode in ("hard", "easy"):
        from .retrieve import encode_query
        q_emb = encode_query(task_description)
        noise = _sample_neighbour_noise(q_emb, n_noise, exclude, mode=noise_mode)
    else:
        raise ValueError(f"Unknown noise_mode: {noise_mode}")

    pool_list = gt_cands + noise
    rng.shuffle(pool_list)
    name_to_idx = {c.name: i for i, c in enumerate(pool_list)}
    gt_positions = [name_to_idx[n] for n in gt_names_in_pool if n in name_to_idx]
    return Pool(candidates=pool_list, gt_positions=gt_positions, gt_names=gt_names_in_pool)
