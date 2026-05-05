"""Terminal-bench candidate-pool builder.

TB has no validated GT skills (P4 scope decision: no harbor execution).
For each task we treat the Qwen top-K retrieval as the "candidate" set.
Pool composition (per task):

    pool = topk_subset ∪ noise(N − |topk_subset|, mode)

`noise_mode`:
    "topk_only": pool == top-K, size==K
    "random"    : top-K + uniform-random ClawHub skills (excluding top-K ids)
    "hard"      : top-K + nearest ClawHub neighbours of task description (excl top-K)
    "easy"      : top-K + farthest ClawHub neighbours
    "none"      : pure noise, no top-K (refusal sanity check, no proxy-GT in pool)
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .pool_builder import _stable_seed, _truncate, _sample_random_noise, _sample_neighbour_noise
from .prompt import SkillCandidate
from .retrieve import REPO_ROOT


@dataclass
class TBPool:
    candidates: list[SkillCandidate]
    topk_positions: list[int]   # indices in shuffled pool of top-K skills present
    topk_names: list[str]       # the top-K skill names (slugs) actually placed
    proxy_gt_name: str | None   # top-1 retrieval slug (or None if topk empty)


@lru_cache(maxsize=1)
def _load_topk_table() -> dict[str, list[dict]]:
    path = REPO_ROOT / "testsets" / "data" / "terminal_bench_topk.jsonl"
    out: dict[str, list[dict]] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["task_id"]] = row.get("topk", [])
    return out


def _topk_to_candidates(topk: list[dict], n: int) -> list[tuple[SkillCandidate, str, str]]:
    """Take first n entries; return (candidate, slug, skill_id) triples."""
    out = []
    for entry in topk[:n]:
        slug = entry.get("slug") or entry.get("name") or entry.get("skill_id")
        cand = SkillCandidate(name=slug, description=_truncate(entry.get("description", "")))
        out.append((cand, slug, entry["skill_id"]))
    return out


def build_tb_pool(
    *,
    task_id: str,
    task_description: str,
    pool_size: int,
    noise_mode: str,
    topk_in_pool: int = 5,
    seed: int = 0,
) -> TBPool:
    rng = random.Random(_stable_seed("tb", task_id, pool_size, noise_mode, seed))
    topk_entries = _load_topk_table().get(task_id, [])
    topk_used = _topk_to_candidates(topk_entries, topk_in_pool)
    topk_cands = [c for c, _, _ in topk_used]
    topk_slugs = [s for _, s, _ in topk_used]
    topk_ids = {sid for _, _, sid in topk_used}
    proxy_gt = topk_slugs[0] if topk_slugs else None

    if noise_mode == "topk_only":
        pool_list = list(topk_cands)
        # No shuffle for topk_only — preserve retrieval rank order.
        name_to_idx = {c.name: i for i, c in enumerate(pool_list)}
        positions = [name_to_idx[s] for s in topk_slugs if s in name_to_idx]
        return TBPool(candidates=pool_list, topk_positions=positions,
                      topk_names=topk_slugs, proxy_gt_name=proxy_gt)

    if noise_mode == "none":
        n_noise = pool_size
        noise = _sample_random_noise(n_noise, topk_ids, rng)
        rng.shuffle(noise)
        return TBPool(candidates=noise, topk_positions=[], topk_names=[], proxy_gt_name=None)

    n_noise = max(0, pool_size - len(topk_cands))
    if noise_mode == "random":
        noise = _sample_random_noise(n_noise, topk_ids, rng)
    elif noise_mode in ("hard", "easy"):
        from .retrieve import encode_query
        q_emb = encode_query(task_description)
        noise = _sample_neighbour_noise(q_emb, n_noise, topk_ids, mode=noise_mode)
    else:
        raise ValueError(f"Unknown noise_mode: {noise_mode}")

    pool_list = topk_cands + noise
    rng.shuffle(pool_list)
    name_to_idx = {c.name: i for i, c in enumerate(pool_list)}
    positions = [name_to_idx[s] for s in topk_slugs if s in name_to_idx]
    return TBPool(candidates=pool_list, topk_positions=positions,
                  topk_names=topk_slugs, proxy_gt_name=proxy_gt)
