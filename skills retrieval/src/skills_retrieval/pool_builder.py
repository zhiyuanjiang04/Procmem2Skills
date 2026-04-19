from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .config import PoolSpec, Strategy
from .data import Corpus, Task

COSINE_EPS_DUP = 0.85
COSINE_EPS_FUNC_GUARD = 0.80


@dataclass
class Pool:
    spec: PoolSpec
    display_ids: list[str]
    id_map: dict[str, str] = field(default_factory=dict)
    cards: dict[str, dict] = field(default_factory=dict)
    gt_display_ids: list[str] = field(default_factory=list)


def _rng(seed: int, extra: str) -> np.random.Generator:
    h = hashlib.md5(f"{seed}:{extra}".encode()).digest()[:8]
    return np.random.default_rng(np.frombuffer(h, dtype=np.uint64)[0])


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return a @ b.T


def _sample_random(corpus: Corpus, n_needed: int, exclude: set[int], rng: np.random.Generator) -> list[int]:
    all_idx = np.arange(len(corpus))
    mask = np.ones(len(corpus), dtype=bool)
    for i in exclude:
        mask[i] = False
    pool = all_idx[mask]
    picks = rng.choice(pool, size=min(n_needed, len(pool)), replace=False)
    return picks.tolist()


def _sample_hard_neg_semantic(corpus: Corpus, task_embedding: np.ndarray, n_needed: int, exclude: set[int]) -> list[int]:
    sims = _cosine(task_embedding[None, :], corpus.embeddings).ravel()
    order = np.argsort(-sims)
    picks: list[int] = []
    for idx in order:
        if int(idx) in exclude:
            continue
        picks.append(int(idx))
        if len(picks) >= n_needed:
            break
    return picks


def build_pool(
    spec: PoolSpec,
    task: Task,
    corpus: Corpus,
    task_embedding: np.ndarray,
    gt_entries: list[tuple[str, str, str, np.ndarray]] | None = None,
) -> Pool:
    if gt_entries is None:
        gt_entries = [
            (f"gt_{task.task_id}_{name}", name, body, np.zeros(corpus.embeddings.shape[1], dtype=np.float32))
            for name, body in zip(task.gt_skill_names, task.gt_skill_bodies)
        ]

    n_gt = len(gt_entries)
    n_distractors = max(0, spec.n - n_gt)
    if n_gt > spec.n:
        gt_entries = gt_entries[: spec.n]
        n_gt = len(gt_entries)
        n_distractors = 0

    gt_emb = np.stack([e[3] for e in gt_entries]) if gt_entries else np.zeros((0, corpus.embeddings.shape[1]), dtype=np.float32)

    exclude: set[int] = set()
    if n_distractors > 0 and n_gt > 0 and np.any(gt_emb):
        sims = _cosine(corpus.embeddings, gt_emb)
        max_sim = sims.max(axis=1)
        exclude = set(int(i) for i, s in enumerate(max_sim) if s > COSINE_EPS_DUP)

    rng = _rng(spec.seed, f"{spec.task_id}:{spec.strategy}:{spec.n}:distractor")
    if spec.strategy == "random":
        distractor_idx = _sample_random(corpus, n_distractors, exclude, rng)
    elif spec.strategy == "hard_neg_semantic":
        distractor_idx = _sample_hard_neg_semantic(corpus, task_embedding, n_distractors, exclude)
    else:
        raise NotImplementedError(f"Strategy {spec.strategy} is Plan 2")

    entries: list[tuple[str, str, str]] = []
    for gt_id, gt_name, gt_body, _ in gt_entries:
        entries.append((gt_id, gt_name, gt_body))
    for idx in distractor_idx:
        entries.append((corpus.ids[idx], corpus.names[idx], corpus.descriptions[idx]))

    order_rng = _rng(spec.seed, f"{spec.task_id}:{spec.strategy}:{spec.n}:order")
    order = np.arange(len(entries))
    order_rng.shuffle(order)

    display_ids = [f"SKILL_{k:03d}" for k in range(len(entries))]
    id_map: dict[str, str] = {}
    cards: dict[str, dict] = {}
    gt_display_ids: list[str] = []
    gt_canonical = {g[0] for g in gt_entries}

    for k, original_idx in enumerate(order):
        canonical_id, name, body_or_desc = entries[original_idx]
        did = display_ids[k]
        id_map[did] = canonical_id
        cards[did] = {"name": name, "description": body_or_desc[:200] if body_or_desc else "", "body": body_or_desc}
        if canonical_id in gt_canonical:
            gt_display_ids.append(did)

    return Pool(spec=spec, display_ids=display_ids, id_map=id_map, cards=cards, gt_display_ids=gt_display_ids)
