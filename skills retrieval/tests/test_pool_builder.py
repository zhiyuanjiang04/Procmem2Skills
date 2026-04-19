import numpy as np

from skills_retrieval.config import PoolSpec
from skills_retrieval.data import Corpus, Task
from skills_retrieval.pool_builder import build_pool


def _corpus(tiny_corpus) -> Corpus:
    return Corpus.from_paths(tiny_corpus["metadata_path"], tiny_corpus["embeddings_path"])


def _task_with_gt(gt_id: str, gt_embedding) -> Task:
    return Task(task_id="t0", instruction="do a thing", gt_skill_names=[f"name-{int(gt_id.split('_')[-1])}"], gt_skill_bodies=["body"], domain="test")


def test_random_pool_includes_gt_and_has_unique_ids(tiny_corpus):
    corpus = _corpus(tiny_corpus)
    task = _task_with_gt("skill_000003", corpus.embeddings[3])
    spec = PoolSpec(task_id="t0", strategy="random", n=5, seed=0)
    pool = build_pool(spec, task, corpus, task_embedding=corpus.embeddings[3])
    assert len(pool.display_ids) == 5
    assert len(set(pool.display_ids)) == 5
    assert any(cid.startswith("gt_t0_") for cid in pool.id_map.values())


def test_hard_neg_semantic_orders_by_similarity(tiny_corpus):
    corpus = _corpus(tiny_corpus)
    task = _task_with_gt("skill_000003", corpus.embeddings[3])
    spec = PoolSpec(task_id="t0", strategy="hard_neg_semantic", n=4, seed=0)
    pool = build_pool(spec, task, corpus, task_embedding=corpus.embeddings[3])
    canonical_in_pool = [cid for cid in pool.id_map.values() if not cid.startswith("gt_t0_")]
    assert len(canonical_in_pool) == 3


def test_id_randomization_is_seeded_and_stable(tiny_corpus):
    corpus = _corpus(tiny_corpus)
    task = _task_with_gt("skill_000003", corpus.embeddings[3])
    spec = PoolSpec(task_id="t0", strategy="random", n=5, seed=42)
    p1 = build_pool(spec, task, corpus, task_embedding=corpus.embeddings[3])
    p2 = build_pool(spec, task, corpus, task_embedding=corpus.embeddings[3])
    assert p1.display_ids == p2.display_ids
    assert p1.id_map == p2.id_map


def test_n_equals_one_is_gt_only(tiny_corpus):
    corpus = _corpus(tiny_corpus)
    task = _task_with_gt("skill_000003", corpus.embeddings[3])
    spec = PoolSpec(task_id="t0", strategy="random", n=1, seed=0)
    pool = build_pool(spec, task, corpus, task_embedding=corpus.embeddings[3])
    assert len(pool.display_ids) == 1
    assert all(cid.startswith("gt_t0_") for cid in pool.id_map.values())
