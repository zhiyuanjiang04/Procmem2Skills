"""Embedding-based top-K retrieval over the prebuilt FAISS skill index.

Reuses the existing index at:
  data/embeddings/skill_embeddings.npy   (N x D fp32)
  data/embeddings/index/index.faiss      (IndexFlatIP)
  data/embeddings/skill_metadata.jsonl   (id, slug, name, category)
  data/processed/skill_corpus.jsonl      (full bodies w/ description)

Encoder: BAAI/bge-small-en-v1.5 (384-dim), CPU-compatible.
Built from skill_docs.json via testsets/setup_corpus_bge.py.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .prompt import SkillCandidate

REPO_ROOT = Path(__file__).resolve().parents[2]

# Allow env-var overrides for paths to large data files not in git.
# Set SKILL_CORPUS_PATH / SKILL_INDEX_PATH / SKILL_META_PATH to point
# to wherever these files live on your system.
INDEX_PATH = Path(os.environ.get("SKILL_INDEX_PATH",
                  str(REPO_ROOT / "data" / "embeddings" / "index" / "index.faiss")))
META_PATH = Path(os.environ.get("SKILL_META_PATH",
                 str(REPO_ROOT / "data" / "embeddings" / "skill_metadata.jsonl")))
CORPUS_PATH = Path(os.environ.get("SKILL_CORPUS_PATH",
                   str(REPO_ROOT / "data" / "processed" / "skill_corpus.jsonl")))
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass
class SkillRecord:
    id: str
    slug: str
    name: str
    description: str


@lru_cache(maxsize=1)
def _load_corpus() -> list[SkillRecord]:
    records: list[SkillRecord] = []
    with CORPUS_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            records.append(
                SkillRecord(
                    id=row["id"],
                    slug=row.get("slug", ""),
                    name=row.get("name") or row.get("slug", ""),
                    description=(row.get("description") or "").strip(),
                )
            )
    return records


@lru_cache(maxsize=1)
def _load_index():
    import faiss  # type: ignore
    return faiss.read_index(str(INDEX_PATH))


@lru_cache(maxsize=1)
def _load_encoder(model_name: str = DEFAULT_MODEL):
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


_TASK_EMB_CACHE_NPY = REPO_ROOT / "testsets" / "embeddings" / "task_description_embeddings.npy"
_TASK_EMB_KEYS = REPO_ROOT / "testsets" / "embeddings" / "task_description_embeddings_keys.jsonl"


@lru_cache(maxsize=1)
def _load_task_emb_cache() -> dict[str, np.ndarray]:
    if not (_TASK_EMB_CACHE_NPY.exists() and _TASK_EMB_KEYS.exists()):
        return {}
    embs = np.load(_TASK_EMB_CACHE_NPY)
    out: dict[str, np.ndarray] = {}
    with _TASK_EMB_KEYS.open() as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            out[row["description"]] = embs[i].astype(np.float32)
    return out


@lru_cache(maxsize=512)
def _encode_query_cached(text: str, model_name: str) -> bytes:
    enc = _load_encoder(model_name)
    vec = enc.encode([text], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    return vec.tobytes()


def encode_query(text: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Return 1xD task embedding. Checks disk cache first; cache hits with
    wrong dimension (e.g. stale Qwen3 cache vs BAAI) are silently bypassed."""
    cache = _load_task_emb_cache()
    if text in cache:
        vec = cache[text].reshape(1, -1)
        expected_dim = _load_encoder(model_name).get_sentence_embedding_dimension()
        if vec.shape[-1] == expected_dim:
            return vec
    raw = _encode_query_cached(text, model_name)
    return np.frombuffer(raw, dtype=np.float32).reshape(1, -1)


def topk(task_description: str, k: int = 5) -> list[SkillCandidate]:
    """Return top-k skill candidates for a task description."""
    corpus = _load_corpus()
    index = _load_index()
    query = encode_query(task_description)
    scores, ids = index.search(query, k)
    out: list[SkillCandidate] = []
    for rank, idx in enumerate(ids[0].tolist()):
        if idx < 0 or idx >= len(corpus):
            continue
        rec = corpus[idx]
        # Use slug as the canonical name (matches the directory name harbor
        # would mount under /root/.claude/skills/<slug>).
        display_name = rec.slug or rec.name or rec.id
        desc = rec.description or "No description provided."
        out.append(SkillCandidate(name=display_name, description=desc))
    return out
