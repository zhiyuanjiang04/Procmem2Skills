"""Qwen-embedding top-K retrieval over the prebuilt 44787-skill FAISS index.

Reuses the existing index at:
  data/embeddings/skill_embeddings.npy   (44787 x 1024 fp32)
  data/embeddings/index/index.faiss      (IndexFlatIP)
  data/embeddings/skill_metadata.jsonl   (id, slug, name, category)
  data/processed/skill_corpus.jsonl      (full bodies w/ description)

Encoder: Qwen/Qwen3-Embedding-0.6B via SentenceTransformer
(matches procmem2skills/src/retrieval/encoder.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

from .prompt import SkillCandidate

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = REPO_ROOT / "data" / "embeddings" / "index" / "index.faiss"
META_PATH = REPO_ROOT / "data" / "embeddings" / "skill_metadata.jsonl"
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "skill_corpus.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


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
def _load_index() -> "faiss.Index":
    return faiss.read_index(str(INDEX_PATH))


@lru_cache(maxsize=1)
def _load_encoder(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
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
    """Return 1x1024 task embedding. Looks up disk cache by task description
    first (populated by `embed_tasks.py` on GPU); falls back to CPU encode."""
    cache = _load_task_emb_cache()
    if text in cache:
        return cache[text].reshape(1, -1)
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
