#!/usr/bin/env python3
"""
Build hard-negative distractor pools for the Selection Collapse experiment.

For each GT skill in the selected tasks, finds the most embedding-similar
ClawHub skills (using Qwen3-Embedding-0.6B) and filters near-duplicates
by word-level Jaccard overlap. The resulting hard negatives are saved
per-task, then assembled into N=50 pools.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["HF_HOME"] = "/anvil/projects/x-cis260386/william/hf_cache"

# ── Constants ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "skill_embeddings.npy"
METADATA_PATH = ROOT / "data" / "embeddings" / "skill_metadata.jsonl"
CORPUS_PATH = ROOT / "data" / "processed" / "skill_corpus.jsonl"
SELECTED_TASKS_PATH = ROOT / "data" / "selection_collapse" / "skillsbench" / "selected_tasks.jsonl"
HARD_NEG_DIR = ROOT / "data" / "selection_collapse" / "hard_negatives"

TOP_K_CANDIDATES = 200
MAX_HARD_NEGATIVES = 100
JACCARD_THRESHOLD = 0.90
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _word_set(text: str) -> set[str]:
    """Lowercase word set for Jaccard computation."""
    return set(text.lower().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    """Word-level Jaccard similarity."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _get_skill_text(skill_dict: dict) -> str:
    """Extract the best available text representation of a skill."""
    # For GT skills, content is the main field
    if "content" in skill_dict and skill_dict["content"]:
        return skill_dict["content"]
    # For corpus skills, try full_text > body > description
    for field in ("full_text", "body", "description"):
        if field in skill_dict and skill_dict[field]:
            return skill_dict[field]
    return skill_dict.get("name", "")


def build_hard_negatives() -> None:
    """Compute hard negatives for all selected tasks and save to disk."""
    from sentence_transformers import SentenceTransformer

    print("Loading skill embeddings ...")
    embeddings = np.load(EMBEDDINGS_PATH)  # (44787, 1024)
    print(f"  Shape: {embeddings.shape}")

    # Normalize embeddings for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings_normed = embeddings / norms

    print("Loading skill metadata ...")
    metadata = _load_jsonl(METADATA_PATH)
    print(f"  {len(metadata)} entries")

    print("Loading skill corpus ...")
    corpus = _load_jsonl(CORPUS_PATH)
    corpus_by_id = {c["id"]: c for c in corpus}
    print(f"  {len(corpus)} skills")

    print("Loading selected tasks ...")
    tasks = _load_jsonl(SELECTED_TASKS_PATH)
    print(f"  {len(tasks)} tasks")

    print(f"Loading encoder model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    print("  Model loaded.")

    HARD_NEG_DIR.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        task_id = task["task_id"]
        gt_skills = task["gt_skills"]
        gt_skill_ids = {s["skill_id"] for s in gt_skills}

        # Collect all hard negative candidates across all GT skills for this task
        # Use a dict keyed by corpus skill id to deduplicate
        all_candidates: dict[str, dict] = {}

        for gt_skill in gt_skills:
            gt_text = _get_skill_text(gt_skill)
            gt_words = _word_set(gt_text)

            # Encode GT skill text
            gt_embedding = model.encode([gt_text], normalize_embeddings=True)
            gt_embedding = gt_embedding.astype(np.float32)

            # Cosine similarity via dot product (both normalized)
            similarities = embeddings_normed @ gt_embedding.T  # (44787, 1)
            similarities = similarities.squeeze()

            # Top-K indices
            top_indices = np.argsort(similarities)[::-1][:TOP_K_CANDIDATES]

            for idx in top_indices:
                meta = metadata[idx]
                skill_id = meta["id"]

                # Skip if it's a GT skill (by corpus id)
                if skill_id in gt_skill_ids:
                    continue
                # Also skip GT skills matched by slug prefix
                if any(skill_id in sid for sid in gt_skill_ids):
                    continue

                # Skip if already selected as candidate with higher similarity
                if skill_id in all_candidates:
                    # Keep the higher similarity score
                    if similarities[idx] > all_candidates[skill_id]["similarity"]:
                        all_candidates[skill_id]["similarity"] = float(similarities[idx])
                    continue

                # Get corpus entry for description text
                corpus_entry = corpus_by_id.get(skill_id)
                if corpus_entry is None:
                    continue

                candidate_text = _get_skill_text(corpus_entry)
                candidate_words = _word_set(candidate_text)

                # Filter near-duplicates by Jaccard overlap
                jacc = _jaccard(gt_words, candidate_words)
                if jacc > JACCARD_THRESHOLD:
                    continue

                all_candidates[skill_id] = {
                    "id": skill_id,
                    "slug": meta.get("slug", ""),
                    "name": meta.get("name", ""),
                    "category": meta.get("category", ""),
                    "description": corpus_entry.get("description", ""),
                    "body": corpus_entry.get("body", ""),
                    "similarity": float(similarities[idx]),
                    "jaccard_with_gt": float(jacc),
                }

        # Sort by similarity (descending) and keep top MAX_HARD_NEGATIVES
        hard_negs = sorted(
            all_candidates.values(),
            key=lambda x: x["similarity"],
            reverse=True,
        )[:MAX_HARD_NEGATIVES]

        out_path = HARD_NEG_DIR / f"{task_id}_hard_negatives.json"
        record = {
            "task_id": task_id,
            "n_gt_skills": len(gt_skills),
            "n_hard_negatives": len(hard_negs),
            "hard_negatives": hard_negs,
        }
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

        print(f"  {task_id}: {len(hard_negs)} hard negatives saved")

    print(f"Done. Hard negatives saved to {HARD_NEG_DIR}")


if __name__ == "__main__":
    build_hard_negatives()

    # Build hard-negative pools at N=50
    # Import from sibling module in the same directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_pools import build_hard_negative_pools
    build_hard_negative_pools()
