"""Compute Qwen3-Embedding-0.6B embeddings for SkillsBench pilot tasks and their GT skills.

Output: skills retrieval/pools/tasks_gt_embeddings.npz with keys:
  task_ids, task_embeddings (T, 1024)
  gt_ids, gt_offsets (T+1,) int64, gt_embeddings (sum(n_gt), 1024)

Uses same model as data/embeddings/skill_embeddings.npy (Qwen3-Embedding-0.6B, 1024 dim).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def embed_texts(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return emb.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="data/selection_collapse/skillsbench/tasks.jsonl")
    p.add_argument("--out", default="skills retrieval/pools/tasks_gt_embeddings.npz")
    p.add_argument("--task_ids", nargs="+", default=["sb_000", "sb_003", "sb_004", "sb_006", "sb_007"])
    args = p.parse_args()

    tasks: list[dict] = []
    with Path(args.tasks).open() as f:
        for line in f:
            row = json.loads(line)
            if row["task_id"] in args.task_ids:
                tasks.append(row)
    # preserve requested order
    tasks.sort(key=lambda t: args.task_ids.index(t["task_id"]))

    task_ids = [t["task_id"] for t in tasks]
    task_instr = [t["instruction"] for t in tasks]
    gt_ids: list[str] = []
    gt_texts: list[str] = []
    gt_offsets = [0]
    for t in tasks:
        for g in t.get("gt_skills", []):
            gt_ids.append(f"gt_{t['task_id']}_{g['name']}")
            gt_texts.append(g.get("content", g["name"]))
        gt_offsets.append(len(gt_ids))

    task_emb = embed_texts(task_instr)
    gt_emb = embed_texts(gt_texts) if gt_texts else np.zeros((0, task_emb.shape[1]), dtype=np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        task_ids=np.asarray(task_ids),
        task_embeddings=task_emb.astype(np.float32),
        gt_ids=np.asarray(gt_ids),
        gt_offsets=np.asarray(gt_offsets, dtype=np.int64),
        gt_embeddings=gt_emb.astype(np.float32),
    )
    print(f"Saved {len(task_ids)} tasks, {len(gt_ids)} GTs (dim={task_emb.shape[1]}) → {out}")


if __name__ == "__main__":
    main()
