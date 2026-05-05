"""Run Qwen top-K retrieval over a tasks JSONL using the prebuilt FAISS index.

For each task (task_id, task_description), retrieve top-K ClawHub skill
candidates (id, slug, description, similarity). Output one row per task to
the specified JSONL.

Used for terminal-bench testset construction: each task gets K=5 candidates
that will then be incrementally injected into harbor (separate script) to
identify ground-truth skills via lift signal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .retrieve import _load_corpus, _load_index, encode_query


def topk_for(task_description: str, k: int) -> list[dict]:
    corpus = _load_corpus()
    index = _load_index()
    q = encode_query(task_description)
    scores, ids = index.search(q, k)
    out = []
    for rank, (idx, score) in enumerate(zip(ids[0].tolist(), scores[0].tolist()), 1):
        if idx < 0 or idx >= len(corpus):
            continue
        rec = corpus[idx]
        out.append({
            "rank": rank,
            "skill_id": rec.id,
            "slug": rec.slug,
            "name": rec.name,
            "description": rec.description,
            "similarity": float(score),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]
    out_rows = []
    for r in rows:
        cands = topk_for(r["task_description"], args.k)
        out_rows.append({
            "task_id": r["task_id"],
            "task_description": r["task_description"],
            "topk": cands,
        })
    args.out.write_text("\n".join(json.dumps(o) for o in out_rows) + "\n")
    print(f"Wrote {len(out_rows)} tasks × top{args.k} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
