"""GPU-batch version: encode all task descriptions on GPU + FAISS top-K in one shot.

Usage (under slurm with GPU):
    python -m testsets.skill_selection_eval.retrieve_topk_gpu \\
        --tasks testsets/data/terminal_bench_tasks.jsonl \\
        --out   testsets/data/terminal_bench_topk.jsonl \\
        --k 5

Avoids the CPU encoder bottleneck that stalls the asyncio event loop in
the in-process retrieval path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .retrieve import _load_corpus, _load_encoder, _load_index


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]
    descs = [r["task_description"] for r in rows]
    print(f"Encoding {len(descs)} task descriptions on GPU ...")
    enc = _load_encoder()
    embs = enc.encode(descs, normalize_embeddings=True, convert_to_numpy=True,
                      batch_size=8, show_progress_bar=True).astype(np.float32)

    corpus = _load_corpus()
    index = _load_index()
    print(f"FAISS top-{args.k} search ...")
    scores, ids = index.search(embs, args.k)
    out_rows = []
    for i, r in enumerate(rows):
        cands = []
        for rank, (idx, score) in enumerate(zip(ids[i].tolist(), scores[i].tolist()), 1):
            if idx < 0 or idx >= len(corpus):
                continue
            rec = corpus[idx]
            cands.append({
                "rank": rank,
                "skill_id": rec.id,
                "slug": rec.slug,
                "name": rec.name,
                "description": rec.description,
                "similarity": float(score),
            })
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
