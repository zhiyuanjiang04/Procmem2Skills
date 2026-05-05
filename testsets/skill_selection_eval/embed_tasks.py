"""Precompute Qwen embeddings for all SkillsBench task descriptions.

Run on GPU (slurm). Output is a dict keyed by task_id mapping to a 1024-dim
fp32 vector, stored as npz. retrieve.encode_query() looks up this cache
first (avoiding CPU-bound re-encoding inside the event loop during grid runs).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .retrieve import REPO_ROOT, _load_encoder

OUT = REPO_ROOT / "testsets" / "embeddings" / "task_description_embeddings.npy"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sources = [
        REPO_ROOT / "testsets" / "data" / "skillsbench_tasks.jsonl",
        REPO_ROOT / "testsets" / "data" / "terminal_bench_tasks.jsonl",
    ]
    rows: list[dict] = []
    for src in sources:
        if not src.exists():
            print(f"SKIP missing source: {src}")
            continue
        for l in src.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            rows.append({"task_id": f"{src.stem}::{r['task_id']}",
                         "description": r["task_description"]})
    print(f"Encoding {len(rows)} task descriptions from {len(sources)} sources...")
    enc = _load_encoder()
    descs = [r["description"] for r in rows]
    embs = enc.encode(descs, normalize_embeddings=True, convert_to_numpy=True,
                      batch_size=8, show_progress_bar=True).astype(np.float32)
    np.save(OUT, embs)
    with open(str(OUT).replace(".npy", "_keys.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved -> {OUT} ({len(rows)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
