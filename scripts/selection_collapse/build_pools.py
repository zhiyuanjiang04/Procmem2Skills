#!/usr/bin/env python3
"""
Build skill pools for the Selection Collapse experiment.

For each (task, pool_size, trial), constructs a pool containing:
- The task's ground-truth (GT) skills
- Randomly sampled ClawHub distractors to fill the remaining slots

GT skill positions are randomized. Each skill gets a sequential display_id.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Constants ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
SELECTED_TASKS_PATH = ROOT / "data" / "selection_collapse" / "skillsbench" / "selected_tasks.jsonl"
CORPUS_PATH = ROOT / "data" / "processed" / "skill_corpus.jsonl"
POOLS_DIR = ROOT / "data" / "selection_collapse" / "pools"

POOL_SIZES = [5, 50, 200, 1000]
N_TRIALS = 3
BASE_SEED = 42_000


def build_pool(
    task: dict[str, Any],
    corpus: list[dict[str, Any]],
    pool_size: int,
    seed: int,
    hard_negatives: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a single skill pool for a task.

    Args:
        task: Task dict with task_id, gt_skills, etc.
        corpus: List of ClawHub skill dicts (id, slug, name, description, body).
        pool_size: Total number of skills in the pool (GT + distractors).
        seed: Random seed for reproducibility.
        hard_negatives: Optional list of hard negative skill dicts to include
                        before random distractors.

    Returns:
        List of skill dicts, shuffled, with sequential display_ids.
    """
    rng = random.Random(seed)
    gt_skills = task["gt_skills"]
    n_gt = len(gt_skills)

    if pool_size < n_gt:
        raise ValueError(
            f"pool_size ({pool_size}) must be >= number of GT skills ({n_gt})"
        )

    # Build GT entries
    gt_ids = {s["skill_id"] for s in gt_skills}
    pool_entries = []
    for s in gt_skills:
        pool_entries.append({
            "skill_id": s["skill_id"],
            "name": s["name"],
            "description": s.get("content", "")[:200],
            "content": s.get("content", ""),
            "is_gt": True,
            "source": "gt",
        })

    # Number of distractors needed
    n_distractors = pool_size - n_gt

    # Hard negatives first (if provided)
    hard_neg_entries = []
    if hard_negatives:
        for hn in hard_negatives[:n_distractors]:
            hard_neg_entries.append({
                "skill_id": f"dist_{hn['id']}",
                "name": hn.get("name", ""),
                "description": hn.get("description", ""),
                "content": hn.get("body", ""),
                "is_gt": False,
                "source": "hard_negative",
            })

    n_random = n_distractors - len(hard_neg_entries)

    # Sample random distractors from corpus (excluding any GT skill IDs)
    candidates = [c for c in corpus if c["id"] not in gt_ids]
    sampled = rng.sample(candidates, min(n_random, len(candidates)))

    distractor_entries = []
    for c in sampled:
        distractor_entries.append({
            "skill_id": f"dist_{c['id']}",
            "name": c.get("name", ""),
            "description": c.get("description", ""),
            "content": c.get("body", ""),
            "is_gt": False,
            "source": "clawhub_distractor",
        })

    pool_entries.extend(hard_neg_entries)
    pool_entries.extend(distractor_entries)

    # Shuffle the whole pool
    rng.shuffle(pool_entries)

    # Assign sequential display_ids based on final position
    for i, entry in enumerate(pool_entries):
        entry["display_id"] = f"SKILL_{i + 1:03d}"

    return pool_entries


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_all_pools() -> None:
    """Build pools for all tasks x pool_sizes x trials and save to disk."""
    print(f"Loading corpus from {CORPUS_PATH} ...")
    corpus = _load_jsonl(CORPUS_PATH)
    print(f"  Loaded {len(corpus)} skills")

    print(f"Loading tasks from {SELECTED_TASKS_PATH} ...")
    tasks = _load_jsonl(SELECTED_TASKS_PATH)
    print(f"  Loaded {len(tasks)} tasks")

    POOLS_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for task in tasks:
        task_id = task["task_id"]
        gt_ids = [s["skill_id"] for s in task["gt_skills"]]
        n_gt = len(gt_ids)

        for pool_size in POOL_SIZES:
            if pool_size < n_gt:
                print(f"  SKIP {task_id} pool_size={pool_size} (only {n_gt} GT skills)")
                continue

            for trial in range(N_TRIALS):
                seed = BASE_SEED + hash((task_id, pool_size, trial)) % 100_000
                pool = build_pool(task, corpus, pool_size=pool_size, seed=seed)

                record = {
                    "task_id": task_id,
                    "pool_size": pool_size,
                    "trial": trial,
                    "seed": seed,
                    "condition": "random",
                    "pool": pool,
                    "gt_skill_ids": gt_ids,
                }

                fname = f"{task_id}_ps{pool_size}_t{trial}.json"
                out_path = POOLS_DIR / fname
                with open(out_path, "w") as f:
                    json.dump(record, f, indent=2)
                total += 1

    print(f"Done. Wrote {total} pool files to {POOLS_DIR}")


if __name__ == "__main__":
    build_all_pools()
