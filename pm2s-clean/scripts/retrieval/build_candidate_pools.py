#!/usr/bin/env python3
"""Build deterministic candidate skill pools for retrieval and execution."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import DEFAULT_ROOT, iter_jsonl, tokens, write_json, write_jsonl


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build skill retrieval candidate pools.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--benchmark", default="skillsbench")
    parser.add_argument("--task-skill-map", type=Path, default=None)
    parser.add_argument("--noise-manifest", type=Path, default=None)
    parser.add_argument("--pool-sizes", default="5,10,20,50,100")
    parser.add_argument("--noise-modes", default="random,similar,dissimilar")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--shuffle-candidates", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def task_query(row: dict) -> str:
    primary = row.get("primary_gt_skill") or {}
    return "\n".join(
        [
            str(row.get("task_name") or ""),
            str(row.get("task_description") or ""),
            str(primary.get("description") or ""),
        ]
    )


def cosine_tokens(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / ((len(a) * len(b)) ** 0.5)


def choose_noise(*, task_row: dict, noise_index: list[tuple[dict, set[str]]], mode: str, count: int, seed: int) -> list[dict]:
    if count <= 0:
        return []
    primary = task_row.get("primary_gt_skill") or {}
    gt_names = {
        str(primary.get("skill_name") or "").strip().lower(),
        str(primary.get("skill_slug") or "").strip().lower(),
    }
    candidates = [
        (row, toks) for row, toks in noise_index
        if str(row.get("skill_name") or "").strip().lower() not in gt_names
        and str(row.get("skill_slug") or "").strip().lower() not in gt_names
    ]
    if len(candidates) < count:
        raise RuntimeError(f"not enough noise skills: need={count} available={len(candidates)}")

    rng = random.Random(f"{seed}:{task_row.get('task_name')}:{mode}:{count}")
    if mode == "random":
        return [row for row, _ in rng.sample(candidates, count)]

    query_toks = tokens(task_query(task_row))
    scored = [
        (cosine_tokens(query_toks, toks), row)
        for row, toks in candidates
    ]
    scored.sort(key=lambda pair: (pair[0], str(pair[1].get("skill_name") or "")), reverse=(mode == "similar"))
    return [row for _, row in scored[:count]]


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    task_map = args.task_skill_map or (root / "manifests" / f"task_skill_map.{args.benchmark}.jsonl")
    noise_manifest = args.noise_manifest or (root / "manifests" / "noise_skills.filtered.jsonl")
    pool_sizes = parse_csv_ints(args.pool_sizes)
    modes = parse_csv(args.noise_modes)
    seeds = parse_csv_ints(args.seeds)

    task_rows = list(iter_jsonl(task_map))
    if args.max_tasks:
        task_rows = task_rows[: args.max_tasks]
    noise_rows = list(iter_jsonl(noise_manifest))
    noise_index = [(row, tokens(str(row.get("description") or ""))) for row in noise_rows]
    if not task_rows:
        raise RuntimeError(f"no task rows: {task_map}")
    if not noise_rows:
        raise RuntimeError(f"no noise rows: {noise_manifest}")

    summary_rows: list[dict] = []
    for mode in modes:
        if mode not in {"random", "similar", "dissimilar"}:
            raise RuntimeError(f"unsupported noise mode: {mode}")
        for k in pool_sizes:
            for seed in seeds:
                retrieval_rows: list[dict] = []
                execution_rows: list[dict] = []
                for task in task_rows:
                    primary = task.get("primary_gt_skill") or {}
                    primary_md = str(primary.get("primary_skill_md") or primary.get("skill_md") or "")
                    if not primary_md:
                        continue
                    noise = choose_noise(task_row=task, noise_index=noise_index, mode=mode, count=max(0, k - 1), seed=seed)
                    candidates = [
                        {
                            "skill_name": str(primary.get("skill_name") or primary.get("skill_slug") or task.get("task_name")),
                            "skill_slug": str(primary.get("skill_slug") or "gt"),
                            "skill_md": primary_md,
                            "description": str(primary.get("description") or ""),
                            "role": "gt",
                        }
                    ]
                    for row in noise:
                        candidates.append(
                            {
                                "skill_name": str(row.get("skill_name") or row.get("skill_slug")),
                                "skill_slug": str(row.get("skill_slug") or row.get("skill_name")),
                                "skill_md": str(row.get("skill_md") or ""),
                                "description": str(row.get("description") or ""),
                                "role": "noise",
                            }
                        )
                    if args.shuffle_candidates:
                        rng = random.Random(f"shuffle:{seed}:{mode}:{k}:{task.get('task_name')}")
                        rng.shuffle(candidates)

                    task_name = str(task.get("task_name") or "")
                    retrieval_rows.append(
                        {
                            "benchmark": args.benchmark,
                            "task_name": task_name,
                            "task_description": str(task.get("task_description") or ""),
                            "pool_size": k,
                            "noise_mode": mode,
                            "seed": seed,
                            "gt_skill": candidates[[c.get("role") for c in candidates].index("gt")],
                            "gt_skill_names": [c["skill_name"] for c in candidates if c.get("role") == "gt"],
                            "candidate_skills": candidates,
                        }
                    )
                    execution_rows.append(
                        {
                            "task_name": task_name,
                            "neighbors": [
                                {
                                    "neighbor_task_name": c["skill_name"],
                                    "source_skill_md": c["skill_md"],
                                    "role": c["role"],
                                    "skill_slug": c["skill_slug"],
                                }
                                for c in candidates
                            ],
                        }
                    )

                base = root / "pools" / "candidate_pools" / args.benchmark / mode / f"k{k}" / f"seed-{seed}"
                retrieval_path = base.with_suffix(".jsonl")
                execution_path = root / "execution_manifests" / args.benchmark / mode / f"k{k}" / f"seed-{seed}.json"
                write_jsonl(retrieval_path, retrieval_rows)
                write_json(execution_path, {"rows": execution_rows})
                summary_rows.append(
                    {
                        "benchmark": args.benchmark,
                        "noise_mode": mode,
                        "pool_size": k,
                        "seed": seed,
                        "task_count": len(retrieval_rows),
                        "candidate_pool": str(retrieval_path),
                        "execution_manifest": str(execution_path),
                    }
                )

    summary_path = root / "pools" / "candidate_pools" / args.benchmark / "summary.json"
    write_json(summary_path, {"settings": summary_rows})
    print(f"wrote {len(summary_rows)} candidate-pool settings")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
