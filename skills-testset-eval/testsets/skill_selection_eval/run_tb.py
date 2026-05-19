"""Run the harbor-mimic skill-selection harness on terminal-bench tasks.

Each trial: feed (task_description + candidate pool) to claude-sonnet via
`claude -p`; parse out the skill name(s) the model picks. Pool composition
is controlled via tb_pool_builder.

Outputs JSONL with per-trial:
    task_id, pool_size, noise_mode, seed, topk_names, topk_positions,
    proxy_gt_name (top-1 retrieval), response, top1, all_selected.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from .parser import parse_selected_skills
from .prompt import build_prompt
from .run import call_cli_async, RateLimited
from .tb_pool_builder import build_tb_pool


async def _run_one(*, task: dict, pool_size: int, noise_mode: str, seed: int,
                   topk_in_pool: int, model: str, sem: asyncio.Semaphore) -> dict:
    pool = build_tb_pool(
        task_id=task["task_id"],
        task_description=task["task_description"],
        pool_size=pool_size,
        noise_mode=noise_mode,
        topk_in_pool=topk_in_pool,
        seed=seed,
    )
    if not pool.candidates:
        return {"task_id": task["task_id"], "pool_size": pool_size, "noise_mode": noise_mode,
                "seed": seed, "error": "empty_pool"}
    prompt = build_prompt(task["task_description"], pool.candidates)
    response = await call_cli_async(prompt, model, sem=sem)
    selected = parse_selected_skills(response)
    validated_gt_slugs = [s["slug"] for s in task.get("gt_skills") or []]
    return {
        "task_id": task["task_id"],
        "pool_size": pool_size,
        "noise_mode": noise_mode,
        "seed": seed,
        "topk_in_pool": topk_in_pool,
        "topk_names": pool.topk_names,
        "topk_positions": pool.topk_positions,
        "proxy_gt_name": pool.proxy_gt_name,
        "validated_gt_slugs": validated_gt_slugs,
        "n_candidates": len(pool.candidates),
        "model": model,
        "response": response,
        "top1": selected[0] if selected else None,
        "all_selected": selected,
    }


async def _run_safe(trial, *, topk_in_pool, model, sem):
    task, sz, mode, seed = trial
    try:
        return await _run_one(task=task, pool_size=sz, noise_mode=mode, seed=seed,
                              topk_in_pool=topk_in_pool, model=model, sem=sem)
    except RateLimited:
        raise
    except Exception as e:
        return {"task_id": task.get("task_id"), "pool_size": sz,
                "noise_mode": mode, "seed": seed,
                "error": f"{type(e).__name__}: {str(e)[:300]}"}


async def main_async(args) -> int:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]
    if args.limit:
        tasks = tasks[: args.limit]

    done: set[tuple] = set()
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add((r["task_id"], r["pool_size"], r["noise_mode"], r["seed"]))

    trials = [
        (task, sz, mode, seed)
        for task in tasks
        for sz in args.pool_sizes
        for mode in args.noise_modes
        for seed in args.seeds
        if (task["task_id"], sz, mode, seed) not in done
    ]
    print(f"TB trials to run: {len(trials)} (resumed {len(done)}); concurrency={args.concurrency}", file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency)
    fout = args.out.open("a" if args.resume else "w")
    completed = 0
    t_start = time.time()
    try:
        coros = [
            _run_safe(t, topk_in_pool=args.topk_in_pool, model=args.model, sem=sem)
            for t in trials
        ]
        for fut in asyncio.as_completed(coros):
            try:
                result = await fut
            except RateLimited as e:
                print(f"!!! RATE LIMITED: {e}. Aborting; rerun with --resume.", file=sys.stderr)
                break
            fout.write(json.dumps(result) + "\n"); fout.flush()
            completed += 1
            tag = result.get("top1") if "error" not in result else "ERR"
            elapsed = time.time() - t_start
            rate = completed / max(elapsed, 1)
            eta = (len(trials) - completed) / max(rate, 0.001)
            print(f"[{completed}/{len(trials)} {elapsed:.0f}s eta={eta:.0f}s] "
                  f"{result.get('task_id')} sz={result.get('pool_size')} "
                  f"{result.get('noise_mode')}/s{result.get('seed')} -> {tag}",
                  file=sys.stderr)
    finally:
        fout.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, required=True,
                   help="terminal_bench_tasks.jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pool-sizes", type=int, nargs="+", default=[5, 50, 500])
    p.add_argument("--noise-modes", nargs="+",
                   default=["topk_only", "random", "hard", "easy", "none"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--topk-in-pool", type=int, default=5,
                   help="how many top-K retrieval entries to keep in pool")
    p.add_argument("--model", default="claude-sonnet-4-5")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
