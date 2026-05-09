"""Position-bias sanity check: fix task + GT, vary GT position in pool.

For each (task, GT) pair, build a pool of fixed size (default 500) by
sampling random ClawHub noise and inserting the GT skill at one of three
controlled bins {early, middle, late}. Run model and record Hit@1.

If Hit@1 differs across bins, the model has positional attention bias —
a confound for the main grid sweep at large pool sizes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

from .parser import parse_selected_skills
from .pool_builder import _gt_records_for_task, _gt_to_candidate, _sample_random_noise
from .prompt import SkillCandidate, build_prompt
from .run import call_cli_async, RateLimited

POSITIONS = {
    "early":  lambda n, rng: rng.randint(0, max(0, n // 10 - 1)),
    "middle": lambda n, rng: rng.randint(max(0, n // 2 - n // 20), min(n - 1, n // 2 + n // 20)),
    "late":   lambda n, rng: rng.randint(max(0, n - n // 10), n - 1),
}


def build_positioned_pool(*, task_id: str, pool_size: int, position: str, seed: int):
    from .pool_builder import _stable_seed
    rng = random.Random(_stable_seed(task_id, pool_size, position, seed))
    gt_records = _gt_records_for_task(task_id)
    if not gt_records:
        return None
    gt_rec = rng.choice(gt_records)
    gt_cand, gt_name = _gt_to_candidate(gt_rec)
    exclude = {gt_rec["clawhub_alias_id"]} if gt_rec.get("clawhub_alias_id") else set()
    noise = _sample_random_noise(pool_size - 1, exclude, rng)
    if len(noise) < pool_size - 1:
        return None
    pool: list[SkillCandidate] = list(noise)
    target_idx = POSITIONS[position](pool_size, rng)
    pool.insert(target_idx, gt_cand)
    return pool, target_idx, gt_name


async def _run_one(task, position, pool_size, seed, model, sem):
    built = build_positioned_pool(
        task_id=task["task_id"], pool_size=pool_size, position=position, seed=seed,
    )
    if built is None:
        return {"task_id": task["task_id"], "position": position, "error": "no_gt_or_noise"}
    pool, idx, gt_name = built
    prompt = build_prompt(task["task_description"], pool)
    try:
        response = await call_cli_async(prompt, model, sem=sem)
    except RateLimited:
        raise
    except Exception as e:
        return {
            "task_id": task["task_id"], "position": position,
            "gt_name": gt_name, "gt_index": idx,
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }
    selected = parse_selected_skills(response)
    top1 = selected[0] if selected else None
    return {
        "task_id": task["task_id"], "position": position,
        "pool_size": pool_size,
        "gt_name": gt_name, "gt_index": idx,
        "response": response, "top1": top1,
        "all_selected": selected, "hit": int(top1 == gt_name),
    }


async def main_async(args) -> int:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()][: args.limit]
    sem = asyncio.Semaphore(args.concurrency)
    coros = [
        _run_one(task, pos, args.pool_size, args.seed, args.model, sem)
        for task in tasks
        for pos in args.positions
    ]
    fout = args.out.open("w")
    completed = 0
    t_start = time.time()
    try:
        for fut in asyncio.as_completed(coros):
            try:
                out = await fut
            except RateLimited as e:
                print(f"!!! RATE LIMITED: {e}", file=sys.stderr)
                break
            fout.write(json.dumps(out) + "\n"); fout.flush()
            completed += 1
            elapsed = time.time() - t_start
            tag = out.get("top1") if "error" not in out else "ERR"
            print(f"[{completed}/{len(coros)} {elapsed:.0f}s] {out.get('task_id')} {out.get('position')}@{out.get('gt_index')} -> top1={tag} hit={out.get('hit')}", file=sys.stderr)
    finally:
        fout.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pool-size", type=int, default=500)
    p.add_argument("--positions", nargs="+", default=["early", "middle", "late"])
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="claude-sonnet-4-5")
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
