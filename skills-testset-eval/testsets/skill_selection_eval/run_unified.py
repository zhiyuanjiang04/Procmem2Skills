"""Unified skill-selection eval runner for SkillsBench + TerminalBench.

Experiment design (per dataset):
  Step 1 — GT skills are always placed in the pool.
  Step 2 — Distractors fill remaining slots; pool sizes = {5, 50, 500}.
  Distractor types: random | hard (emb-near) | easy (emb-far)

Each (task_id, pool_size, noise_mode, seed) trial is independent — conditions
cannot interfere with each other. Resume via --resume re-reads the output file
and skips already-completed trials.

Usage examples:
  python -m skill_selection_eval.run_unified --dataset sb \\
      --tasks testsets/data/skillsbench_tasks.jsonl \\
      --out testsets/runs/sb_sonnet46.jsonl --model claude-sonnet-4-6

  python -m skill_selection_eval.run_unified --dataset tb \\
      --tasks testsets/data/terminal_bench_validated.jsonl \\
      --out testsets/runs/tb_sonnet46.jsonl --model claude-sonnet-4-6
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
from .run import call_cli_async, call_sdk_async, RateLimited

_DEFAULT_POOL_SIZES = [5, 50, 500]
_DEFAULT_NOISE_MODES = ["random", "hard", "easy"]
_DEFAULT_SEEDS = [0]


def _build_sb_pool(task: dict, pool_size: int, noise_mode: str, seed: int, max_gt: int):
    from .pool_builder import build_pool
    return build_pool(
        task_id=task["task_id"],
        task_description=task["task_description"],
        pool_size=pool_size,
        noise_mode=noise_mode,
        max_gt=max_gt,
        seed=seed,
    )


def _build_tb_pool(task: dict, pool_size: int, noise_mode: str, seed: int):
    from .tb_pool_builder_v2 import build_tb_pool_v2
    return build_tb_pool_v2(
        task_id=task["task_id"],
        task_description=task["task_description"],
        gt_skills=task.get("gt_skills") or [],
        pool_size=pool_size,
        noise_mode=noise_mode,
        seed=seed,
    )


async def _run_one(
    *,
    dataset: str,
    task: dict,
    pool_size: int,
    noise_mode: str,
    seed: int,
    max_gt: int,
    model: str,
    backend: str,
    sem: asyncio.Semaphore,
) -> dict:
    if dataset == "sb":
        pool = _build_sb_pool(task, pool_size, noise_mode, seed, max_gt)
    else:
        pool = _build_tb_pool(task, pool_size, noise_mode, seed)

    prompt = build_prompt(task["task_description"], pool.candidates)

    if backend == "cli":
        response = await call_cli_async(prompt, model, sem=sem)
    else:
        response = await call_sdk_async(prompt, model, sem=sem)

    selected = parse_selected_skills(response)
    gt_names = pool.gt_names

    return {
        "dataset": dataset,
        "task_id": task["task_id"],
        "pool_size": pool_size,
        "noise_mode": noise_mode,
        "seed": seed,
        "gt_names": gt_names,
        "gt_positions": pool.gt_positions,
        "n_candidates": len(pool.candidates),
        "model": model,
        "response": response,
        "top1": selected[0] if selected else None,
        "all_selected": selected,
        "hit1": int(bool(selected and selected[0] in gt_names)),
        "recall": int(bool(set(selected) & set(gt_names))),
    }


async def _run_safe(trial: tuple, *, dataset, max_gt, backend, model, sem):
    task, sz, mode, seed = trial
    try:
        return await _run_one(
            dataset=dataset, task=task, pool_size=sz, noise_mode=mode, seed=seed,
            max_gt=max_gt, backend=backend, model=model, sem=sem,
        )
    except RateLimited:
        raise
    except Exception as e:
        return {
            "dataset": dataset,
            "task_id": task.get("task_id"),
            "pool_size": sz,
            "noise_mode": mode,
            "seed": seed,
            "error": f"{type(e).__name__}: {str(e)[:300]}",
        }


async def main_async(args) -> int:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    if args.limit:
        tasks = tasks[: args.limit]

    # Skip trials already written (resume mode for isolation guarantee).
    done: set[tuple] = set()
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" not in r:
                done.add((r["task_id"], r["pool_size"], r["noise_mode"], r["seed"]))

    trials = [
        (task, sz, mode, seed)
        for task in tasks
        for sz in args.pool_sizes
        for mode in args.noise_modes
        for seed in args.seeds
        if (task["task_id"], sz, mode, seed) not in done
    ]

    total_expected = len(tasks) * len(args.pool_sizes) * len(args.noise_modes) * len(args.seeds)
    print(
        f"[{args.dataset.upper()}] tasks={len(tasks)} pool_sizes={args.pool_sizes} "
        f"modes={args.noise_modes} seeds={args.seeds}",
        file=sys.stderr,
    )
    print(
        f"Trials to run: {len(trials)} / {total_expected} (resumed {len(done)}); "
        f"concurrency={args.concurrency} model={args.model}",
        file=sys.stderr,
    )

    sem = asyncio.Semaphore(args.concurrency)
    fout = args.out.open("a" if args.resume else "w")
    completed = 0
    errors = 0
    t_start = time.time()

    try:
        coros = [
            _run_safe(t, dataset=args.dataset, max_gt=args.max_gt,
                      backend=args.backend, model=args.model, sem=sem)
            for t in trials
        ]
        for fut in asyncio.as_completed(coros):
            try:
                result = await fut
            except RateLimited as e:
                print(f"!!! RATE LIMITED: {e}. Rerun with --resume.", file=sys.stderr)
                break

            fout.write(json.dumps(result) + "\n")
            fout.flush()
            completed += 1
            if "error" in result:
                errors += 1

            elapsed = time.time() - t_start
            rate = completed / max(elapsed, 1)
            eta = (len(trials) - completed) / max(rate, 0.001)
            tag = result.get("top1") if "error" not in result else f"ERR:{result.get('error','?')[:40]}"
            hit = result.get("hit1", "?")
            print(
                f"[{completed}/{len(trials)} {elapsed:.0f}s eta={eta:.0f}s] "
                f"{result.get('task_id')} sz={result.get('pool_size')} "
                f"{result.get('noise_mode')}/s{result.get('seed')} -> {tag} hit={hit}",
                file=sys.stderr,
            )
    finally:
        fout.close()

    print(f"Done. Completed={completed} errors={errors}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Unified SB/TB skill-selection eval with GT-in-pool design."
    )
    p.add_argument("--dataset", choices=["sb", "tb"], required=True,
                   help="sb=SkillsBench, tb=TerminalBench (validated)")
    p.add_argument("--tasks", type=Path, required=True,
                   help="skillsbench_tasks.jsonl or terminal_bench_validated.jsonl")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL path (appended if --resume)")
    p.add_argument("--model", default="claude-sonnet-4-6",
                   help="Claude model ID (default: claude-sonnet-4-6)")
    p.add_argument("--pool-sizes", type=int, nargs="+", default=_DEFAULT_POOL_SIZES,
                   help="Pool size(s) to sweep (default: 5 50 500)")
    p.add_argument("--noise-modes", nargs="+", default=_DEFAULT_NOISE_MODES,
                   help="Distractor strategies (default: random hard easy)")
    p.add_argument("--seeds", type=int, nargs="+", default=_DEFAULT_SEEDS,
                   help="Random seeds (default: 0)")
    p.add_argument("--max-gt", type=int, default=1,
                   help="Max GT skills per pool (SB only; default: 1)")
    p.add_argument("--backend", choices=["cli", "sdk"], default="cli",
                   help="API backend (default: cli)")
    p.add_argument("--concurrency", type=int, default=6,
                   help="Max concurrent model calls (default: 6)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of tasks (for smoke tests)")
    p.add_argument("--resume", action="store_true",
                   help="Append to existing output, skip completed trials")
    args = p.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
