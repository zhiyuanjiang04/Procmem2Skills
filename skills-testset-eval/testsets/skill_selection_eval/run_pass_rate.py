"""Pass-rate-style eval over the (n_noise, dataset) grid.

Spec (from user 2026-05-13):
  - Always place ALL ground-truth skills in the pool (Step 1).
  - Distractors (Step 2) come from ClawHub corpus, mode = random.
  - Pool size per trial = |GT_task| + n_noise.
  - n_noise sweep: {0, 1, 5, 10, 20, 50}.
  - Datasets: SB (88) + TB (62 validated).

NOTE: This is the **no-Docker proxy** for pass-rate. The harness presents the
XML pool to Sonnet 4.6, parses the first skill it requests, and records
whether the picked skill is in the GT set. Real container execution would
load that skill's SKILL.md and let the agent run shell commands; with the
correct skill loaded, the agent has the strongest available guidance to pass
the task. So this hit-rate is the upper-bound of execution pass-rate
(execution can still fail downstream even with the right skill loaded; it
cannot succeed without it for these GT-aligned tasks).

Each trial is independent: (task_id, n_noise, seed) keys are unique, --resume
skips already-completed rows, JSONL output is appended.
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

_DEFAULT_N_NOISE = [0, 1, 5, 10, 20, 50]
_DEFAULT_SEEDS = [0]


def _build_sb_pool(task: dict, n_noise: int, seed: int):
    """SB pool: ALL GT skills + n_noise random distractors."""
    from .pool_builder import build_pool
    n_gt = len(task.get("gt_skills") or [])
    pool_size = n_gt + n_noise
    return build_pool(
        task_id=task["task_id"],
        task_description=task["task_description"],
        pool_size=pool_size,
        noise_mode="random",
        max_gt=max(n_gt, 1),  # keep all GT, never subsample
        seed=seed,
    )


def _build_tb_pool(task: dict, n_noise: int, seed: int):
    """TB pool: ALL validated GT skills + n_noise random distractors."""
    from .tb_pool_builder_v2 import build_tb_pool_v2
    gt_skills = task.get("gt_skills") or []
    pool_size = len(gt_skills) + n_noise
    return build_tb_pool_v2(
        task_id=task["task_id"],
        task_description=task["task_description"],
        gt_skills=gt_skills,
        pool_size=pool_size,
        noise_mode="random",
        seed=seed,
    )


async def _run_one(
    *,
    dataset: str,
    task: dict,
    n_noise: int,
    seed: int,
    model: str,
    backend: str,
    sem: asyncio.Semaphore,
) -> dict:
    if dataset == "sb":
        pool = _build_sb_pool(task, n_noise, seed)
    else:
        pool = _build_tb_pool(task, n_noise, seed)

    prompt = build_prompt(task["task_description"], pool.candidates)

    if backend == "cli":
        response = await call_cli_async(prompt, model, sem=sem)
    else:
        response = await call_sdk_async(prompt, model, sem=sem)

    selected = parse_selected_skills(response)
    gt_names = pool.gt_names
    gt_set = set(gt_names)
    pick_set = set(selected)

    hit1 = int(bool(selected and selected[0] in gt_set))
    # multi-GT recall = |picked ∩ GT| / |GT|
    recall = (len(gt_set & pick_set) / max(len(gt_set), 1)) if gt_set else 0.0
    refusal = int(bool(selected) is False or (selected and selected[0].upper() == "NONE"))

    return {
        "dataset": dataset,
        "task_id": task["task_id"],
        "n_noise": n_noise,
        "pool_size": len(pool.candidates),
        "n_gt": len(gt_names),
        "seed": seed,
        "gt_names": gt_names,
        "gt_positions": pool.gt_positions,
        "model": model,
        "response": response,
        "top1": selected[0] if selected else None,
        "all_selected": selected,
        "hit1": hit1,
        "recall": recall,
        "refusal": refusal,
    }


async def _run_safe(trial, *, dataset, backend, model, sem):
    task, n_noise, seed = trial
    try:
        return await _run_one(
            dataset=dataset, task=task, n_noise=n_noise, seed=seed,
            backend=backend, model=model, sem=sem,
        )
    except RateLimited:
        raise
    except Exception as e:
        return {
            "dataset": dataset,
            "task_id": task.get("task_id"),
            "n_noise": n_noise,
            "seed": seed,
            "error": f"{type(e).__name__}: {str(e)[:300]}",
        }


async def main_async(args) -> int:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]
    # Drop any task with empty GT (1 SB task has gt_skills=[]).
    tasks = [t for t in tasks if t.get("gt_skills")]
    if args.limit:
        tasks = tasks[: args.limit]

    done: set[tuple] = set()
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" not in r:
                done.add((r["task_id"], r["n_noise"], r["seed"]))

    trials = [
        (task, nn, seed)
        for task in tasks
        for nn in args.n_noise
        for seed in args.seeds
        if (task["task_id"], nn, seed) not in done
    ]

    total_expected = len(tasks) * len(args.n_noise) * len(args.seeds)
    print(
        f"[{args.dataset.upper()}] tasks={len(tasks)} n_noise={args.n_noise} seeds={args.seeds}",
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
        coros = [_run_safe(t, dataset=args.dataset, backend=args.backend, model=args.model, sem=sem)
                 for t in trials]
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
                f"{result.get('task_id')} nn={result.get('n_noise')} "
                f"s{result.get('seed')} -> {tag} hit={hit}",
                file=sys.stderr,
            )
    finally:
        fout.close()

    print(f"Done. Completed={completed} errors={errors}", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Sonnet 4.6 pass-rate-proxy eval over n_noise grid.")
    p.add_argument("--dataset", choices=["sb", "tb"], required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--n-noise", type=int, nargs="+", default=_DEFAULT_N_NOISE,
                   help="Noise-skill counts to sweep (default: 0 1 5 10 20 50)")
    p.add_argument("--seeds", type=int, nargs="+", default=_DEFAULT_SEEDS)
    p.add_argument("--backend", choices=["cli", "sdk"], default="cli")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
