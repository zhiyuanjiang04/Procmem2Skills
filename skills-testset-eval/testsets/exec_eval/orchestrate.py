"""Execution-pass-rate orchestrator (Docker-required, dry-run safe on Anvil).

For each (task_id, n_noise, seed) trial:
  1. Build the candidate pool (reuses skill_selection_eval pool builders).
  2. Materialize a /tmp/eval/<trial>/skills/ dir (real GT + stubbed noise).
  3. Call harbor run -p <task_dir> -a terminus-2-skills -m sonnet-4-6
        --skill-dir <skills/> --skill-format xml --temperature 0.0
  4. Parse harbor output → exit code → pass/fail.
  5. Append JSONL row.

Modes:
  --dry-run     skip steps 3-5, just verify pool/skill-dir construction works.
  --run-cmd     prints the exact harbor command per trial (no execution).
  (default)     real execution — requires Docker.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add testsets/ to path so skill_selection_eval modules import.
THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from skill_selection_eval.pool_builder import build_pool          # noqa: E402
from skill_selection_eval.tb_pool_builder_v2 import build_tb_pool_v2  # noqa: E402

from testsets.exec_eval.pool_to_skill_dir import build_skill_dir  # noqa: E402

REPO_ROOT = THIS.parent.parent.parent
SB_TASKS_ROOT = REPO_ROOT.parent / "skillsbench_repo" / "tasks"
TB_TASKS_ROOT = REPO_ROOT.parent / "terminal-bench" / "tasks"  # may not exist; TB needs adapter


def _build_pool_for_trial(dataset: str, task: dict, n_noise: int, seed: int):
    if dataset == "sb":
        n_gt = len(task.get("gt_skills") or [])
        return build_pool(
            task_id=task["task_id"],
            task_description=task["task_description"],
            pool_size=n_gt + n_noise,
            noise_mode="random",
            max_gt=max(n_gt, 1),
            seed=seed,
        )
    else:
        gts = task.get("gt_skills") or []
        return build_tb_pool_v2(
            task_id=task["task_id"],
            task_description=task["task_description"],
            gt_skills=gts,
            pool_size=len(gts) + n_noise,
            noise_mode="random",
            seed=seed,
        )


def _resolve_task_dir(dataset: str, task_id: str) -> Path:
    if dataset == "sb":
        return SB_TASKS_ROOT / task_id
    return TB_TASKS_ROOT / task_id


async def _run_trial(
    *, dataset, task, n_noise, seed, model, sem, corpus_path,
    work_root: Path, dry_run: bool, run_cmd_only: bool,
) -> dict:
    task_id = task["task_id"]
    trial_id = f"{dataset}__{task_id}__nn{n_noise}__s{seed}"
    trial_dir = work_root / trial_id
    skills_dir = trial_dir / "skills"

    pool = _build_pool_for_trial(dataset, task, n_noise, seed)
    gt_slugs = pool.gt_names
    noise_slugs = [c.name for c in pool.candidates if c.name not in set(gt_slugs)]

    task_dir = _resolve_task_dir(dataset, task_id)
    if not task_dir.exists() and dataset == "sb":
        return {"trial_id": trial_id, "error": f"task_dir not found: {task_dir}"}

    # Step 2 of pipeline — works without Docker; useful for verification.
    try:
        if dataset == "sb":
            build_skill_dir(
                task_dir=task_dir,
                gt_slugs=gt_slugs,
                noise_slugs=noise_slugs,
                out_dir=skills_dir,
                corpus_path=corpus_path,
            )
    except Exception as e:
        return {"trial_id": trial_id, "stage": "build_skill_dir",
                "error": f"{type(e).__name__}: {e}"}

    cmd = [
        "harbor", "run",
        "-p", str(task_dir),
        "-a", "terminus-2-skills",
        "-m", model,
        "--skill-dir", str(skills_dir),
        "--skill-format", "xml",
        "--temperature", "0.0",
    ]

    if run_cmd_only:
        return {"trial_id": trial_id, "cmd": " ".join(cmd)}

    if dry_run:
        return {
            "trial_id": trial_id, "dataset": dataset, "task_id": task_id,
            "n_noise": n_noise, "seed": seed, "pool_size": len(pool.candidates),
            "gt_slugs": gt_slugs, "n_noise_built": len(noise_slugs),
            "skills_dir": str(skills_dir), "task_dir": str(task_dir),
            "dry_run": True,
        }

    # Real execution path
    t0 = time.time()
    async with sem:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
    wall = time.time() - t0
    rc = proc.returncode

    return {
        "trial_id": trial_id, "dataset": dataset, "task_id": task_id,
        "n_noise": n_noise, "seed": seed, "pool_size": len(pool.candidates),
        "gt_slugs": gt_slugs, "model": model,
        "exit_code": rc, "pass": rc == 0,
        "wall_clock_s": round(wall, 1),
        "stdout_tail": stdout.decode(errors="replace")[-2000:],
        "stderr_tail": stderr.decode(errors="replace")[-2000:],
    }


async def main_async(args):
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]
    tasks = [t for t in tasks if t.get("gt_skills")]
    if args.limit:
        tasks = tasks[: args.limit]

    done = set()
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    done.add((r.get("task_id"), r.get("n_noise"), r.get("seed")))

    trials = [(t, nn, sd) for t in tasks for nn in args.n_noise for sd in args.seeds
              if (t["task_id"], nn, sd) not in done]
    print(f"[{args.dataset.upper()}] tasks={len(tasks)} n_noise={args.n_noise} "
          f"trials={len(trials)} done={len(done)} dry={args.dry_run}",
          file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency)
    fout = args.out.open("a" if args.resume else "w")
    completed = 0

    try:
        coros = [_run_trial(
            dataset=args.dataset, task=t, n_noise=nn, seed=sd, model=args.model,
            sem=sem, corpus_path=args.corpus,
            work_root=args.work_root, dry_run=args.dry_run,
            run_cmd_only=args.run_cmd_only,
        ) for (t, nn, sd) in trials]
        for fut in asyncio.as_completed(coros):
            r = await fut
            fout.write(json.dumps(r) + "\n")
            fout.flush()
            completed += 1
            tag = "PASS" if r.get("pass") else ("DRY" if r.get("dry_run") else
                  ("CMD" if "cmd" in r else f"FAIL/ERR({r.get('exit_code','?')})"))
            print(f"[{completed}/{len(trials)}] {r.get('trial_id', r)} -> {tag}",
                  file=sys.stderr)
    finally:
        fout.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["sb", "tb"], required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work-root", type=Path, default=Path("/tmp/exec_eval"))
    p.add_argument("--corpus", type=Path,
                   default=Path(os.environ.get("SKILL_CORPUS_PATH",
                                               "data/processed/skill_corpus.jsonl")))
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--n-noise", type=int, nargs="+", default=[0, 1, 5, 10, 20, 50])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="build skill dirs but do NOT call harbor (use on Anvil w/o Docker)")
    p.add_argument("--run-cmd-only", action="store_true",
                   help="print the harbor command per trial; build nothing")
    args = p.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
