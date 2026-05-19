"""Run the harbor-mimic skill-selection eval over a JSONL of tasks (async).

Concurrency model adapted from skills_retrieval/cli_driver.py:
  - asyncio.Semaphore caps in-flight `claude -p` subprocesses
  - prompt sent via stdin (avoids 128KB argv limit at pool_size=500)
  - retry on transient API errors (5xx/overloaded); abort on Max-plan quota
  - resume-aware: skips trials already present in --out

Output: one JSONL row per (task, pool_size, noise_mode, seed) trial.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

from .parser import parse_selected_skills
from .pool_builder import build_pool
from .prompt import build_prompt

_CLI_SYSTEM_PROMPT = (
    "You are a skill-selection assistant. Read the user's task description "
    "and the <available_skills> block. Choose the SINGLE skill that best fits "
    "the task and emit it using the EXACT format below — no other text:\n"
    '<tool_call name="skill"><name>SKILL-NAME</name></tool_call>\n'
    "If none of the skills is relevant, output:\n"
    '<tool_call name="skill"><name>NONE</name></tool_call>'
)

_RATE_LIMIT_MARKER = "hit your limit"
_TRANSIENT_PATTERNS = (
    "api error: 500", "api error: 502", "api error: 503", "api error: 529",
    "internal server error", "server is temporarily limiting requests",
    "overloaded",
)


class RateLimited(Exception):
    pass


async def call_cli_async(prompt: str, model: str, *, sem: asyncio.Semaphore,
                         max_retries: int = 4) -> str:
    """Async claude -p with retry + rate-limit detection. Raises RateLimited
    when Max-plan quota is hit (caller should abort, not retry)."""
    for attempt in range(max_retries):
        if attempt > 0:
            await asyncio.sleep(min(60.0, 2.0 * (4 ** (attempt - 1))))
        async with sem:
            proc = await asyncio.create_subprocess_exec(
                "claude", "-p",
                "--model", model,
                "--system-prompt", _CLI_SYSTEM_PROMPT,
                "--output-format", "text",
                "--tools", "",
                "--no-session-persistence",
                "--permission-mode", "bypassPermissions",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert proc.stdin is not None
            try:
                proc.stdin.write(prompt.encode())
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            try:
                proc.stdin.close()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            stdout, stderr = await proc.communicate()
        raw = stdout.decode(errors="replace")
        raw_lower = raw.lower()
        if proc.returncode != 0 and not raw.strip():
            last = f"rc={proc.returncode}: {stderr.decode(errors='replace').strip()[:200]}"
            if attempt == max_retries - 1:
                raise RuntimeError(f"claude CLI failed: {last}")
            continue
        if _RATE_LIMIT_MARKER in raw_lower:
            raise RateLimited(raw.strip()[:200])
        if any(p in raw_lower for p in _TRANSIENT_PATTERNS):
            if attempt == max_retries - 1:
                raise RuntimeError(f"transient API error after retries: {raw.strip()[:200]}")
            continue
        return raw
    raise RuntimeError("call_cli_async exhausted retries unexpectedly")


async def call_sdk_async(prompt: str, model: str, *, sem: asyncio.Semaphore,
                          max_tokens: int = 256) -> str:
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    async with sem:
        msg = await client.messages.create(
            model=model, max_tokens=max_tokens,
            system=_CLI_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


async def run_one(*, task: dict, pool_size: int, noise_mode: str, seed: int,
                  max_gt: int, backend: str, model: str,
                  sem: asyncio.Semaphore) -> dict:
    pool = build_pool(
        task_id=task["task_id"],
        task_description=task["task_description"],
        pool_size=pool_size,
        noise_mode=noise_mode,
        max_gt=max_gt,
        seed=seed,
    )
    prompt = build_prompt(task["task_description"], pool.candidates)
    if backend == "cli":
        response = await call_cli_async(prompt, model, sem=sem)
    else:
        response = await call_sdk_async(prompt, model, sem=sem)
    selected = parse_selected_skills(response)
    return {
        "task_id": task["task_id"],
        "pool_size": pool_size,
        "noise_mode": noise_mode,
        "seed": seed,
        "max_gt": max_gt,
        "gt_names": pool.gt_names,
        "gt_positions": pool.gt_positions,
        "n_candidates": len(pool.candidates),
        "model": model,
        "response": response,
        "top1": selected[0] if selected else None,
        "all_selected": selected,
    }


async def _run_safe(trial, *, max_gt, backend, model, sem):
    task, sz, mode_str, seed = trial
    try:
        return await run_one(
            task=task, pool_size=sz, noise_mode=mode_str, seed=seed,
            max_gt=max_gt, backend=backend, model=model, sem=sem,
        )
    except RateLimited:
        raise
    except Exception as e:
        return {
            "task_id": task.get("task_id"),
            "pool_size": sz, "noise_mode": mode_str, "seed": seed,
            "error": f"{type(e).__name__}: {str(e)[:300]}",
        }


async def main_async(args) -> int:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
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
    print(f"Trials to run: {len(trials)} (resumed {len(done)}); concurrency={args.concurrency}", file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency)
    fout = args.out.open("a" if args.resume else "w")
    completed = 0
    t_start = time.time()
    try:
        coros = [
            _run_safe(t, max_gt=args.max_gt, backend=args.backend, model=args.model, sem=sem)
            for t in trials
        ]
        for fut in asyncio.as_completed(coros):
            try:
                result = await fut
            except RateLimited as e:
                print(f"!!! RATE LIMITED: {e}. Aborting; rerun with --resume after window resets.", file=sys.stderr)
                break
            fout.write(json.dumps(result) + "\n")
            fout.flush()
            completed += 1
            tag = result.get("top1") if "error" not in result else "ERR"
            elapsed = time.time() - t_start
            rate = completed / max(elapsed, 1)
            eta = (len(trials) - completed) / max(rate, 0.001)
            print(
                f"[{completed}/{len(trials)} {elapsed:.0f}s eta={eta:.0f}s] "
                f"{result.get('task_id')} sz={result.get('pool_size')} "
                f"{result.get('noise_mode')}/s{result.get('seed')} -> {tag}",
                file=sys.stderr,
            )
    finally:
        fout.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pool-sizes", type=int, nargs="+", default=[5, 50, 500])
    p.add_argument("--noise-modes", nargs="+", default=["random", "hard", "easy", "none"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--max-gt", type=int, default=1)
    p.add_argument("--backend", choices=["cli", "sdk"], default="cli")
    p.add_argument("--model", default="claude-sonnet-4-5")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
