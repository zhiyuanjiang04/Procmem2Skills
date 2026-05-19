"""Opus LLM-as-judge GT discovery for terminal-bench tasks.

For each (task, candidate_skill) pair from terminal_bench_topk.jsonl, ask Opus
via Max-plan `claude -p` whether the skill would materially help solve the
task. Yes verdicts become ground-truth skills.

Output: one JSONL row per (task_id, skill_id) with verdict + 1-line reason.
Resume-aware: skips pairs already present in --out.

Usage:
    python -m testsets.skill_selection_eval.judge_tb \\
        --topk testsets/data/terminal_bench_topk.jsonl \\
        --out  testsets/runs/2026-05-04-tb-judge.jsonl \\
        --model claude-opus-4-7 --conc 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

_SYSTEM_PROMPT = (
    "You are a strict relevance judge for an agent skill-retrieval evaluation. "
    "You will be shown one TERMINAL-BENCH TASK and one CANDIDATE SKILL "
    "(name + short description). Decide whether loading this skill would "
    "MATERIALLY help an autonomous shell agent solve the task — i.e., the "
    "skill's documented capability is directly relevant to the task's core "
    "requirement, not merely thematically related.\n\n"
    "Be strict: tangential, generic, or domain-adjacent skills are NO. "
    "Only YES if a competent agent loading this skill would have a clear "
    "advantage over not loading it.\n\n"
    "Respond with EXACTLY one line in this format:\n"
    "VERDICT: <YES|NO> | REASON: <one short sentence>"
)

_RATE_LIMIT_MARKER = "hit your limit"
_TRANSIENT_PATTERNS = (
    "api error: 500", "api error: 502", "api error: 503", "api error: 529",
    "internal server error", "server is temporarily limiting requests",
    "overloaded",
)
_VERDICT_RE = re.compile(r"VERDICT:\s*(YES|NO)\s*\|?\s*REASON:\s*(.*)", re.I | re.S)


class RateLimited(Exception):
    pass


def build_user_prompt(task_description: str, skill: dict) -> str:
    return (
        "TERMINAL-BENCH TASK:\n"
        f"{task_description.strip()}\n\n"
        "CANDIDATE SKILL:\n"
        f"name: {skill['name']}\n"
        f"slug: {skill['slug']}\n"
        f"description: {skill['description']}\n\n"
        "Respond with the single VERDICT/REASON line as instructed."
    )


async def _call_claude(prompt: str, model: str, sem: asyncio.Semaphore,
                       max_retries: int = 4) -> tuple[str, int]:
    for attempt in range(max_retries):
        if attempt > 0:
            await asyncio.sleep(min(60.0, 2.0 * (4 ** (attempt - 1))))
        async with sem:
            t0 = time.time()
            proc = await asyncio.create_subprocess_exec(
                "claude", "-p",
                "--model", model,
                "--system-prompt", _SYSTEM_PROMPT,
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
            latency_ms = int((time.time() - t0) * 1000)
        raw = stdout.decode(errors="replace")
        raw_lower = raw.lower()
        if proc.returncode != 0 and not raw.strip():
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"claude CLI rc={proc.returncode}: "
                    f"{stderr.decode(errors='replace').strip()[:200]}"
                )
            continue
        if _RATE_LIMIT_MARKER in raw_lower:
            raise RateLimited(raw.strip()[:200])
        if any(p in raw_lower for p in _TRANSIENT_PATTERNS):
            if attempt == max_retries - 1:
                raise RuntimeError(f"transient API error: {raw.strip()[:200]}")
            continue
        return raw, latency_ms
    raise RuntimeError("exhausted retries unexpectedly")


def parse_verdict(raw: str) -> tuple[str, str]:
    m = _VERDICT_RE.search(raw)
    if not m:
        return ("unparseable", raw.strip()[:200])
    verdict = m.group(1).strip().lower()
    reason = m.group(2).strip().splitlines()[0][:300]
    return (verdict, reason)


async def _judge_one(task: dict, skill: dict, *, model: str,
                     sem: asyncio.Semaphore) -> dict:
    prompt = build_user_prompt(task["task_description"], skill)
    try:
        raw, latency_ms = await _call_claude(prompt, model, sem)
    except RateLimited:
        raise
    except Exception as exc:
        return {
            "task_id": task["task_id"],
            "skill_id": skill["skill_id"],
            "skill_slug": skill["slug"],
            "skill_name": skill["name"],
            "rank": skill["rank"],
            "verdict": "error",
            "reason": repr(exc)[:200],
            "raw": "",
            "latency_ms": 0,
            "model": model,
        }
    verdict, reason = parse_verdict(raw)
    return {
        "task_id": task["task_id"],
        "skill_id": skill["skill_id"],
        "skill_slug": skill["slug"],
        "skill_name": skill["name"],
        "rank": skill["rank"],
        "verdict": verdict,
        "reason": reason,
        "raw": raw.strip()[:1000],
        "latency_ms": latency_ms,
        "model": model,
    }


def _load_done(out_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("verdict") in {"yes", "no", "unparseable"}:
                done.add((r["task_id"], r["skill_id"]))
    return done


async def main_async(args) -> int:
    topk_path = Path(args.topk)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    with topk_path.open() as f:
        for line in f:
            tasks.append(json.loads(line))
    if args.limit:
        tasks = tasks[: args.limit]

    done = _load_done(out_path)
    pairs: list[tuple[dict, dict]] = []
    for t in tasks:
        for s in t.get("topk", [])[: args.k]:
            if (t["task_id"], s["skill_id"]) in done:
                continue
            pairs.append((t, s))

    print(f"[judge_tb] tasks={len(tasks)} k={args.k} pairs_to_run={len(pairs)} "
          f"already_done={len(done)} out={out_path}", file=sys.stderr)

    if not pairs:
        print("[judge_tb] nothing to do", file=sys.stderr)
        return 0

    sem = asyncio.Semaphore(args.conc)
    out_f = out_path.open("a")
    completed = 0
    rate_limited = False
    try:
        coros = [_judge_one(t, s, model=args.model, sem=sem) for t, s in pairs]
        for fut in asyncio.as_completed(coros):
            try:
                rec = await fut
            except RateLimited as e:
                print(f"[judge_tb] RATE LIMITED — aborting: {e}", file=sys.stderr)
                rate_limited = True
                break
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            completed += 1
            if completed % 25 == 0:
                print(f"[judge_tb] {completed}/{len(pairs)} done", file=sys.stderr)
    finally:
        out_f.close()
    print(f"[judge_tb] wrote {completed} judgements"
          + (" (rate-limited; resume later)" if rate_limited else ""),
          file=sys.stderr)
    return 2 if rate_limited else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--topk", required=True,
                   help="Path to terminal_bench_topk.jsonl")
    p.add_argument("--out", required=True,
                   help="Output JSONL path (resume-aware)")
    p.add_argument("--model", default="claude-opus-4-7")
    p.add_argument("--k", type=int, default=5,
                   help="How many top-K candidates per task to judge")
    p.add_argument("--conc", type=int, default=4)
    p.add_argument("--limit", type=int, default=0,
                   help="Limit number of tasks (0 = all)")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
