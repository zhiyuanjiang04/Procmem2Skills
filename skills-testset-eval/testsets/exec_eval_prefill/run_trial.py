"""Single-trial driver for prefill-context execution eval (SB on Anvil host).

Per trial:
  1. Build pool → system prompt with all SKILL.md content concatenated.
  2. Set up isolated workspace at /tmp/exec_prefill/<trial>/work/
       - copy task environment files (except Dockerfile + skills/)
       - resolve task instruction (instruction.md → user prompt)
  3. Invoke `claude -p` with --system-prompt + Bash tool access + cwd=work/
       - Patch task instruction: rewrite /root/ paths → workspace paths
  4. After agent exits, run the task's tests/test_outputs.py against the
     workspace (via WORKDIR env var, since SB tests hardcode /root/).
  5. Emit JSONL row with pass/fail + metadata.

LIMITATIONS (host execution, no Docker):
  - Skip tasks whose Dockerfile needs non-trivial setup (gcc, nodejs, browsers).
  - SB tests reference /root/ — we patch via a `_workdir_aware` test wrapper.
  - pytest version is whatever marl conda env ships (currently 9.x).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from exec_eval_prefill.pool_builder_v3 import build_pool, PoolSpec  # noqa: E402
from exec_eval_prefill.prompt_assembler import build_system_prompt  # noqa: E402


REPO_ROOT = THIS.parent.parent.parent
SB_TASKS_ROOT = REPO_ROOT.parent / "skillsbench_repo" / "tasks"

# Tasks whose Dockerfile is too heavy to host-execute (filter at orchestrate step).
_HEAVY_DOCKERFILE_MARKERS = re.compile(
    r"(playwright|nodejs|npm|texlive|poppler|gcc|build-essential|cuda|chromium|docker)",
    re.IGNORECASE,
)


def is_lightweight(task_dir: Path) -> bool:
    df = task_dir / "environment" / "Dockerfile"
    if not df.exists():
        return False
    return not _HEAVY_DOCKERFILE_MARKERS.search(df.read_text())


# Container top-level dirs SB/TB tasks commonly reference. We rewrite these to
# subdirs of the workspace (or workspace root for /root,/app,/home).
_PATH_ROOTS = {
    "/root": "",
    "/app": "",
    "/home": "",
    "/data": "/data",
    "/output": "/output",
    "/logs": "/logs",
    "/tests": "/tests",
    "/workspace": "",
}


def _mirror_dockerfile_copies(task_dir: Path, work: Path) -> None:
    """Parse `COPY src dst` lines and mirror the file layout into the workspace.

    Reflects host fixture → container path mapping. Skips skills/ (we manage
    those via the prefill prompt, not the filesystem).
    """
    import re as _re
    df_path = task_dir / "environment" / "Dockerfile"
    if not df_path.exists():
        return
    env_dir = task_dir / "environment"
    for line in df_path.read_text().splitlines():
        m = _re.match(r"^\s*COPY\s+(?:--[^\s]+\s+)?(.+?)\s+(\S+)\s*$", line)
        if not m:
            continue
        src_pat, dst = m.group(1).strip(), m.group(2).strip()
        if "skills" in src_pat:
            continue
        if dst.startswith("/"):
            # Map absolute container path → workspace subdir.
            dst_local = work / dst.lstrip("/").rstrip("/")
        else:
            dst_local = work / dst
        dst_local.parent.mkdir(parents=True, exist_ok=True)
        # src may be a glob (e.g. *.cif), a file, or a dir.
        if "*" in src_pat or "?" in src_pat:
            from glob import glob
            matches = glob(str(env_dir / src_pat))
            dst_local.mkdir(parents=True, exist_ok=True)
            for m_path in matches:
                tgt = dst_local / Path(m_path).name
                if Path(m_path).is_dir():
                    if not tgt.exists():
                        shutil.copytree(m_path, tgt)
                else:
                    shutil.copy2(m_path, tgt)
        else:
            src_path = env_dir / src_pat
            if not src_path.exists():
                continue
            if src_path.is_dir():
                if not dst_local.exists():
                    shutil.copytree(src_path, dst_local)
            else:
                shutil.copy2(src_path, dst_local)


def setup_workspace(task_dir: Path, work: Path) -> None:
    """Copy task fixtures into workspace, mirroring Dockerfile COPY layout.

    Two-pass:
      1. Generic: copy environment/ minus skills/+Dockerfile to workspace root
         (for /root/-style tasks).
      2. Dockerfile-aware: mirror COPY src → dst mappings to the workspace
         (for tasks that use /data/, /output/, /app/ etc.).
      3. Also copy tests/ to workspace/tests/ for tasks whose tests are
         referenced as /tests/...
    """
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    src_env = task_dir / "environment"
    if src_env.exists():
        for child in src_env.iterdir():
            if child.name in ("Dockerfile", "skills"):
                continue
            if child.is_dir():
                shutil.copytree(child, work / child.name)
            else:
                shutil.copy2(child, work / child.name)
    _mirror_dockerfile_copies(task_dir, work)
    # Mirror tests/ into workspace/tests/ for /tests/-referencing tasks.
    src_tests = task_dir / "tests"
    if src_tests.exists():
        dst_tests = work / "tests"
        if not dst_tests.exists():
            shutil.copytree(src_tests, dst_tests)
    # Pre-create commonly-written-to dirs.
    for sub in ("output", "logs", "logs/verifier", "logs/verifier/scores"):
        (work / sub).mkdir(parents=True, exist_ok=True)


def _reconcile_output_paths(work: Path) -> None:
    """After agent exits, copy outputs from common wrong locations to expected locations.

    The model may write to /root/, cwd-relative output/, or workspace root.
    Tests expect files at workspace-root level. This reconciles mismatches.
    """
    # Collect all files the agent may have written in subdirs
    candidates = []
    for sub in ("output", "root", "app", "app/output"):
        d = work / sub
        if d.is_dir():
            for f in d.rglob("*"):
                if f.is_file():
                    candidates.append((f, f.relative_to(d)))

    # Copy to workspace root if not already there
    for src_file, rel in candidates:
        dst = work / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst)


def _rewrite_paths(text: str, work_str: str) -> str:
    out = text
    for src, dst_suffix in _PATH_ROOTS.items():
        dst = (work_str + dst_suffix).rstrip("/")
        # Quoted absolute paths
        out = out.replace(f'"{src}/', f'"{dst}/')
        out = out.replace(f"'{src}/", f"'{dst}/")
        out = out.replace(f'"{src}"', f'"{dst}"')
        out = out.replace(f"'{src}'", f"'{dst}'")
        # f-string-style and Path("/foo/...") style
        out = out.replace(f"({src}/", f"({dst}/")
    return out


def patch_instruction(instr: str, work_str: str) -> str:
    return _rewrite_paths(instr, work_str)


def patch_tests(task_dir: Path, work: Path) -> Path:
    """Copy tests/ into work/.tests/ and rewrite container paths → workspace."""
    src = task_dir / "tests"
    dst = work / ".tests"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    workdir_str = str(work)
    for f in dst.rglob("*"):
        if f.is_file() and f.suffix in (".py", ".sh", ".txt"):
            try:
                txt = f.read_text()
            except UnicodeDecodeError:
                continue
            new = _rewrite_paths(txt, workdir_str)
            # Also handle Path("/foo") without quotes preceding
            for src_root, suf in _PATH_ROOTS.items():
                dst_abs = (workdir_str + suf).rstrip("/")
                new = new.replace(f"Path({src_root!r}", f"Path({dst_abs!r}")
                new = new.replace(f'Path("{src_root}")', f'Path("{dst_abs}")')
                new = new.replace(f"Path('{src_root}')", f"Path('{dst_abs}')")
                new = new.replace(f'sys.path.insert(0, "{src_root}', f'sys.path.insert(0, "{dst_abs}')
                new = new.replace(f"sys.path.insert(0, '{src_root}", f"sys.path.insert(0, '{dst_abs}")
            if new != txt:
                f.write_text(new)
    return dst


async def call_claude(
    *, system_prompt: str, user_prompt: str, cwd: Path, model: str,
    timeout_s: int = 900,
) -> tuple[int, str, str, float]:
    """Run claude -p in the workspace with Bash tool access. Returns (rc, stdout, stderr, wall)."""
    import tempfile
    t0 = time.time()
    # Write system prompt to temp file to avoid ARG_MAX limits on large pools.
    sp_file = Path(tempfile.mktemp(suffix=".txt", dir=str(cwd)))
    sp_file.write_text(system_prompt)
    try:
        proc = await asyncio.create_subprocess_exec(
            "claude", "-p",
            "--model", model,
            "--system-prompt-file", str(sp_file),
            "--output-format", "text",
            "--allowed-tools", "Bash,Read,Write,Edit,Glob,Grep",
            "--no-session-persistence",
            "--permission-mode", "bypassPermissions",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(input=user_prompt.encode()), timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return 124, "", "TIMEOUT", time.time() - t0
        return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace"), time.time() - t0
    finally:
        sp_file.unlink(missing_ok=True)


def run_tests(tests_dir: Path, work: Path, python: str) -> tuple[bool, str]:
    """Run the patched tests against the workspace. Returns (passed, log)."""
    env = dict(os.environ)
    env["WORKDIR"] = str(work)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [python, "-m", "pytest", str(tests_dir / "test_outputs.py"), "-rA", "-v", "--no-header"]
    try:
        r = subprocess.run(cmd, env=env, cwd=str(work), timeout=300,
                           capture_output=True, text=True)
        return r.returncode == 0, (r.stdout + "\n--- stderr ---\n" + r.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return False, "TESTS_TIMEOUT"


async def run_one_trial(
    *, dataset: str, task: dict, pool_size: int, noise_mode: str, seed: int,
    model: str, corpus_path: Path, work_root: Path, python: str,
) -> dict:
    trial_id = f"{dataset}__{task['task_id']}__sz{pool_size}__{noise_mode}__s{seed}"
    trial_dir = work_root / trial_id

    base = {"trial_id": trial_id, "task_id": task["task_id"],
            "pool_size": pool_size, "noise_mode": noise_mode, "seed": seed}

    pool: PoolSpec | None = build_pool(
        dataset=dataset, task=task, pool_size=pool_size,
        noise_mode=noise_mode, seed=seed,
    )
    if pool is None:
        return {**base, "skipped": True,
                "reason": f"|GT|={len(task.get('gt_skills') or [])} > pool_size={pool_size}"}

    if dataset != "sb":
        return {**base, "skipped": True,
                "reason": "tb host-execution not implemented (needs container)"}

    task_dir = SB_TASKS_ROOT / task["task_id"]
    if not task_dir.exists():
        return {**base, "error": f"task_dir missing: {task_dir}"}
    if not is_lightweight(task_dir):
        return {**base, "skipped": True,
                "reason": "heavy Dockerfile (host execution unsafe)"}

    work = trial_dir / "work"
    setup_workspace(task_dir, work)
    tests_dir = patch_tests(task_dir, work)
    instr_path = task_dir / "instruction.md"
    if not instr_path.exists():
        return {**base, "error": "instruction.md missing"}
    user_prompt = patch_instruction(instr_path.read_text(), str(work))

    if noise_mode == "noskill":
        # Baseline: no skills, just the preamble
        from exec_eval_prefill.prompt_assembler import SYSTEM_PREAMBLE
        system_prompt = SYSTEM_PREAMBLE
    elif noise_mode == "emptyframe":
        # Control: skill prompt framing but zero skill blocks
        from exec_eval_prefill.prompt_assembler import SYSTEM_PREAMBLE
        system_prompt = (
            SYSTEM_PREAMBLE
            + "\n\nThe following skills are prefilled. Use any whose instructions match the task.\n\n"
            + "(No skills available.)\n\n---\n\nTask:\n"
        )
    elif not pool.candidates:
        from exec_eval_prefill.prompt_assembler import SYSTEM_PREAMBLE
        system_prompt = SYSTEM_PREAMBLE
    else:
        system_prompt = build_system_prompt(
            candidate_slugs=pool.candidates,
            task_instruction="",  # we put task in user prompt
            sb_task_dir=task_dir,
            corpus_path=corpus_path,
        )

    rc, agent_out, agent_err, agent_wall = await call_claude(
        system_prompt=system_prompt, user_prompt=user_prompt, cwd=work, model=model,
    )

    # Reconcile output paths before running tests (fixes path rewriting false negatives)
    _reconcile_output_paths(work)

    passed, test_log = run_tests(tests_dir, work, python)

    return {
        "trial_id": trial_id,
        "dataset": dataset,
        "task_id": task["task_id"],
        "pool_size": pool_size,
        "noise_mode": noise_mode,
        "seed": seed,
        "gt_slugs": pool.gt_slugs,
        "gt_positions": pool.gt_positions,
        "n_candidates": len(pool.candidates),
        "model": model,
        "agent_rc": rc,
        "agent_wall_s": round(agent_wall, 1),
        "agent_stdout_tail": agent_out[-1500:],
        "agent_stderr_tail": agent_err[-1500:],
        "pass": passed,
        "test_log_tail": test_log[-2000:],
        "system_prompt_chars": len(system_prompt),
    }


async def main_async(args):
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]
    tasks = [t for t in tasks if t.get("gt_skills")]
    if args.task_id:
        tasks = [t for t in tasks if t["task_id"] in set(args.task_id)]
    if args.limit:
        tasks = tasks[: args.limit]

    done = set()
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                # Skip rate-limited results so they get retried
                stdout = (r.get("agent_stdout_tail") or "").lower()
                if "hit your limit" in stdout or re.search(r"resets \d+:\d+", stdout):
                    continue
                if r.get("pass") is not None or "error" in r or r.get("skipped"):
                    done.add((r.get("task_id"), r.get("pool_size"), r.get("noise_mode"), r.get("seed")))

    trials = [
        (t, sz, mode, sd)
        for t in tasks
        for sz in args.pool_sizes
        for mode in args.noise_modes
        for sd in args.seeds
        if (t["task_id"], sz, mode, sd) not in done
    ]
    if args.skip_trials:
        trials = trials[args.skip_trials:]
    if args.max_trials and len(trials) > args.max_trials:
        trials = trials[:args.max_trials]
    print(f"[{args.dataset.upper()}] tasks={len(tasks)} trials={len(trials)} (done={len(done)})",
          file=sys.stderr, flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    fout = args.out.open("a")

    async def _go(t, sz, mode, sd):
        tid = f"{args.dataset}__{t['task_id']}__sz{sz}__{mode}__s{sd}"
        print(f"  [START] {tid}", file=sys.stderr, flush=True)
        async with sem:
            r = await run_one_trial(
                dataset=args.dataset, task=t, pool_size=sz, noise_mode=mode, seed=sd,
                model=args.model, corpus_path=args.corpus,
                work_root=args.work_root, python=args.python,
            )
            print(f"  [DONE]  {tid} -> pass={r.get('pass')}", file=sys.stderr, flush=True)
            return r

    completed = 0
    t_start = time.time()
    try:
        coros = [_go(t, sz, m, s) for (t, sz, m, s) in trials]
        for fut in asyncio.as_completed(coros):
            r = await fut
            fout.write(json.dumps(r) + "\n"); fout.flush()
            completed += 1
            tag = ("PASS" if r.get("pass") else
                   "SKIP" if r.get("skipped") else
                   "FAIL" if r.get("pass") is False else
                   "ERR")
            print(f"[{completed}/{len(trials)} {(time.time()-t_start):.0f}s] "
                  f"{r.get('trial_id', r)} -> {tag} ({r.get('reason') or r.get('error') or ''})",
                  file=sys.stderr)
    finally:
        fout.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["sb", "tb"], required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work-root", type=Path, default=Path("/tmp/exec_prefill"))
    p.add_argument("--corpus", type=Path,
                   default=Path(os.environ.get("SKILL_CORPUS_PATH",
                                               "data/processed/skill_corpus.jsonl")))
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--pool-sizes", type=int, nargs="+", default=[5, 10, 20, 50])
    p.add_argument("--noise-modes", nargs="+", default=["random", "hard", "easy"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--task-id", nargs="+", default=None,
                   help="Restrict to specific task IDs (smoke testing).")
    p.add_argument("--python", default="/anvil/projects/x-cis260386/william/llm-colearning/envs/marl/bin/python")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-trials", type=int, default=None,
                   help="Stop after this many trials (for sequential container mode).")
    p.add_argument("--skip-trials", type=int, default=0,
                   help="Skip first N remaining trials (for parallel container sharding).")
    args = p.parse_args(argv)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
