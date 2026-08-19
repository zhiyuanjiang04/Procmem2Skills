#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CLEAN_ROOT = Path(__file__).resolve().parents[1]

PROVIDER_DEFAULT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GOOGLE_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "uniapi": "UNIAPI_API_KEY",
}


@dataclass(frozen=True)
class SimilarTaskRef:
    task_uid: str
    task_name: str
    benchmark_id: str
    benchmark_slug: str
    similarity: float


@dataclass(frozen=True)
class TargetTask:
    task_uid: str
    task_name: str
    benchmark_id: str
    benchmark_slug: str


def _normalize_slug(value: str, *, default: str = "value", max_len: int = 80) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
    if not tokens:
        return default
    slug = "-".join(tokens)
    if len(slug) <= max_len:
        return slug
    trimmed = slug[:max_len].rstrip("-")
    return trimmed or default


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_traces(trace_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(trace_root.rglob("*.json")):
        if path.name.endswith("summary.json"):
            continue
        payload = _load_json(path)
        if payload is None:
            continue
        if payload.get("task_name") and payload.get("trial_name"):
            rows.append(payload)
    return rows


def _group_traces_by_task(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        task_name = str(rec.get("task_name") or "").strip()
        benchmark_id = str(rec.get("benchmark") or "").strip()
        if not task_name or not benchmark_id:
            continue
        key = (benchmark_id, task_name)
        bucket = out.setdefault(
            key,
            {
                "benchmark_id": benchmark_id,
                "task_name": task_name,
                "success": [],
                "failure": [],
            },
        )
        status = str(rec.get("status") or "runtime_error").strip().lower()
        if status == "success":
            bucket["success"].append(rec)
        elif status == "failure":
            bucket["failure"].append(rec)
    return out


def _prepare_provider_env(provider: str, api_key_env: str, base_url: str | None) -> dict[str, str]:
    env = os.environ.copy()
    key = env.get(api_key_env)
    if not key:
        raise RuntimeError(f"missing API key env: {api_key_env}")

    p = provider.lower().strip()
    if p == "openai":
        env["OPENAI_API_KEY"] = key
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
        return env
    if p == "openrouter":
        env["OPENROUTER_API_KEY"] = key
        env["OPENAI_API_KEY"] = key
        env.setdefault("OPENAI_BASE_URL", base_url or env.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1")
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
            env["OPENROUTER_BASE_URL"] = base_url
        return env
    if p == "uniapi":
        env["UNIAPI_API_KEY"] = key
        env["OPENAI_API_KEY"] = key
        env.setdefault("OPENAI_BASE_URL", base_url or env.get("UNIAPI_BASE_URL") or "https://api.uniapi.io/v1")
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
            env["UNIAPI_BASE_URL"] = base_url
        return env
    if p in {"google", "claude"}:
        if not base_url:
            raise RuntimeError(f"provider={p} requires --base-url")
        env["OPENAI_API_KEY"] = key
        env["OPENAI_BASE_URL"] = base_url
        return env
    raise RuntimeError(f"unsupported provider: {provider}")


def _resolve_stable_harbor_runtime(*, procmem2skills_root: Path, env: dict[str, str]) -> dict[str, str]:
    out = dict(env)
    candidates = [
        procmem2skills_root,
        Path(os.environ.get("PROCMEM2SKILLS_ROOT", "/raid/zhiyuan/procmem2skills")),
    ]
    for root in candidates:
        harbor_bin = root / ".venv-py312" / "bin" / "harbor"
        python_bin = root / ".venv-py312" / "bin" / "python"
        if harbor_bin.is_file() and python_bin.is_file():
            out["HARBOR_BIN"] = str(harbor_bin)
            out["PROCMEM_BENCHMARK_PYTHON"] = str(python_bin)
            out.setdefault("PM2S_RUNTIME_ROOT", str(root))
            return out
    return out


def _collect_golden_skills(golden_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for skill_md in sorted(golden_root.rglob("SKILL.md")):
        parts = skill_md.parts
        if len(parts) < 4:
            continue
        bench = parts[-4]
        mode = parts[-3]
        task = parts[-2]
        if mode != "normal":
            continue
        key = f"{bench}::{task}"
        out[key] = skill_md
    return out


def _load_similarity_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    if payload is None:
        raise RuntimeError(f"invalid similarity json: {path}")
    rows = payload.get("tasks")
    if not isinstance(rows, list):
        raise RuntimeError(f"invalid similarity json tasks[]: {path}")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        uid = str(row.get("task_uid") or "").strip()
        if uid:
            out[uid] = row
    return out


def _extract_targets(
    *,
    by_task: dict[tuple[str, str], dict[str, Any]],
    m_success: int,
    n_failure: int,
    task_filter_uids: set[str],
) -> list[TargetTask]:
    selected: list[TargetTask] = []
    for (benchmark_id, task_name), row in sorted(by_task.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if len(row.get("success") or []) < max(0, int(m_success)):
            continue
        if len(row.get("failure") or []) < max(0, int(n_failure)):
            continue
        if benchmark_id == "skillsbench":
            bench_slug = "skillsbench"
        elif benchmark_id == "terminal-bench-2":
            bench_slug = "terminalbench2"
        elif benchmark_id == "terminal-bench-pro-1":
            bench_slug = "terminalbenchpro"
        else:
            bench_slug = _normalize_slug(benchmark_id, default="benchmark", max_len=48)
        uid = f"{benchmark_id}::{task_name}"
        if task_filter_uids and uid not in task_filter_uids:
            continue
        selected.append(
            TargetTask(
                task_uid=uid,
                task_name=task_name,
                benchmark_id=benchmark_id,
                benchmark_slug=bench_slug,
            )
        )
    return selected


def _materialize_related_skill_repo(
    *,
    repo_root: Path,
    condition: str,
    targets: list[TargetTask],
    similarity_map: dict[str, dict[str, Any]],
    golden_skills: dict[str, Path],
    k: int,
) -> dict[str, Any]:
    condition_root = repo_root / condition
    if condition_root.exists():
        shutil.rmtree(condition_root)
    condition_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []

    for target in targets:
        task_slug = _normalize_slug(target.task_name, default="task", max_len=64)
        out_dir = condition_root / task_slug
        out_dir.mkdir(parents=True, exist_ok=True)

        row = similarity_map.get(target.task_uid)
        refs = row.get("similar_tasks") if isinstance(row, dict) else []
        if not isinstance(refs, list):
            refs = []

        selected_refs: list[SimilarTaskRef] = []
        for item in refs:
            if not isinstance(item, dict):
                continue
            ref_uid = str(item.get("task_uid") or "").strip()
            if not ref_uid or ref_uid == target.task_uid:
                continue
            ref_name = str(item.get("task_name") or "").strip()
            ref_bid = str(item.get("benchmark_id") or "").strip()
            ref_bslug = str(item.get("benchmark_slug") or "").strip()
            sim = float(item.get("similarity") or 0.0)
            selected_refs.append(
                SimilarTaskRef(
                    task_uid=ref_uid,
                    task_name=ref_name,
                    benchmark_id=ref_bid,
                    benchmark_slug=ref_bslug,
                    similarity=sim,
                )
            )
            if len(selected_refs) >= max(1, int(k)):
                break

        body_lines = [
            "---",
            f"name: crossbench-related-skills-{task_slug}",
            f"description: Cross-benchmark top-{int(k)} related skills for task {target.task_name}.",
            "---",
            "",
            "# Cross-Benchmark Related Skills",
            "",
            "Use the following related skills as references. Adapt commands and paths to the current workspace and observations.",
            "",
        ]

        copied_neighbors: list[dict[str, Any]] = []
        for ref in selected_refs:
            key = f"{ref.benchmark_slug}::{ref.task_name}"
            src_skill = golden_skills.get(key)
            if src_skill is None:
                continue
            text = src_skill.read_text(encoding="utf-8", errors="replace")
            body_lines.append(f"## Neighbor: {ref.task_name} ({ref.benchmark_id}, sim={ref.similarity:.4f})")
            body_lines.append("")
            body_lines.append(text)
            body_lines.append("")
            copied_neighbors.append(
                {
                    "task_uid": ref.task_uid,
                    "task_name": ref.task_name,
                    "benchmark_id": ref.benchmark_id,
                    "benchmark_slug": ref.benchmark_slug,
                    "similarity": ref.similarity,
                    "source_skill": str(src_skill),
                }
            )

        if not copied_neighbors:
            body_lines.append("No related golden skill found for this task.")
            body_lines.append("")

        skill_md = out_dir / "SKILL.md"
        skill_md.write_text("\n".join(body_lines).strip() + "\n", encoding="utf-8")

        manifest_rows.append(
            {
                "task_uid": target.task_uid,
                "task_name": target.task_name,
                "benchmark_id": target.benchmark_id,
                "benchmark_slug": target.benchmark_slug,
                "skill_md": str(skill_md),
                "neighbors": copied_neighbors,
                "neighbor_count": len(copied_neighbors),
            }
        )

    payload = {
        "condition": condition,
        "task_count": len(targets),
        "rows": manifest_rows,
    }
    _json_dump(repo_root / "synthetic_related_skills_manifest.json", payload)
    return payload


def _run_one_benchmark_eval(
    *,
    pm2s_root: Path,
    procmem2skills_root: Path,
    trace_root: Path,
    benchmark_config: Path,
    task_source_root: Path,
    eval_output_root: Path,
    benchmark_output: str,
    synthetic_skills_root: Path,
    condition: str,
    provider: str,
    api_key_env: str,
    base_url: str | None,
    agent: str,
    model: str,
    m_success: int,
    n_failure: int,
    n_attempts: int,
    n_concurrent: int,
    max_steps: int,
    command_timeout_sec: int,
    docker_cleanup: bool,
    docker_cleanup_timeout_sec: int,
    docker_cleanup_strict: bool,
    run_id: str,
    task_names: list[str],
    dry_run: bool,
) -> tuple[int, list[str], Path]:
    script_candidates = [
        pm2s_root / "scripts" / "run_context_comparison.py",
        pm2s_root / "analysis" / "skill_vs_procmem" / "scripts" / "run_context_comparison.py",
    ]
    script = next((candidate for candidate in script_candidates if candidate.is_file()), None)
    if script is None:
        raise RuntimeError(f"missing script; checked: {script_candidates}")

    cmd = [
        sys.executable,
        str(script),
        "--trace-root",
        str(trace_root),
        "--skills-root",
        str(synthetic_skills_root),
        "--benchmark-config",
        str(benchmark_config),
        "--task-source-root",
        str(task_source_root),
        "--procmem2skills-root",
        str(procmem2skills_root),
        "--output-root",
        str(eval_output_root),
        "--output-layout",
        "normal",
        "--benchmark-output",
        str(benchmark_output),
        "--provider",
        str(provider),
        "--api-key-env",
        str(api_key_env),
        "--agent",
        str(agent),
        "--model",
        str(model),
        "--m-success",
        str(max(0, int(m_success))),
        "--n-failure",
        str(max(0, int(n_failure))),
        "--n-attempts",
        str(max(1, int(n_attempts))),
        "--n-concurrent",
        str(max(1, int(n_concurrent))),
        "--max-steps",
        str(max(1, int(max_steps))),
        "--command-timeout-sec",
        str(max(1, int(command_timeout_sec))),
        "--run-id",
        str(run_id),
        "--arms",
        "skill",
        "--docker-cleanup-timeout-sec",
        str(max(1, int(docker_cleanup_timeout_sec))),
    ]

    if base_url:
        cmd.extend(["--base-url", str(base_url)])
    if docker_cleanup:
        cmd.append("--docker-cleanup")
    else:
        cmd.append("--no-docker-cleanup")
    if docker_cleanup_strict:
        cmd.append("--docker-cleanup-strict")

    for task_name in task_names:
        cmd.extend(["--task-name", task_name])

    if dry_run:
        cmd.append("--dry-run")

    env = _prepare_provider_env(provider=provider, api_key_env=api_key_env, base_url=base_url)
    env = _resolve_stable_harbor_runtime(procmem2skills_root=procmem2skills_root, env=env)

    completed = subprocess.run(
        cmd,
        cwd=pm2s_root,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    tails = [
        "[stdout_tail]",
        (completed.stdout or "")[-3000:],
        "[stderr_tail]",
        (completed.stderr or "")[-3000:],
    ]

    report_path = eval_output_root / benchmark_output / condition / "runs" / run_id / "comparison_report.json"
    return int(completed.returncode), tails, report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-benchmark similarity-based skill injection eval from static similarity JSON."
    )
    parser.add_argument("--similarity-json", type=Path, required=True)
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=CLEAN_ROOT / "retrieval-data" / "traces",
    )
    parser.add_argument(
        "--golden-skills-root",
        type=Path,
        default=CLEAN_ROOT / "retrieval-data" / "golden-skills",
    )
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=CLEAN_ROOT / "configs" / "benchmarks.json",
    )
    parser.add_argument(
        "--task-source-root",
        type=Path,
        default=Path(os.environ.get("TASK_SOURCE_ROOT", "/raid/zhiyuan/procmem2skills/benchmarks/harbor-datasets")),
    )
    parser.add_argument("--procmem2skills-root", type=Path, default=Path(os.environ.get("PROCMEM2SKILLS_ROOT", "/raid/zhiyuan/procmem2skills")))
    parser.add_argument("--pm2s-root", type=Path, default=CLEAN_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CLEAN_ROOT / "outputs" / "transfer" / "concrete",
        help="Base output root. Runs are stored under {output_root}/top-{k}/runs/{run_id}.",
    )

    parser.add_argument("--provider", choices=sorted(PROVIDER_DEFAULT_KEY_ENV.keys()), default="uniapi")
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--agent", default="codex")
    parser.add_argument("--model", default="gpt-5.3-codex")

    parser.add_argument("--m-success", type=int, default=5)
    parser.add_argument("--n-failure", type=int, default=5)
    parser.add_argument("--n-attempts", type=int, default=5)
    parser.add_argument("--n-concurrent", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--command-timeout-sec", type=int, default=1200)
    parser.add_argument("--docker-cleanup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--docker-cleanup-timeout-sec", type=int, default=180)
    parser.add_argument("--docker-cleanup-strict", action="store_true")

    parser.add_argument("--k", type=int, required=True, help="Use top-k similar tasks from similarity JSON.")
    parser.add_argument("--task-uid", action="append", default=[])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if int(args.k) <= 0:
        raise RuntimeError("--k must be > 0")

    similarity_map = _load_similarity_map(args.similarity_json.resolve())
    golden_skills = _collect_golden_skills(args.golden_skills_root.resolve())

    records = _load_traces(args.trace_root.resolve())
    by_task = _group_traces_by_task(records)

    task_filter_uids = {t.strip() for t in (args.task_uid or []) if t.strip()}
    targets = _extract_targets(
        by_task=by_task,
        m_success=int(args.m_success),
        n_failure=int(args.n_failure),
        task_filter_uids=task_filter_uids,
    )
    if not targets:
        raise RuntimeError("no eligible target tasks")

    # Only keep tasks that exist in similarity map.
    targets = [t for t in targets if t.task_uid in similarity_map]
    if not targets:
        raise RuntimeError("no eligible target tasks found in similarity json")

    condition = f"{max(0, int(args.m_success))}s{max(0, int(args.n_failure))}f"
    run_slug = _normalize_slug(str(args.run_id or f"crossbench-k{int(args.k)}-{int(time.time())}"), default="crossbench", max_len=96)

    run_root = args.output_root.resolve() / f"top-{int(args.k)}" / "runs" / run_slug
    run_root.mkdir(parents=True, exist_ok=True)

    synthetic_root = run_root / "synthetic-skills"
    synthetic_manifest = _materialize_related_skill_repo(
        repo_root=synthetic_root,
        condition=condition,
        targets=targets,
        similarity_map=similarity_map,
        golden_skills=golden_skills,
        k=int(args.k),
    )

    by_bench: dict[str, list[TargetTask]] = {}
    for t in targets:
        by_bench.setdefault(t.benchmark_slug, []).append(t)

    api_key_env = str(args.api_key_env or PROVIDER_DEFAULT_KEY_ENV[str(args.provider).lower()])
    eval_root = run_root / "eval"

    bench_results: dict[str, Any] = {}
    worst_rc = 0

    for bench_slug, rows in sorted(by_bench.items(), key=lambda kv: kv[0]):
        task_names = [x.task_name for x in rows]
        eval_run_id = _normalize_slug(f"crossbench-sim-k{int(args.k)}-{bench_slug}-{run_slug}", default="run", max_len=96)
        rc, tails, report_path = _run_one_benchmark_eval(
            pm2s_root=args.pm2s_root.resolve(),
            procmem2skills_root=args.procmem2skills_root.resolve(),
            trace_root=args.trace_root.resolve(),
            benchmark_config=args.benchmark_config.resolve(),
            task_source_root=args.task_source_root.resolve(),
            eval_output_root=eval_root,
            benchmark_output=bench_slug,
            synthetic_skills_root=synthetic_root,
            condition=condition,
            provider=str(args.provider),
            api_key_env=api_key_env,
            base_url=args.base_url,
            agent=str(args.agent),
            model=str(args.model),
            m_success=int(args.m_success),
            n_failure=int(args.n_failure),
            n_attempts=int(args.n_attempts),
            n_concurrent=int(args.n_concurrent),
            max_steps=int(args.max_steps),
            command_timeout_sec=int(args.command_timeout_sec),
            docker_cleanup=bool(args.docker_cleanup),
            docker_cleanup_timeout_sec=int(args.docker_cleanup_timeout_sec),
            docker_cleanup_strict=bool(args.docker_cleanup_strict),
            run_id=eval_run_id,
            task_names=task_names,
            dry_run=bool(args.dry_run),
        )
        report = _load_json(report_path)
        bench_results[bench_slug] = {
            "task_count": len(task_names),
            "tasks": task_names,
            "return_code": rc,
            "report_path": str(report_path),
            "report": report,
            "tails": tails,
            "eval_run_id": eval_run_id,
        }
        if rc != 0:
            worst_rc = rc if worst_rc == 0 else worst_rc

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "k": int(args.k),
        "condition": condition,
        "provider": str(args.provider),
        "api_key_env": api_key_env,
        "base_url": args.base_url,
        "agent": str(args.agent),
        "model": str(args.model),
        "run_slug": run_slug,
        "run_root": str(run_root),
        "similarity_json": str(args.similarity_json.resolve()),
        "target_task_count": len(targets),
        "synthetic_skill_manifest": synthetic_manifest,
        "benchmarks": bench_results,
        "dry_run": bool(args.dry_run),
    }
    _json_dump(run_root / "crossbench_transfer_eval_summary.json", summary)

    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "k": int(args.k),
                "target_task_count": len(targets),
                "benchmarks": {k: {"task_count": v["task_count"], "return_code": v["return_code"]} for k, v in bench_results.items()},
                "dry_run": bool(args.dry_run),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return worst_rc


if __name__ == "__main__":
    raise SystemExit(main())
