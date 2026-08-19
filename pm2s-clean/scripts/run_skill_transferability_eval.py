#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
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

SCOPE_TO_BENCHMARK_IDS = {
    "all": set(),
    "tb2": {"terminal-bench-2"},
    "tbpro": {"terminal-bench-pro-1"},
    "sb": {"skillsbench"},
}

KNOWN_BENCHMARK_SLUGS = {"terminalbench2", "terminalbenchpro", "skillsbench"}


@dataclass
class TaskPoolItem:
    task_uid: str
    benchmark_id: str
    benchmark_slug: str
    task_name: str
    task_slug: str
    success_count: int
    failure_count: int
    instruction_path: Path
    instruction_text: str
    skill_md_path: Path
    skill_text: str


@dataclass
class TransferEvalOutcome:
    space: str
    benchmark_slug: str
    k: int
    run_id: str
    return_code: int
    report_path: Path
    per_task_success_rate: dict[str, float]
    overall_success_rate: float | None
    tails: list[str]


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


def _jsonl_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _group_traces_by_uid(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    by_uid: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        task_name = str(rec.get("task_name") or "").strip()
        benchmark_id = str(rec.get("benchmark") or "").strip()
        if not task_name or not benchmark_id:
            continue
        key = (benchmark_id, task_name)
        bucket = by_uid.setdefault(
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
    return by_uid


def _load_benchmark_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    if payload is None:
        raise RuntimeError(f"invalid benchmark config: {path}")
    rows = payload.get("benchmarks")
    if not isinstance(rows, list):
        raise RuntimeError(f"invalid benchmark config rows: {path}")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        bid = str(row.get("id") or "").strip()
        if bid:
            out[bid] = row
    return out


def _benchmark_output_slug(*, benchmark_id: str, benchmark_cfg: dict[str, Any] | None) -> str:
    bid = str(benchmark_id or "").strip().lower()
    cfg = benchmark_cfg or {}
    dataset = str(cfg.get("dataset") or "").strip().lower()
    runner = str(cfg.get("runner") or "").strip().lower()

    text = " ".join(x for x in [bid, dataset, runner] if x)
    if "skillsbench" in text or "skills-bench" in text:
        return "skillsbench"
    if "terminal-bench-pro" in text or "tbpro" in text:
        return "terminalbenchpro"
    if "terminal-bench@2" in text or "terminal-bench-2" in text or "tb2" in text:
        return "terminalbench2"
    return _normalize_slug(bid or dataset or "benchmark", default="benchmark", max_len=48)


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
        provider_key = env.get("OPENROUTER_API_KEY")
        if api_key_env != "OPENROUTER_API_KEY" and provider_key and provider_key != key:
            raise RuntimeError(
                "provider=openrouter requested with mismatched api_key_env; "
                "use --api-key-env OPENROUTER_API_KEY or align the exported keys"
            )
        env["OPENROUTER_API_KEY"] = key
        env["OPENAI_API_KEY"] = key
        env.setdefault("OPENAI_BASE_URL", base_url or env.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1")
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
            env["OPENROUTER_BASE_URL"] = base_url
        return env
    if p == "uniapi":
        provider_key = env.get("UNIAPI_API_KEY")
        if api_key_env != "UNIAPI_API_KEY" and provider_key and provider_key != key:
            raise RuntimeError(
                "provider=uniapi requested with mismatched api_key_env; "
                "use --api-key-env UNIAPI_API_KEY or align the exported keys"
            )
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
        Path("/raid/zhiyuan/procmem2skills"),
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


def _load_qwen_embeddings(
    texts: list[str],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> list[list[float]]:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "embedding dependencies unavailable. run with /raid/zhiyuan/procmem2skills/.venv-py312/bin/python"
        ) from exc

    if not texts:
        return []

    actual_device = device
    if actual_device == "auto":
        actual_device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(actual_device)
    model.eval()

    vectors: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(texts), max(1, batch_size)):
            chunk = texts[start : start + max(1, batch_size)]
            encoded = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max(32, int(max_length)),
                return_tensors="pt",
            )
            encoded = {k: v.to(actual_device) for k, v in encoded.items()}
            out = model(**encoded)
            hidden = out.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            pooled = F.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.detach().cpu().tolist())
    return vectors


def _dot(a: list[float], b: list[float]) -> float:
    return float(sum((x * y) for x, y in zip(a, b)))


def _strip_frontmatter(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return "\n".join(lines[idx + 1 :]).strip()
    return text.strip()


def _clean_text(text: str, *, char_limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) > max(1, int(char_limit)):
        cleaned = cleaned[: max(1, int(char_limit))].rstrip()
    return cleaned


def _run_eval(
    *,
    pm2s_root: Path,
    procmem2skills_root: Path,
    trace_root: Path,
    benchmark_config: Path,
    task_source_root: Path,
    eval_output_root: Path,
    benchmark_output: str,
    skills_root: Path,
    skills_manifest: Path | None,
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
) -> tuple[int, list[str]]:
    script = pm2s_root / "analysis" / "skill_vs_procmem" / "scripts" / "run_context_comparison.py"
    if not script.is_file():
        raise RuntimeError(f"missing script: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--trace-root",
        str(trace_root),
        "--skills-root",
        str(skills_root),
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
    if skills_manifest is not None:
        cmd.extend(["--skills-manifest", str(skills_manifest)])

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
        (completed.stdout or "")[-4000:],
        "[stderr_tail]",
        (completed.stderr or "")[-4000:],
    ]
    return int(completed.returncode), tails


# ---------------------------
# Cross-space implementation
# ---------------------------

def _parse_csv_ints(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def _parse_thresholds(raw: str) -> list[float]:
    txt = str(raw or "").strip()
    if not txt:
        return []
    if ":" in txt and "," not in txt:
        parts = [p.strip() for p in txt.split(":")]
        if len(parts) != 3:
            raise RuntimeError(f"invalid thresholds range: {txt}")
        start, end, step = float(parts[0]), float(parts[1]), float(parts[2])
        if step <= 0:
            raise RuntimeError("threshold step must be > 0")
        vals: list[float] = []
        cur = start
        eps = step / 10_000.0
        while cur <= end + eps:
            vals.append(round(cur, 8))
            cur += step
        return vals
    vals = [float(t.strip()) for t in txt.split(",") if t.strip()]
    vals = sorted(set(round(x, 8) for x in vals))
    return vals


def _match_scope(benchmark_id: str, benchmark_slug: str, scope: str) -> bool:
    scope_key = str(scope or "all").strip().lower()
    if scope_key == "all":
        return True
    allowed_ids = SCOPE_TO_BENCHMARK_IDS.get(scope_key)
    if allowed_ids and benchmark_id in allowed_ids:
        return True
    if scope_key == "tb2" and benchmark_slug == "terminalbench2":
        return True
    if scope_key == "tbpro" and benchmark_slug == "terminalbenchpro":
        return True
    if scope_key == "sb" and benchmark_slug == "skillsbench":
        return True
    return False


def _scan_skill_task_uids(
    *,
    skills_root: Path,
    benchmark_map: dict[str, dict[str, Any]],
) -> set[str]:
    skill_files = _scan_all_skill_files(skills_root)
    out: set[str] = set()
    for skill_md in skill_files:
        benchmark_slug = _infer_skill_benchmark_slug(skill_md)
        if not benchmark_slug:
            continue
        benchmark_id = ""
        for bid, cfg in benchmark_map.items():
            if _benchmark_output_slug(benchmark_id=bid, benchmark_cfg=cfg) == benchmark_slug:
                benchmark_id = bid
                break
        if not benchmark_id:
            continue
        task_name = skill_md.parent.name
        out.add(f"{benchmark_id}::{task_name}")
    return out


def _candidate_task_dirs(
    *,
    benchmark_id: str,
    task_name: str,
    task_source_root: Path,
    skillsbench_task_root: Path,
) -> list[Path]:
    cands: list[Path] = []
    roots: list[Path] = []
    if benchmark_id == "skillsbench":
        roots.extend(
            [
                skillsbench_task_root,
                Path("/raid/zhiyuan/procmem2skills/benchmarks/skillsbench/tasks"),
                Path("/raid/zhiyuan/procmem2skills/benchmarks/skillsbench"),
            ]
        )
    roots.append(task_source_root)

    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        direct = root / task_name
        nested = root / "tasks" / task_name
        for p in [direct, nested]:
            key = str(p)
            if key not in seen:
                cands.append(p)
                seen.add(key)
        # harbor-datasets pattern: /.../<random-id>/<task_name>
        try:
            for p in sorted(root.glob(f"*/{task_name}")):
                key = str(p)
                if key not in seen:
                    cands.append(p)
                    seen.add(key)
        except Exception:
            pass
    return cands


def _resolve_instruction_path(
    *,
    benchmark_id: str,
    task_name: str,
    task_source_root: Path,
    skillsbench_task_root: Path,
) -> Path | None:
    for candidate in _candidate_task_dirs(
        benchmark_id=benchmark_id,
        task_name=task_name,
        task_source_root=task_source_root,
        skillsbench_task_root=skillsbench_task_root,
    ):
        inst = candidate / "instruction.md"
        if inst.is_file():
            return inst
    return None


def _scan_all_skill_files(skills_root: Path) -> list[Path]:
    if not skills_root.is_dir():
        return []
    return sorted(skills_root.rglob("SKILL.md"))


def _infer_skill_benchmark_slug(skill_md: Path) -> str | None:
    parts = list(skill_md.parts)
    if len(parts) >= 4 and parts[-3] == "normal":
        bench = parts[-4]
        if bench in KNOWN_BENCHMARK_SLUGS:
            return bench
    if len(parts) >= 3:
        maybe = parts[-3]
        if maybe in KNOWN_BENCHMARK_SLUGS:
            return maybe
    return None


def _resolve_skill_path_for_task(
    *,
    skills_root: Path,
    benchmark_slug: str,
    task_name: str,
    task_slug: str,
    pre_scanned: list[Path],
) -> Path | None:
    direct_paths = [
        skills_root / benchmark_slug / "normal" / task_name / "SKILL.md",
        skills_root / benchmark_slug / "normal" / task_slug / "SKILL.md",
        skills_root / benchmark_slug / task_name / "SKILL.md",
        skills_root / benchmark_slug / task_slug / "SKILL.md",
        skills_root / task_name / "SKILL.md",
        skills_root / task_slug / "SKILL.md",
    ]
    for p in direct_paths:
        if p.is_file():
            return p

    for p in pre_scanned:
        p_task = p.parent.name
        p_slug = _normalize_slug(p_task, default="task", max_len=64)
        if p_slug != task_slug:
            continue
        p_bench = _infer_skill_benchmark_slug(p)
        if p_bench == benchmark_slug:
            return p
    for p in pre_scanned:
        p_task = p.parent.name
        p_slug = _normalize_slug(p_task, default="task", max_len=64)
        if p_slug == task_slug:
            return p
    return None


def _build_task_pool(
    *,
    trace_records: list[dict[str, Any]],
    benchmark_map: dict[str, dict[str, Any]],
    skills_root: Path,
    task_source_root: Path,
    skillsbench_task_root: Path,
    benchmark_scope: str,
    m_success: int,
    n_failure: int,
    text_char_limit: int,
    restrict_to_skill_tasks: bool,
) -> tuple[list[TaskPoolItem], list[dict[str, Any]]]:
    grouped = _group_traces_by_uid(trace_records)
    skill_files = _scan_all_skill_files(skills_root)
    allowed_skill_task_uids = (
        _scan_skill_task_uids(skills_root=skills_root, benchmark_map=benchmark_map) if restrict_to_skill_tasks else set()
    )

    missing_rows: list[dict[str, Any]] = []
    items: list[TaskPoolItem] = []

    for (benchmark_id, task_name), row in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        benchmark_slug = _benchmark_output_slug(benchmark_id=benchmark_id, benchmark_cfg=benchmark_map.get(benchmark_id))
        if not _match_scope(benchmark_id, benchmark_slug, benchmark_scope):
            continue
        task_uid = f"{benchmark_id}::{task_name}"
        if allowed_skill_task_uids and task_uid not in allowed_skill_task_uids:
            missing_rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "benchmark_slug": benchmark_slug,
                    "task_name": task_name,
                    "reason": "not_in_golden_skill_pool",
                }
            )
            continue

        succ = len(row.get("success") or [])
        fail = len(row.get("failure") or [])
        if succ < max(0, int(m_success)) or fail < max(0, int(n_failure)):
            missing_rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "benchmark_slug": benchmark_slug,
                    "task_name": task_name,
                    "reason": "insufficient_trials",
                    "success_count": succ,
                    "failure_count": fail,
                    "required_success": max(0, int(m_success)),
                    "required_failure": max(0, int(n_failure)),
                }
            )
            continue

        task_slug = _normalize_slug(task_name, default="task", max_len=64)
        instruction_path = _resolve_instruction_path(
            benchmark_id=benchmark_id,
            task_name=task_name,
            task_source_root=task_source_root,
            skillsbench_task_root=skillsbench_task_root,
        )
        if instruction_path is None:
            missing_rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "benchmark_slug": benchmark_slug,
                    "task_name": task_name,
                    "reason": "missing_instruction",
                }
            )
            continue

        raw_instruction = instruction_path.read_text(encoding="utf-8", errors="replace")
        instruction_text = _clean_text(raw_instruction, char_limit=max(1000, int(text_char_limit)))
        if not instruction_text:
            missing_rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "benchmark_slug": benchmark_slug,
                    "task_name": task_name,
                    "reason": "empty_instruction",
                }
            )
            continue

        skill_md = _resolve_skill_path_for_task(
            skills_root=skills_root,
            benchmark_slug=benchmark_slug,
            task_name=task_name,
            task_slug=task_slug,
            pre_scanned=skill_files,
        )
        if skill_md is None:
            missing_rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "benchmark_slug": benchmark_slug,
                    "task_name": task_name,
                    "reason": "missing_golden_skill",
                }
            )
            continue

        skill_text = skill_md.read_text(encoding="utf-8", errors="replace")
        skill_text = _clean_text(_strip_frontmatter(skill_text), char_limit=max(1000, int(text_char_limit)))
        if not skill_text:
            missing_rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "benchmark_slug": benchmark_slug,
                    "task_name": task_name,
                    "reason": "empty_skill_text",
                    "skill_md": str(skill_md),
                }
            )
            continue

        items.append(
            TaskPoolItem(
                task_uid=task_uid,
                benchmark_id=benchmark_id,
                benchmark_slug=benchmark_slug,
                task_name=task_name,
                task_slug=task_slug,
                success_count=succ,
                failure_count=fail,
                instruction_path=instruction_path,
                instruction_text=instruction_text,
                skill_md_path=skill_md,
                skill_text=skill_text,
            )
        )

    return items, missing_rows


def _pairwise_similarity_matrix(vectors: list[list[float]]) -> list[list[float]]:
    n = len(vectors)
    mat: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = _dot(vectors[i], vectors[j])
            if s > 1.0:
                s = 1.0
            elif s < -1.0:
                s = -1.0
            mat[i][j] = float(s)
            mat[j][i] = float(s)
    return mat


def _cluster_assignments_threshold_sweep(
    *,
    vectors: list[list[float]],
    thresholds: list[float],
    linkage_method: str,
) -> dict[str, list[int]]:
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist
    except Exception as exc:
        raise RuntimeError(
            "scipy unavailable for clustering. run with /raid/zhiyuan/procmem2skills/.venv-py312/bin/python"
        ) from exc

    if len(vectors) < 2:
        raise RuntimeError("need >=2 vectors for clustering")

    condensed = pdist(vectors, metric="cosine")
    z = linkage(condensed, method=linkage_method)
    out: dict[str, list[int]] = {}
    for t in thresholds:
        labels = [int(x) for x in fcluster(z, t=float(t), criterion="distance").tolist()]
        out[f"{float(t):.8f}"] = labels
    return out


def _cluster_stats(labels: list[int]) -> dict[str, Any]:
    n = len(labels)
    counts = Counter(labels)
    cluster_sizes = sorted(counts.values(), reverse=True)
    singleton_cluster_count = sum(1 for s in cluster_sizes if s == 1)
    singleton_task_count = sum(s for s in cluster_sizes if s == 1)
    non_singleton_task_count = n - singleton_task_count
    return {
        "task_count": n,
        "cluster_count": len(counts),
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "singleton_cluster_count": singleton_cluster_count,
        "singleton_cluster_ratio": float(singleton_cluster_count / len(counts)) if counts else 0.0,
        "singleton_task_count": singleton_task_count,
        "singleton_task_ratio": float(singleton_task_count / n) if n else 0.0,
        "non_singleton_task_coverage": float(non_singleton_task_count / n) if n else 0.0,
        "cluster_size_histogram": {str(k): int(v) for k, v in sorted(Counter(cluster_sizes).items())},
    }


def _entropy_from_counts(counts: list[int], n: int) -> float:
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log(p)
    return float(h)


def _variation_of_information(labels_a: list[int], labels_b: list[int]) -> tuple[float, float]:
    n = len(labels_a)
    if n == 0 or n != len(labels_b):
        return 0.0, 0.0

    ca = Counter(labels_a)
    cb = Counter(labels_b)
    hab = Counter(zip(labels_a, labels_b))

    h_a = _entropy_from_counts(list(ca.values()), n)
    h_b = _entropy_from_counts(list(cb.values()), n)

    mi = 0.0
    for (la, lb), c in hab.items():
        if c <= 0:
            continue
        p_ij = c / n
        p_i = ca[la] / n
        p_j = cb[lb] / n
        mi += p_ij * math.log(p_ij / (p_i * p_j))

    vi = float(h_a + h_b - (2.0 * mi))
    if n <= 1:
        return vi, 0.0
    vi_norm = float(vi / math.log(n)) if math.log(n) > 0 else 0.0
    return vi, vi_norm


def _pair_disagreement(labels_task: list[int], labels_skill: list[int]) -> dict[str, Any]:
    n = len(labels_task)
    if n != len(labels_skill):
        raise RuntimeError("pair disagreement input length mismatch")
    if n < 2:
        return {
            "pair_count": 0,
            "task_same_pair_count": 0,
            "skill_same_pair_count": 0,
            "task_same_skill_diff_count": 0,
            "skill_same_task_diff_count": 0,
            "t_same_s_diff_ratio": 0.0,
            "s_same_t_diff_ratio": 0.0,
        }

    pair_count = 0
    task_same = 0
    skill_same = 0
    t_same_s_diff = 0
    s_same_t_diff = 0

    for i in range(n):
        li_t = labels_task[i]
        li_s = labels_skill[i]
        for j in range(i + 1, n):
            pair_count += 1
            t_eq = li_t == labels_task[j]
            s_eq = li_s == labels_skill[j]
            if t_eq:
                task_same += 1
            if s_eq:
                skill_same += 1
            if t_eq and (not s_eq):
                t_same_s_diff += 1
            if s_eq and (not t_eq):
                s_same_t_diff += 1

    return {
        "pair_count": pair_count,
        "task_same_pair_count": task_same,
        "skill_same_pair_count": skill_same,
        "task_same_skill_diff_count": t_same_s_diff,
        "skill_same_task_diff_count": s_same_t_diff,
        "t_same_s_diff_ratio": float(t_same_s_diff / task_same) if task_same > 0 else 0.0,
        "s_same_t_diff_ratio": float(s_same_t_diff / skill_same) if skill_same > 0 else 0.0,
    }


def _partition_metrics(labels_task: list[int], labels_skill: list[int]) -> dict[str, Any]:
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn unavailable for ARI/NMI. run with /raid/zhiyuan/procmem2skills/.venv-py312/bin/python"
        ) from exc

    if len(labels_task) != len(labels_skill):
        raise RuntimeError("partition metric input length mismatch")

    ari = float(adjusted_rand_score(labels_task, labels_skill)) if labels_task else 0.0
    nmi = float(normalized_mutual_info_score(labels_task, labels_skill)) if labels_task else 0.0
    vi, vi_norm = _variation_of_information(labels_task, labels_skill)
    pair = _pair_disagreement(labels_task, labels_skill)
    return {
        "ari": ari,
        "nmi": nmi,
        "vi": vi,
        "vi_norm": vi_norm,
        **pair,
    }


def _bootstrap_ci(values: list[float], *, alpha: float = 0.05) -> dict[str, float | None]:
    if not values:
        return {"low": None, "high": None}
    vals = sorted(values)
    n = len(vals)
    low_idx = int(math.floor((alpha / 2.0) * (n - 1)))
    high_idx = int(math.ceil((1.0 - alpha / 2.0) * (n - 1)))
    return {"low": float(vals[low_idx]), "high": float(vals[high_idx])}


def _bootstrap_partition_metrics(
    *,
    labels_task: list[int],
    labels_skill: list[int],
    seeds: list[int],
    samples_per_seed: int,
) -> dict[str, Any]:
    n = len(labels_task)
    if n != len(labels_skill) or n < 2:
        return {}

    stats: dict[str, list[float]] = {
        "ari": [],
        "nmi": [],
        "vi": [],
        "vi_norm": [],
        "t_same_s_diff_ratio": [],
        "s_same_t_diff_ratio": [],
    }

    for seed in seeds:
        rng = random.Random(int(seed))
        for _ in range(max(1, int(samples_per_seed))):
            idx = [rng.randrange(n) for _ in range(n)]
            sub_t = [labels_task[i] for i in idx]
            sub_s = [labels_skill[i] for i in idx]
            m = _partition_metrics(sub_t, sub_s)
            for k in stats.keys():
                stats[k].append(float(m[k]))

    out: dict[str, Any] = {}
    for k, vals in stats.items():
        ci = _bootstrap_ci(vals)
        out[k] = {
            "mean": float(sum(vals) / len(vals)) if vals else None,
            "ci95": ci,
        }
    out["bootstrap_sample_count"] = sum(len(v) for v in stats.values()) // max(1, len(stats))
    return out


def _task_pool_hash(items: list[TaskPoolItem]) -> str:
    h = hashlib.sha256()
    for x in sorted(items, key=lambda it: it.task_uid):
        h.update(x.task_uid.encode("utf-8"))
        h.update(str(x.success_count).encode("utf-8"))
        h.update(str(x.failure_count).encode("utf-8"))
        h.update(str(x.instruction_path).encode("utf-8"))
        h.update(str(x.skill_md_path).encode("utf-8"))
    return h.hexdigest()


def _build_space_neighbors(
    items: list[TaskPoolItem],
    embeddings: list[list[float]],
    *,
    k: int,
) -> dict[str, list[dict[str, Any]]]:
    uids = [x.task_uid for x in items]
    n = len(uids)
    k_eff = min(max(1, int(k)), n - 1)

    sim = _pairwise_similarity_matrix(embeddings)
    out: dict[str, list[dict[str, Any]]] = {}
    for i, uid in enumerate(uids):
        scored: list[tuple[int, float]] = []
        for j in range(n):
            if i == j:
                continue
            scored.append((j, sim[i][j]))
        scored.sort(key=lambda x: x[1], reverse=True)
        rows: list[dict[str, Any]] = []
        for rank, (j, s) in enumerate(scored[:k_eff], start=1):
            ref = items[j]
            rows.append(
                {
                    "rank": rank,
                    "neighbor_task_uid": ref.task_uid,
                    "neighbor_task_name": ref.task_name,
                    "neighbor_benchmark_id": ref.benchmark_id,
                    "neighbor_benchmark_slug": ref.benchmark_slug,
                    "similarity": float(s),
                    "distance": float(1.0 - s),
                }
            )
        out[uid] = rows
    return out


def _build_intra_cluster_neighbors(
    *,
    items: list[TaskPoolItem],
    sim_matrix: list[list[float]],
    labels: list[int],
    k: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if len(items) != len(sim_matrix) or len(items) != len(labels):
        raise RuntimeError("intra-cluster neighbor input length mismatch")
    n = len(items)
    if n < 2:
        raise RuntimeError("need >=2 tasks for intra-cluster neighbors")

    by_cluster: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_cluster[int(label)].append(idx)

    k_eff = min(max(1, int(k)), n - 1)
    out: dict[str, list[dict[str, Any]]] = {}
    shortages: list[dict[str, Any]] = []
    no_candidate_count = 0

    for i, item in enumerate(items):
        same_cluster = [j for j in by_cluster[int(labels[i])] if j != i]
        scored: list[tuple[int, float]] = [(j, float(sim_matrix[i][j])) for j in same_cluster]
        scored.sort(key=lambda x: x[1], reverse=True)
        chosen = scored[:k_eff]

        rows: list[dict[str, Any]] = []
        for rank, (j, s) in enumerate(chosen, start=1):
            ref = items[j]
            rows.append(
                {
                    "rank": rank,
                    "neighbor_task_uid": ref.task_uid,
                    "neighbor_task_name": ref.task_name,
                    "neighbor_benchmark_id": ref.benchmark_id,
                    "neighbor_benchmark_slug": ref.benchmark_slug,
                    "similarity": float(s),
                    "distance": float(1.0 - s),
                }
            )
        out[item.task_uid] = rows

        if len(rows) == 0:
            no_candidate_count += 1
        if len(rows) < k_eff:
            shortages.append(
                {
                    "task_uid": item.task_uid,
                    "task_name": item.task_name,
                    "benchmark_slug": item.benchmark_slug,
                    "cluster_label": int(labels[i]),
                    "requested_k": int(k_eff),
                    "available_same_cluster_neighbors": len(same_cluster),
                }
            )

    stats = {
        "k_requested": int(k),
        "k_effective": int(k_eff),
        "task_count": n,
        "no_candidate_task_count": int(no_candidate_count),
        "insufficient_neighbor_task_count": int(len(shortages)),
        "coverage_ratio": float((n - no_candidate_count) / n) if n else 0.0,
        "shortages": shortages,
    }
    return out, stats


def _build_transfer_skill_pool_manifest(
    *,
    items: list[TaskPoolItem],
    neighbors_by_uid: dict[str, list[dict[str, Any]]],
    k: int,
    space_name: str,
) -> dict[str, Any]:
    item_map = {x.task_uid: x for x in items}
    rows: list[dict[str, Any]] = []

    for item in items:
        used: list[dict[str, Any]] = []
        for ref in neighbors_by_uid.get(item.task_uid, []):
            ref_uid = str(ref.get("neighbor_task_uid") or "")
            if not ref_uid:
                continue
            src = item_map.get(ref_uid)
            if src is None:
                continue
            # strict protocol: exclude self-skill
            if src.task_uid == item.task_uid:
                continue

            sim = float(ref.get("similarity") or 0.0)
            used.append(
                {
                    "neighbor_task_uid": src.task_uid,
                    "neighbor_task_name": src.task_name,
                    "neighbor_benchmark_id": src.benchmark_id,
                    "neighbor_benchmark_slug": src.benchmark_slug,
                    "similarity": sim,
                    "source_skill_md": str(src.skill_md_path),
                }
            )

        rows.append(
            {
                "task_uid": item.task_uid,
                "task_name": item.task_name,
                "benchmark_id": item.benchmark_id,
                "benchmark_slug": item.benchmark_slug,
                "task_slug": item.task_slug,
                "neighbor_count": len(used),
                "neighbors": used,
            }
        )

    payload = {
        "space": space_name,
        "k": int(k),
        "task_count": len(items),
        "rows": rows,
    }
    return payload


def _extract_skill_arm_success(report: dict[str, Any], benchmark_slug: str) -> tuple[dict[str, float], float | None]:
    arms = report.get("arms") if isinstance(report, dict) else None
    if not isinstance(arms, dict):
        return {}, None
    skill = arms.get("skill")
    if not isinstance(skill, dict):
        return {}, None
    overall = skill.get("overall")
    overall_rate = None
    if isinstance(overall, dict) and isinstance(overall.get("success_rate"), (int, float)):
        overall_rate = float(overall["success_rate"])

    benches = skill.get("benchmarks")
    if not isinstance(benches, dict):
        return {}, overall_rate

    # normal layout uses exactly one key per run_context_comparison call;
    # still try exact match first then fallback first entry.
    bench_node = benches.get(benchmark_slug)
    if not isinstance(bench_node, dict) and benches:
        first = next(iter(benches.values()))
        bench_node = first if isinstance(first, dict) else None
    if not isinstance(bench_node, dict):
        return {}, overall_rate

    per_task = bench_node.get("per_task")
    if not isinstance(per_task, dict):
        return {}, overall_rate

    out: dict[str, float] = {}
    for task_name, row in per_task.items():
        if not isinstance(row, dict):
            continue
        val = row.get("success_rate")
        if isinstance(val, (int, float)):
            out[str(task_name)] = float(val)
    return out, overall_rate


def _run_transfer_eval_for_space(
    *,
    space_name: str,
    selected_threshold_key: str,
    k: int,
    neighbors_by_uid: dict[str, list[dict[str, Any]]],
    neighbor_stats: dict[str, Any],
    items: list[TaskPoolItem],
    run_root: Path,
    condition: str,
    args: argparse.Namespace,
    api_key_env: str,
) -> dict[str, Any]:
    transfer_root = run_root / "transfer_eval" / space_name / f"t{selected_threshold_key}" / f"k{k}"
    summary_path = transfer_root / "transfer_eval_summary.json"
    if summary_path.is_file():
        cached = _load_json(summary_path)
        if isinstance(cached, dict):
            print(
                f"[resume] skip completed transfer eval: space={space_name} threshold={selected_threshold_key} k={k}"
            )
            return cached
    skills_manifest = transfer_root / "skills_manifest.json"

    manifest = _build_transfer_skill_pool_manifest(
        items=items,
        neighbors_by_uid=neighbors_by_uid,
        k=int(k),
        space_name=space_name,
    )
    _json_dump(skills_manifest, manifest)

    by_bench: dict[str, list[TaskPoolItem]] = defaultdict(list)
    for item in items:
        by_bench[item.benchmark_slug].append(item)

    outcomes: list[TransferEvalOutcome] = []
    eval_root = transfer_root / "eval"

    for bench_slug, bench_items in sorted(by_bench.items(), key=lambda kv: kv[0]):
        task_names = [x.task_name for x in bench_items]
        eval_run_id = _normalize_slug(
            f"transfer-{space_name}-k{k}-{bench_slug}-{args.run_id}",
            default="transfer",
            max_len=96,
        )
        rc, tails = _run_eval(
            pm2s_root=args.pm2s_root.resolve(),
            procmem2skills_root=args.procmem2skills_root.resolve(),
            trace_root=args.trace_root.resolve(),
            benchmark_config=args.benchmark_config.resolve(),
            task_source_root=args.task_source_root.resolve(),
            eval_output_root=eval_root,
            benchmark_output=bench_slug,
            skills_root=args.golden_skills_root.resolve(),
            skills_manifest=skills_manifest,
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
        report_path = eval_root / bench_slug / condition / "runs" / eval_run_id / "comparison_report.json"
        report = _load_json(report_path)
        per_task, overall_rate = _extract_skill_arm_success(report or {}, bench_slug)

        outcomes.append(
            TransferEvalOutcome(
                space=space_name,
                benchmark_slug=bench_slug,
                k=int(k),
                run_id=eval_run_id,
                return_code=int(rc),
                report_path=report_path,
                per_task_success_rate=per_task,
                overall_success_rate=overall_rate,
                tails=tails,
            )
        )

    by_uid: dict[str, float] = {}
    task_lookup = {(x.benchmark_slug, x.task_name): x.task_uid for x in items}
    for out in outcomes:
        for task_name, rate in out.per_task_success_rate.items():
            uid = task_lookup.get((out.benchmark_slug, task_name))
            if uid:
                by_uid[uid] = float(rate)

    payload = {
        "space": space_name,
        "selected_threshold": selected_threshold_key,
        "k": int(k),
        "condition": condition,
        "neighbor_stats": neighbor_stats,
        "synthetic_manifest": manifest,
        "outcomes": [
            {
                "benchmark_slug": o.benchmark_slug,
                "k": o.k,
                "run_id": o.run_id,
                "return_code": o.return_code,
                "report_path": str(o.report_path),
                "overall_success_rate": o.overall_success_rate,
                "per_task_count": len(o.per_task_success_rate),
                "stdout_stderr_tails": o.tails,
            }
            for o in outcomes
        ],
        "success_rate_by_task_uid": by_uid,
        "worst_return_code": max([o.return_code for o in outcomes], default=0),
    }
    _json_dump(transfer_root / "transfer_eval_summary.json", payload)
    return payload


def _choose_selected_threshold_pair(
    rows: list[dict[str, Any]],
    *,
    force_task_threshold: float | None,
    force_skill_threshold: float | None,
) -> dict[str, Any] | None:
    if not rows:
        return None
    if force_task_threshold is not None and force_skill_threshold is not None:
        tk = f"{float(force_task_threshold):.8f}"
        sk = f"{float(force_skill_threshold):.8f}"
        for row in rows:
            if row.get("task_threshold") == tk and row.get("skill_threshold") == sk:
                return row

    def key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            float(row.get("nmi") or 0.0),
            float(row.get("ari") or 0.0),
            -float(row.get("vi") or 0.0),
            float(row.get("t_same_s_diff_ratio") or 0.0),
            float(row.get("s_same_t_diff_ratio") or 0.0),
        )

    return max(rows, key=key)


def _choose_best_space_threshold(
    assignments: dict[str, list[int]],
) -> str:
    if not assignments:
        raise RuntimeError("empty assignments for threshold selection")

    rows: list[tuple[str, dict[str, Any]]] = []
    for tk, labels in assignments.items():
        rows.append((tk, _cluster_stats(labels)))

    def key(row: tuple[str, dict[str, Any]]) -> tuple[float, float, int]:
        _, st = row
        return (
            float(st.get("non_singleton_task_coverage") or 0.0),
            -float(st.get("singleton_task_ratio") or 0.0),
            -int(st.get("cluster_count") or 0),
        )

    best_tk, _ = max(rows, key=key)
    return best_tk


def _sample_review_pairs(
    *,
    items: list[TaskPoolItem],
    labels_task: list[int],
    labels_skill: list[int],
    per_task_delta: dict[str, float],
    task_space_sim: list[list[float]],
    skill_space_sim: list[list[float]],
    sample_per_category: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    index = {i.task_uid: idx for idx, i in enumerate(items)}
    pairs: list[tuple[int, int]] = []
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))

    cats: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": [], "D": []}

    for i, j in pairs:
        ti = items[i]
        tj = items[j]
        t_same = labels_task[i] == labels_task[j]
        s_same = labels_skill[i] == labels_skill[j]
        di = per_task_delta.get(ti.task_uid)
        dj = per_task_delta.get(tj.task_uid)
        d_pair = None
        if di is not None and dj is not None:
            d_pair = float((di + dj) / 2.0)

        row = {
            "task_uid_a": ti.task_uid,
            "task_name_a": ti.task_name,
            "benchmark_a": ti.benchmark_slug,
            "task_uid_b": tj.task_uid,
            "task_name_b": tj.task_name,
            "benchmark_b": tj.benchmark_slug,
            "task_same_cluster": bool(t_same),
            "skill_same_cluster": bool(s_same),
            "delta_success_pair_mean": d_pair,
            "distance_task_space": float(1.0 - float(task_space_sim[i][j])),
            "distance_skill_space": float(1.0 - float(skill_space_sim[i][j])),
            "evidence": {
                "instruction_a": str(ti.instruction_path),
                "instruction_b": str(tj.instruction_path),
                "skill_a": str(ti.skill_md_path),
                "skill_b": str(tj.skill_md_path),
            },
        }

        # A: task同簇/skill异簇 且迁移差（d_pair<=0）
        if t_same and (not s_same) and (d_pair is not None) and d_pair <= 0.0:
            cats["A"].append(row)
        # B: task异簇/skill同簇 且迁移好（d_pair>0）
        elif (not t_same) and s_same and (d_pair is not None) and d_pair > 0.0:
            cats["B"].append(row)
        elif t_same and s_same:
            cats["C"].append(row)
        elif (not t_same) and (not s_same):
            cats["D"].append(row)

    rng = random.Random(int(seed))
    sampled: dict[str, list[dict[str, Any]]] = {}
    for c, rows in cats.items():
        rows_sorted = sorted(rows, key=lambda x: (x["task_uid_a"], x["task_uid_b"]))
        if len(rows_sorted) <= max(0, int(sample_per_category)):
            sampled[c] = rows_sorted
        else:
            sampled[c] = rng.sample(rows_sorted, k=max(0, int(sample_per_category)))
    return sampled


def _render_review_markdown(samples: dict[str, list[dict[str, Any]]]) -> str:
    lines: list[str] = []
    lines.append("# Transferability Clustering Review Pack")
    lines.append("")
    lines.append("Categories:")
    lines.append("- A: task同簇/skill异簇 且迁移差")
    lines.append("- B: task异簇/skill同簇 且迁移好")
    lines.append("- C: 双同簇")
    lines.append("- D: 双异簇")
    lines.append("")

    for cat in ["A", "B", "C", "D"]:
        rows = samples.get(cat) or []
        lines.append(f"## Category {cat} ({len(rows)})")
        lines.append("")
        for idx, row in enumerate(rows, start=1):
            lines.append(f"### {cat}-{idx}")
            lines.append(f"- Task A: `{row['task_uid_a']}`")
            lines.append(f"- Task B: `{row['task_uid_b']}`")
            lines.append(f"- task_same_cluster: `{row['task_same_cluster']}`")
            lines.append(f"- skill_same_cluster: `{row['skill_same_cluster']}`")
            lines.append(f"- delta_success_pair_mean: `{row['delta_success_pair_mean']}`")
            ev = row.get("evidence") or {}
            lines.append(f"- instruction_a: `{ev.get('instruction_a')}`")
            lines.append(f"- instruction_b: `{ev.get('instruction_b')}`")
            lines.append(f"- skill_a: `{ev.get('skill_a')}`")
            lines.append(f"- skill_b: `{ev.get('skill_b')}`")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def _run_cross_space(args: argparse.Namespace) -> int:
    if int(args.samples_per_seed) <= 0:
        raise RuntimeError("--samples-per-seed must be > 0")

    thresholds = _parse_thresholds(args.cluster_thresholds)
    if not thresholds:
        raise RuntimeError("no threshold provided; set --cluster-thresholds")

    seeds = _parse_csv_ints(args.bootstrap_seeds)
    if not seeds:
        seeds = [13, 17, 23]

    transfer_k_values = _parse_csv_ints(args.transfer_k_values)
    if not transfer_k_values:
        transfer_k_values = [1, 3, 5]

    trace_root = args.trace_root.resolve()
    benchmark_config = args.benchmark_config.resolve()
    task_source_root = args.task_source_root.resolve()
    skillsbench_task_root = args.skillsbench_task_root.resolve()
    skills_root = args.golden_skills_root.resolve()

    benchmark_map = _load_benchmark_map(benchmark_config)
    records = _load_traces(trace_root)
    pool_items, missing_inputs = _build_task_pool(
        trace_records=records,
        benchmark_map=benchmark_map,
        skills_root=skills_root,
        task_source_root=task_source_root,
        skillsbench_task_root=skillsbench_task_root,
        benchmark_scope=str(args.benchmark_scope),
        m_success=int(args.m_success),
        n_failure=int(args.n_failure),
        text_char_limit=int(args.text_char_limit),
        restrict_to_skill_tasks=bool(args.restrict_task_pool_to_golden_skills),
    )

    if len(pool_items) < 2:
        raise RuntimeError(f"eligible task pool <2 after filtering; got {len(pool_items)}")

    task_filter = {x.strip() for x in (args.task_name or []) if x.strip()}
    if task_filter:
        pool_items = [x for x in pool_items if x.task_name in task_filter]

    if len(pool_items) < 2:
        raise RuntimeError(f"eligible task pool <2 after --task-name filter; got {len(pool_items)}")

    target_scope = str(args.target_benchmark_scope or args.benchmark_scope)
    eval_items = [
        x for x in pool_items if _match_scope(x.benchmark_id, x.benchmark_slug, target_scope)
    ]
    if len(eval_items) < 1:
        raise RuntimeError(f"eligible eval target pool <1 after --target-benchmark-scope={target_scope}; got {len(eval_items)}")

    run_id = str(args.run_id or f"cross-space-{int(time.time())}")
    run_id_slug = _normalize_slug(run_id, default="cross-space", max_len=96)
    args.run_id = run_id_slug  # keep deterministic for nested calls

    run_root = args.output_root.resolve() / run_id_slug
    run_root.mkdir(parents=True, exist_ok=True)

    task_space_texts = [
        _clean_text(f"Task: {it.task_name} | Benchmark: {it.benchmark_slug} | Instruction: {it.instruction_text}", char_limit=max(1000, int(args.text_char_limit)))
        for it in pool_items
    ]
    skill_space_texts = [
        _clean_text(_strip_frontmatter(it.skill_text), char_limit=max(1000, int(args.text_char_limit)))
        for it in pool_items
    ]

    task_embeddings = _load_qwen_embeddings(
        task_space_texts,
        model_name=str(args.embedding_model),
        device=str(args.embedding_device),
        batch_size=max(1, int(args.embedding_batch_size)),
        max_length=max(32, int(args.embedding_max_length)),
    )
    skill_embeddings = _load_qwen_embeddings(
        skill_space_texts,
        model_name=str(args.embedding_model),
        device=str(args.embedding_device),
        batch_size=max(1, int(args.embedding_batch_size)),
        max_length=max(32, int(args.embedding_max_length)),
    )

    if len(task_embeddings) != len(pool_items) or len(skill_embeddings) != len(pool_items):
        raise RuntimeError("embedding count mismatch in cross-space mode")

    task_assignments = _cluster_assignments_threshold_sweep(
        vectors=task_embeddings,
        thresholds=thresholds,
        linkage_method=str(args.cluster_linkage),
    )
    skill_assignments = _cluster_assignments_threshold_sweep(
        vectors=skill_embeddings,
        thresholds=thresholds,
        linkage_method=str(args.cluster_linkage),
    )

    _json_dump(
        run_root / "missing_inputs.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(missing_inputs),
            "rows": missing_inputs,
        },
    )

    _json_dump(
        run_root / "task_pool.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "task_count": len(pool_items),
            "eval_target_count": len(eval_items),
            "eval_target_scope": target_scope,
            "task_pool_hash": _task_pool_hash(pool_items),
            "rows": [
                {
                    "task_uid": x.task_uid,
                    "benchmark_id": x.benchmark_id,
                    "benchmark_slug": x.benchmark_slug,
                    "task_name": x.task_name,
                    "task_slug": x.task_slug,
                    "success_count": x.success_count,
                    "failure_count": x.failure_count,
                    "instruction_path": str(x.instruction_path),
                    "skill_md_path": str(x.skill_md_path),
                }
                for x in pool_items
            ],
        },
    )

    task_sim = _pairwise_similarity_matrix(task_embeddings)
    skill_sim = _pairwise_similarity_matrix(skill_embeddings)
    _json_dump(run_root / "task_space_similarity_matrix.json", {"uids": [x.task_uid for x in pool_items], "matrix": task_sim})
    _json_dump(run_root / "skill_space_similarity_matrix.json", {"uids": [x.task_uid for x in pool_items], "matrix": skill_sim})
    _json_dump(
        run_root / "task_space_assignments.json",
        {
            "uids": [x.task_uid for x in pool_items],
            "threshold_assignments": {
                tk: {pool_items[i].task_uid: int(labels[i]) for i in range(len(pool_items))}
                for tk, labels in sorted(task_assignments.items())
            },
        },
    )
    _json_dump(
        run_root / "skill_space_assignments.json",
        {
            "uids": [x.task_uid for x in pool_items],
            "threshold_assignments": {
                sk: {pool_items[i].task_uid: int(labels[i]) for i in range(len(pool_items))}
                for sk, labels in sorted(skill_assignments.items())
            },
        },
    )

    # Threshold sweep summary per space
    sweep_task: list[dict[str, Any]] = []
    sweep_skill: list[dict[str, Any]] = []
    for tk, labels in sorted(task_assignments.items()):
        sweep_task.append({"threshold": tk, **_cluster_stats(labels)})
    for sk, labels in sorted(skill_assignments.items()):
        sweep_skill.append({"threshold": sk, **_cluster_stats(labels)})

    _json_dump(run_root / "task_space_cluster_sweep.json", {"rows": sweep_task})
    _json_dump(run_root / "skill_space_cluster_sweep.json", {"rows": sweep_skill})

    # Cross-partition grid
    partition_rows: list[dict[str, Any]] = []
    for tk, t_labels in sorted(task_assignments.items()):
        for sk, s_labels in sorted(skill_assignments.items()):
            m = _partition_metrics(t_labels, s_labels)
            partition_rows.append(
                {
                    "task_threshold": tk,
                    "skill_threshold": sk,
                    **m,
                }
            )
    _json_dump(run_root / "cross_space_partition_grid.json", {"rows": partition_rows})

    selected_pair = _choose_selected_threshold_pair(
        partition_rows,
        force_task_threshold=args.review_task_threshold,
        force_skill_threshold=args.review_skill_threshold,
    )
    if selected_pair is None:
        raise RuntimeError("failed to select threshold pair")

    sel_tk = str(selected_pair["task_threshold"])
    sel_sk = str(selected_pair["skill_threshold"])
    sel_labels_task = task_assignments[sel_tk]
    sel_labels_skill = skill_assignments[sel_sk]

    if str(args.cluster_source) == "selected_threshold":
        transfer_task_tk = sel_tk
        transfer_skill_sk = sel_sk
    elif str(args.cluster_source) == "best_per_space":
        transfer_task_tk = (
            f"{float(args.review_task_threshold):.8f}"
            if args.review_task_threshold is not None
            else _choose_best_space_threshold(task_assignments)
        )
        transfer_skill_sk = (
            f"{float(args.review_skill_threshold):.8f}"
            if args.review_skill_threshold is not None
            else _choose_best_space_threshold(skill_assignments)
        )
    else:
        raise RuntimeError(f"unsupported --cluster-source: {args.cluster_source}")

    # Stratified metrics + bootstrap CI on selected pair
    bench_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(pool_items):
        bench_to_indices[item.benchmark_slug].append(i)

    stratified: dict[str, Any] = {}
    for bench_slug, idxs in sorted(bench_to_indices.items(), key=lambda kv: kv[0]):
        if len(idxs) < 2:
            stratified[bench_slug] = {"task_count": len(idxs), "note": "<2 tasks, skipped"}
            continue
        lt = [sel_labels_task[i] for i in idxs]
        ls = [sel_labels_skill[i] for i in idxs]
        met = _partition_metrics(lt, ls)
        ci = _bootstrap_partition_metrics(
            labels_task=lt,
            labels_skill=ls,
            seeds=seeds,
            samples_per_seed=int(args.samples_per_seed),
        )
        stratified[bench_slug] = {
            "task_count": len(idxs),
            "metrics": met,
            "bootstrap": ci,
        }

    overall_metrics = _partition_metrics(sel_labels_task, sel_labels_skill)
    overall_ci = _bootstrap_partition_metrics(
        labels_task=sel_labels_task,
        labels_skill=sel_labels_skill,
        seeds=seeds,
        samples_per_seed=int(args.samples_per_seed),
    )

    _json_dump(
        run_root / "selected_partition_metrics.json",
        {
            "selected_threshold_pair": selected_pair,
            "overall": {"metrics": overall_metrics, "bootstrap": overall_ci},
            "by_benchmark": stratified,
        },
    )

    transfer_summary: dict[str, Any] = {
        "enabled": bool(args.run_transfer_eval),
        "cluster_source": str(args.cluster_source),
        "neighbor_source": str(args.neighbor_source),
        "task_space_threshold_for_transfer": transfer_task_tk,
        "skill_space_threshold_for_transfer": transfer_skill_sk,
        "spaces": {},
        "deltas": {},
    }
    per_task_delta_for_review: dict[str, float] = {}

    if args.run_transfer_eval:
        api_key_env = str(args.api_key_env or PROVIDER_DEFAULT_KEY_ENV[str(args.provider).lower()])
        condition = f"{max(0, int(args.m_success))}s{max(0, int(args.n_failure))}f"
        transfer_task_labels = task_assignments[transfer_task_tk]
        transfer_skill_labels = skill_assignments[transfer_skill_sk]

        # Build neighbors either globally or from same-cluster candidates.
        for k in transfer_k_values:
            k_eff = min(max(1, int(k)), len(pool_items) - 1)
            if str(args.neighbor_source) == "global":
                neighbors_task = _build_space_neighbors(pool_items, task_embeddings, k=k_eff)
                neighbors_skill = _build_space_neighbors(pool_items, skill_embeddings, k=k_eff)
                task_neighbor_stats = {
                    "mode": "global",
                    "k_requested": int(k),
                    "k_effective": int(k_eff),
                    "task_count": len(pool_items),
                    "coverage_ratio": 1.0,
                    "no_candidate_task_count": 0,
                    "insufficient_neighbor_task_count": 0,
                    "shortages": [],
                }
                skill_neighbor_stats = dict(task_neighbor_stats)
            else:
                neighbors_task, task_neighbor_stats = _build_intra_cluster_neighbors(
                    items=pool_items,
                    sim_matrix=task_sim,
                    labels=transfer_task_labels,
                    k=k_eff,
                )
                neighbors_skill, skill_neighbor_stats = _build_intra_cluster_neighbors(
                    items=pool_items,
                    sim_matrix=skill_sim,
                    labels=transfer_skill_labels,
                    k=k_eff,
                )

            _json_dump(
                run_root / "neighbors" / f"task_space_t{transfer_task_tk}_k{k_eff}.json",
                {
                    "k": k_eff,
                    "selected_threshold": transfer_task_tk,
                    "space": "task_space",
                    "neighbor_stats": task_neighbor_stats,
                    "rows": neighbors_task,
                },
            )
            _json_dump(
                run_root / "neighbors" / f"skill_space_t{transfer_skill_sk}_k{k_eff}.json",
                {
                    "k": k_eff,
                    "selected_threshold": transfer_skill_sk,
                    "space": "skill_space",
                    "neighbor_stats": skill_neighbor_stats,
                    "rows": neighbors_skill,
                },
            )

            task_eval = _run_transfer_eval_for_space(
                space_name="task_space",
                selected_threshold_key=transfer_task_tk,
                k=k_eff,
                neighbors_by_uid=neighbors_task,
                neighbor_stats=task_neighbor_stats,
                items=eval_items,
                run_root=run_root,
                condition=condition,
                args=args,
                api_key_env=api_key_env,
            )
            skill_eval = _run_transfer_eval_for_space(
                space_name="skill_space",
                selected_threshold_key=transfer_skill_sk,
                k=k_eff,
                neighbors_by_uid=neighbors_skill,
                neighbor_stats=skill_neighbor_stats,
                items=eval_items,
                run_root=run_root,
                condition=condition,
                args=args,
                api_key_env=api_key_env,
            )

            transfer_summary["spaces"][f"k{k_eff}"] = {
                "task_space": task_eval,
                "skill_space": skill_eval,
            }

            t_rates = task_eval.get("success_rate_by_task_uid") or {}
            s_rates = skill_eval.get("success_rate_by_task_uid") or {}
            deltas_rows: list[dict[str, Any]] = []
            deltas_by_bench: dict[str, list[float]] = defaultdict(list)
            for item in pool_items:
                tr = t_rates.get(item.task_uid)
                sr = s_rates.get(item.task_uid)
                if not isinstance(tr, (int, float)) or not isinstance(sr, (int, float)):
                    continue
                delta = float(sr) - float(tr)
                deltas_rows.append(
                    {
                        "task_uid": item.task_uid,
                        "task_name": item.task_name,
                        "benchmark_slug": item.benchmark_slug,
                        "task_neighbor_success_rate": float(tr),
                        "skill_neighbor_success_rate": float(sr),
                        "delta_success_rate": delta,
                    }
                )
                deltas_by_bench[item.benchmark_slug].append(delta)

            overall_delta = None
            if deltas_rows:
                overall_delta = float(sum(r["delta_success_rate"] for r in deltas_rows) / len(deltas_rows))

            per_bench_delta = {
                b: {
                    "task_count": len(vals),
                    "delta_mean": float(sum(vals) / len(vals)) if vals else None,
                }
                for b, vals in sorted(deltas_by_bench.items())
            }

            transfer_summary["deltas"][f"k{k_eff}"] = {
                "task_count": len(deltas_rows),
                "delta_mean": overall_delta,
                "by_benchmark": per_bench_delta,
                "rows": deltas_rows,
            }

            if int(k_eff) == int(transfer_k_values[0]):
                per_task_delta_for_review = {
                    row["task_uid"]: float(row["delta_success_rate"])
                    for row in deltas_rows
                }

        _json_dump(run_root / "transfer_alignment_summary.json", transfer_summary)

    # Review package on selected thresholds + first k delta (if available)
    samples = _sample_review_pairs(
        items=pool_items,
        labels_task=sel_labels_task,
        labels_skill=sel_labels_skill,
        per_task_delta=per_task_delta_for_review,
        task_space_sim=task_sim,
        skill_space_sim=skill_sim,
        sample_per_category=max(0, int(args.review_sample_per_category)),
        seed=int(args.analysis_seed),
    )

    review_rows: list[dict[str, Any]] = []
    for cat in ["A", "B", "C", "D"]:
        for row in samples.get(cat, []):
            review_rows.append({"category": cat, **row})

    review_jsonl = run_root / "human_review" / "review_pairs.jsonl"
    review_md = run_root / "human_review" / "review_pairs.md"
    _jsonl_dump(review_jsonl, review_rows)
    review_md.write_text(_render_review_markdown(samples), encoding="utf-8")

    # Manifest
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "run_skill_transferability_eval.py",
        "mode": "cross_space",
        "run_id": run_id_slug,
        "task_pool_hash": _task_pool_hash(pool_items),
        "task_count": len(pool_items),
        "benchmark_scope": str(args.benchmark_scope),
        "embedding_model": str(args.embedding_model),
        "embedding_device": str(args.embedding_device),
        "cluster_linkage": str(args.cluster_linkage),
        "cluster_thresholds": thresholds,
        "selected_task_threshold": sel_tk,
        "selected_skill_threshold": sel_sk,
        "cluster_source": str(args.cluster_source),
        "neighbor_source": str(args.neighbor_source),
        "restrict_task_pool_to_golden_skills": bool(args.restrict_task_pool_to_golden_skills),
        "target_benchmark_scope": target_scope,
        "transfer_task_threshold": transfer_task_tk,
        "transfer_skill_threshold": transfer_skill_sk,
        "bootstrap_seeds": seeds,
        "samples_per_seed": int(args.samples_per_seed),
        "analysis_seed": int(args.analysis_seed),
        "transfer_eval_enabled": bool(args.run_transfer_eval),
        "transfer_k_values": transfer_k_values,
        "provider": str(args.provider),
        "api_key_env": str(args.api_key_env or PROVIDER_DEFAULT_KEY_ENV[str(args.provider).lower()]),
        "base_url": args.base_url,
        "agent": str(args.agent),
        "model": str(args.model),
        "m_success": int(args.m_success),
        "n_failure": int(args.n_failure),
        "n_attempts": int(args.n_attempts),
        "n_concurrent": int(args.n_concurrent),
        "max_steps": int(args.max_steps),
        "command_timeout_sec": int(args.command_timeout_sec),
        "docker_cleanup": bool(args.docker_cleanup),
        "docker_cleanup_timeout_sec": int(args.docker_cleanup_timeout_sec),
        "docker_cleanup_strict": bool(args.docker_cleanup_strict),
        "dry_run": bool(args.dry_run),
    }
    _json_dump(run_root / "run_manifest.json", manifest)

    brief = {
        "mode": "cross_space",
        "run_root": str(run_root),
        "task_count": len(pool_items),
        "selected_threshold_pair": {"task": sel_tk, "skill": sel_sk},
        "transfer_eval_enabled": bool(args.run_transfer_eval),
        "review_sample_sizes": {k: len(v) for k, v in samples.items()},
    }
    print(json.dumps(brief, indent=2, ensure_ascii=False))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-space clustering runner for skill transferability analysis. "
            "Build task-space and skill-space clusters, compare partitions, and optionally run same-cluster top-k transfer eval."
        )
    )
    # shared inputs
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=CLEAN_ROOT / "configs" / "benchmarks.json",
    )
    parser.add_argument("--task-source-root", type=Path, default=Path(os.environ.get("TASK_SOURCE_ROOT", "/raid/zhiyuan/procmem2skills/benchmarks/harbor-datasets")))
    parser.add_argument("--skillsbench-task-root", type=Path, default=Path(os.environ.get("SKILLSBENCH_TASKS_ROOT", "/raid/zhiyuan/pm2s/skillsbench/tasks-no-skills")))
    parser.add_argument("--procmem2skills-root", type=Path, default=Path(os.environ.get("PROCMEM2SKILLS_ROOT", "/raid/zhiyuan/procmem2skills")))
    parser.add_argument("--pm2s-root", type=Path, default=CLEAN_ROOT)

    # input skill roots
    parser.add_argument(
        "--golden-skills-root",
        type=Path,
        default=CLEAN_ROOT / "retrieval-data" / "golden-skills",
        help="Root containing benchmark subdirs and SKILL.md files.",
    )

    parser.add_argument("--output-root", type=Path, required=True)

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

    parser.add_argument("--task-name", action="append", default=[])

    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-max-length", type=int, default=2048)
    parser.add_argument("--text-char-limit", type=int, default=12000)

    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cluster-source", choices=["selected_threshold", "best_per_space"], default="selected_threshold")
    parser.add_argument("--neighbor-source", choices=["global", "same_cluster"], default="same_cluster")
    parser.add_argument("--target-benchmark-scope", choices=["all", "tbpro", "tb2", "sb"], default=None)
    parser.add_argument("--restrict-task-pool-to-golden-skills", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark-scope", choices=["all", "tbpro", "tb2", "sb"], default="all")
    parser.add_argument("--cluster-linkage", choices=["average", "complete", "single"], default="average")
    parser.add_argument("--cluster-thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--review-task-threshold", type=float, default=None)
    parser.add_argument("--review-skill-threshold", type=float, default=None)
    parser.add_argument("--bootstrap-seeds", default="13,17,23")
    parser.add_argument("--samples-per-seed", type=int, default=400)
    parser.add_argument("--analysis-seed", type=int, default=42)
    parser.add_argument("--review-sample-per-category", type=int, default=30)

    parser.add_argument("--run-transfer-eval", action="store_true")
    parser.add_argument("--transfer-k-values", default="1,3,5")

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return _run_cross_space(args)


if __name__ == "__main__":
    raise SystemExit(main())
