#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
from typing import Any

PROVIDER_DEFAULT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GOOGLE_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "uniapi": "UNIAPI_API_KEY",
}

COMPACT_PROCEDURE_ARMS = {"short-plan", "test-first", "script"}


def _normalize_slug(value: str, *, default: str = "value", max_len: int = 80) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
    if not tokens:
        return default
    slug = "-".join(tokens)
    if len(slug) <= max_len:
        return slug
    trimmed = slug[:max_len].rstrip("-")
    return trimmed or default


def _dataset_storage_slug(dataset: str) -> str:
    text = str(dataset).strip().lower()
    if "@" not in text:
        return _normalize_slug(text, default="dataset", max_len=48)
    name, version = text.split("@", 1)
    if version.strip():
        return _normalize_slug(f"{name.strip()}-{version.strip()}", default="dataset", max_len=48)
    return _normalize_slug(name.strip(), default="dataset", max_len=48)


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



def _normalize_agent_name(agent: str) -> str:
    return _normalize_slug(str(agent or "").strip(), default="agent", max_len=48)


def _agent_skill_container_root(agent: str) -> str:
    """Return skill mount root inside task container for a given agent."""
    name = _normalize_agent_name(agent)
    if name in {"gemini-cli", "gemini"}:
        return "/root/.gemini/skills"
    if name == "codex":
        return "/root/.codex/skills"
    # Keep historical default for compatibility with existing runs.
    return "/root/.codex/skills"


def _agent_home_env_vars(agent: str) -> dict[str, str]:
    """Optional per-agent home env used by installed CLI agents."""
    name = _normalize_agent_name(agent)
    if name == "codex":
        return {"CODEX_HOME": "/root/.codex"}
    if name in {"gemini-cli", "gemini"}:
        return {
            "GEMINI_CLI_HOME": "/root/.gemini",
            "GEMINI_CLI_TRUST_WORKSPACE": "true",
        }
    return {}

SENSITIVE_ENV_KEY_RE = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL|AUTH)", re.IGNORECASE)


def _is_sensitive_env_key(key: str) -> bool:
    return bool(SENSITIVE_ENV_KEY_RE.search(str(key or "")))


def _redact_agent_env(env: dict[str, str] | None) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in (env or {}).items():
        redacted[key] = "[REDACTED]" if _is_sensitive_env_key(key) else str(value)
    return redacted


def _redact_command_for_manifest(cmd: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for part in cmd:
        text = str(part)
        if redact_next:
            if "=" in text:
                key, _, _value = text.partition("=")
                redacted.append(f"{key}=[REDACTED]" if _is_sensitive_env_key(key) else text)
            else:
                redacted.append("[REDACTED]")
            redact_next = False
            continue
        redacted.append(text)
        if text in {"--ae", "--api-key"}:
            redact_next = True
    return redacted


def _redact_sensitive_text(text: str, env: dict[str, str] | None) -> str:
    redacted = str(text or "")
    for key, value in (env or {}).items():
        if not _is_sensitive_env_key(key):
            continue
        value_text = str(value or "")
        if len(value_text) >= 8:
            redacted = redacted.replace(value_text, "[REDACTED]")
    return redacted


def _redact_sensitive_files(root: Path, env: dict[str, str] | None, *, max_bytes: int = 20_000_000) -> int:
    if not root.exists():
        return 0
    changed = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        redacted = _redact_sensitive_text(text, env)
        redacted = re.sub(r"sk-ant-oat01--[^\"'\s,}]+", "[REDACTED]", redacted)
        if path.name == ".credentials.json":
            try:
                def _scrub_credential_json(value: Any) -> Any:
                    if isinstance(value, dict):
                        return {k: _scrub_credential_json(v) for k, v in value.items()}
                    if isinstance(value, list):
                        return [_scrub_credential_json(v) for v in value]
                    if isinstance(value, str) and len(value) >= 8:
                        return "[REDACTED]"
                    return value

                redacted = json.dumps(_scrub_credential_json(json.loads(redacted)), indent=2) + "\n"
            except Exception:
                pass
        if redacted != text:
            path.write_text(redacted)
            changed += 1
    return changed


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _strip_openai_model_prefix(model: str) -> str:
    text = str(model or "").strip()
    if text.lower().startswith("openai/"):
        return text.split("/", 1)[1].strip() or text
    return text


def _cleanup_docker(*, enabled: bool, timeout_sec: int, strict: bool) -> dict[str, Any]:
    timeout = max(1, int(timeout_sec))
    result: dict[str, Any] = {
        "enabled": bool(enabled),
        "strict": bool(strict),
        "timeout_sec": timeout,
        "status": "skipped_disabled",
        "had_error": False,
        "steps": [],
    }
    if not enabled:
        return result

    def _run_cleanup_step(cmd: list[str], *, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        step: dict[str, Any] = {"cmd": cmd}
        if meta:
            step.update(meta)
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            step["return_code"] = int(completed.returncode)
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            step["stdout_tail"] = stdout[-800:]
            step["stderr_tail"] = stderr[-800:]
            step["stdout"] = stdout
            step["stderr"] = stderr
            step["status"] = "ok" if completed.returncode == 0 else "failed"
        except FileNotFoundError:
            step["status"] = "skipped_no_docker"
        except subprocess.TimeoutExpired:
            step["status"] = "timeout"
        except Exception as exc:
            step["status"] = "exception"
            step["exception"] = f"{type(exc).__name__}: {exc}"
        return step

    statuses: list[str] = []

    list_containers_step = _run_cleanup_step(["docker", "ps", "-aq", "--filter", "name=__"])
    result["steps"].append(list_containers_step)
    statuses.append(str(list_containers_step.get("status")))
    if list_containers_step.get("status") == "skipped_no_docker":
        result["status"] = "skipped_no_docker"
        return result

    if list_containers_step.get("status") == "ok":
        container_ids = [line.strip() for line in str(list_containers_step.get("stdout") or "").splitlines() if line.strip()]
        if container_ids:
            result["matched_container_count"] = len(container_ids)
            for start_idx in range(0, len(container_ids), 80):
                chunk = container_ids[start_idx : start_idx + 80]
                rm_step = _run_cleanup_step(
                    ["docker", "rm", "-f", *chunk],
                    meta={"batch_size": len(chunk), "kind": "remove_containers"},
                )
                result["steps"].append(rm_step)
                statuses.append(str(rm_step.get("status")))

    list_networks_step = _run_cleanup_step(["docker", "network", "ls", "--format", "{{.Name}}"])
    result["steps"].append(list_networks_step)
    statuses.append(str(list_networks_step.get("status")))
    if list_networks_step.get("status") == "ok":
        network_names: list[str] = []
        for line in str(list_networks_step.get("stdout") or "").splitlines():
            name = line.strip()
            if not name:
                continue
            if re.search(r"(__|harbor|terminal-bench|pm2s)", name):
                network_names.append(name)
        if network_names:
            result["matched_network_count"] = len(network_names)
            for network_name in network_names:
                rm_network_step = _run_cleanup_step(
                    ["docker", "network", "rm", network_name],
                    meta={"network": network_name, "kind": "remove_network"},
                )
                result["steps"].append(rm_network_step)
                statuses.append(str(rm_network_step.get("status")))

    # Do not prune globally: this runner may share a Docker daemon with other
    # experiments. Only explicitly matched Harbor/PM2S resources above may be
    # removed.

    if statuses and all(st == "skipped_no_docker" for st in statuses):
        result["status"] = "skipped_no_docker"
        return result

    had_error = any(st in {"failed", "timeout", "exception"} for st in statuses)
    result["had_error"] = had_error
    if had_error:
        result["status"] = "failed" if strict else "warning"
    else:
        result["status"] = "ok"

    for step in result.get("steps", []):
        if isinstance(step, dict):
            step.pop("stdout", None)
            step.pop("stderr", None)
    return result
def _load_traces(trace_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(trace_root.rglob("*.json")):
        if path.name.endswith("summary.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("task_name") and payload.get("trial_name"):
            rows.append(payload)
    return rows


def _load_benchmark_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("benchmarks") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError(f"invalid benchmark config: {path}")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        bid = str(row.get("id") or "").strip()
        if not bid:
            continue
        out[bid] = row

        dataset = str(row.get("dataset") or "").strip()
        aliases = {bid}
        if dataset:
            aliases.add(dataset)
            aliases.add(dataset.split("@", 1)[0])
        aliases.add(_benchmark_output_slug(benchmark_id=bid, benchmark_cfg=row))

        if bid == "terminal-bench-pro-1" or dataset.startswith("terminal-bench-pro@"):
            aliases.update({"terminal-bench-pro", "terminal-bench-pro-1.0"})
        if bid == "terminal-bench-2" or dataset.startswith("terminal-bench@2"):
            aliases.update({"terminal-bench", "terminal-bench-2.0"})

        for alias in aliases:
            alias = str(alias or "").strip()
            if alias and alias not in out:
                out[alias] = row
    return out


def _group_traces_by_task(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = {}
    for rec in records:
        task = str(rec.get("task_name") or "").strip()
        if not task:
            continue
        benchmark = str(rec.get("benchmark") or "unknown-benchmark")
        status = str(rec.get("status") or "runtime_error").strip().lower()
        bucket = by_task.setdefault(task, {"benchmark": benchmark, "success": [], "failure": []})
        if bucket.get("benchmark") in {"", "unknown-benchmark"}:
            bucket["benchmark"] = benchmark
        if status == "success":
            bucket["success"].append(rec)
        elif status == "failure":
            bucket["failure"].append(rec)
    return by_task


def _load_skill_for_condition(skills_root: Path, condition: str, task_name: str) -> Path | None:
    task_slug = _normalize_slug(task_name, default="task")
    p = skills_root / condition / task_slug / "SKILL.md"
    return p if p.is_file() else None


def _load_compact_procedure_for_condition(
    compact_procedure_root: Path | None,
    condition: str,
    form: str,
    task_name: str,
) -> Path | None:
    if compact_procedure_root is None:
        return None
    form_slug = _normalize_slug(form, default="procedure")
    task_slug = _normalize_slug(task_name, default="task")
    candidates = [
        compact_procedure_root / condition / form_slug / task_slug / "procedure.md",
        compact_procedure_root / form_slug / condition / task_slug / "procedure.md",
        compact_procedure_root / condition / form_slug / task_slug / "PROCEDURE.md",
        compact_procedure_root / form_slug / condition / task_slug / "PROCEDURE.md",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _load_transfer_skill_pool_manifest(path: Path | None) -> dict[str, list[dict[str, str]]]:
    if path is None:
        return {}
    if not path.is_file():
        raise RuntimeError(f"missing skills manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"invalid skills manifest rows in {path}")
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_name = str(row.get("task_name") or "").strip()
        neighbors = row.get("neighbors") or []
        if not task_name or not isinstance(neighbors, list):
            continue
        cleaned: list[dict[str, str]] = []
        for ref in neighbors:
            if not isinstance(ref, dict):
                continue
            skill_md = str(ref.get("source_skill_md") or "").strip()
            skill_name = str(ref.get("neighbor_task_name") or "").strip()
            if not skill_md:
                continue
            cleaned.append({"skill_md": skill_md, "skill_name": skill_name})
        if cleaned:
            out[task_name] = cleaned
    return out


def _resolve_stable_harbor_runtime(*, procmem2skills_root: Path, env: dict[str, str]) -> dict[str, str]:
    """Pin runner to the known-good Harbor runtime under /raid/zhiyuan/procmem2skills.

    Resolution order:
    1) Explicit root passed by caller (if it has .venv-py312)
    2) Stable absolute fallback /raid/zhiyuan/procmem2skills/.venv-py312
    3) Leave env unchanged if neither exists
    """
    out = dict(env)

    candidates = [
        procmem2skills_root,
        Path('/raid/zhiyuan/procmem2skills'),
    ]
    for root in candidates:
        harbor_bin = root / '.venv-py312' / 'bin' / 'harbor'
        python_bin = root / '.venv-py312' / 'bin' / 'python'
        if harbor_bin.is_file() and python_bin.is_file():
            out['HARBOR_BIN'] = str(harbor_bin)
            out['PROCMEM_BENCHMARK_PYTHON'] = str(python_bin)
            out.setdefault('PM2S_RUNTIME_ROOT', str(root))
            return out

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
        # gemini-cli requires GEMINI_API_KEY for API-key auth.
        if p == "google":
            env["GEMINI_API_KEY"] = key
        return env
    raise RuntimeError(f"unsupported provider: {provider}")


def _extract_codelog_steps(codex_txt_path: str, *, max_steps: int = 8) -> list[dict[str, str]]:
    path = Path(str(codex_txt_path or "").strip())
    if not path.is_file():
        return []

    out: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("type") != "item.completed":
                    continue
                item = row.get("item")
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "command_execution":
                    continue
                cmd = _compact_for_memory(item.get("command"), limit=220)
                result = _compact_for_memory(item.get("aggregated_output"), limit=260)
                if not cmd:
                    continue
                out.append({"command": cmd, "result": result})
                if len(out) >= max_steps:
                    break
    except Exception:
        return []
    return out


def _normalize_trace_steps(rec: dict[str, Any], *, max_steps: int = 8) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for step in list(rec.get("steps") or []):
        if not isinstance(step, dict):
            continue
        cmd = _compact_for_memory(step.get("command"), limit=220)
        res = _compact_for_memory(step.get("result"), limit=260)
        if cmd or res:
            normalized.append({"command": cmd, "result": res})
        if len(normalized) >= max_steps:
            break

    if normalized:
        return normalized

    codex_txt = rec.get("codex_txt")
    if codex_txt:
        parsed = _extract_codelog_steps(str(codex_txt), max_steps=max_steps)
        if parsed:
            return parsed

    return []


def _workflow_block(task_name: str, traces: list[dict[str, Any]], *, hint_mode: str = "with-status") -> str:
    include_status = str(hint_mode).strip().lower() == "with-status"
    lines = [
        "<!-- PM2S_WORKFLOW_CONTEXT_BEGIN -->",
        "## Workflow Context",
        f"Task: {task_name}",
        "You are given one or more execution traces from an agent completing this task.",
        "Use these historical command/result traces as hints, adapt to current state.",
        "",
    ]
    for idx, rec in enumerate((traces or [])[:3], start=1):
        if include_status:
            lines.append(f"### Trace {idx} ({rec.get('status')})")
        else:
            lines.append(f"### Trace {idx}")
        normalized_steps = _normalize_trace_steps(rec, max_steps=8)
        if normalized_steps:
            for step in normalized_steps:
                if not isinstance(step, dict):
                    continue
                cmd = _compact_for_memory(step.get("command"), limit=180)
                res = _compact_for_memory(step.get("result"), limit=180)
                if cmd:
                    lines.append(f"- command: {cmd}")
                if res:
                    lines.append(f"  result: {res}")
        else:
            lines.append("- command: (no extracted command evidence in this trace)")
        lines.append("")
    lines.append("<!-- PM2S_WORKFLOW_CONTEXT_END -->")
    return "\n".join(lines) + "\n"


def _candidate_task_source_roots(
    *,
    base_root: Path,
    benchmark_cfg: dict[str, Any] | None,
    procmem2skills_root: Path,
) -> list[Path]:
    roots: list[Path] = []

    def _append(path: Path) -> None:
        try:
            rp = path.resolve(strict=False)
        except Exception:
            rp = path
        if rp in roots:
            return
        roots.append(rp)

    _append(base_root)

    cfg = benchmark_cfg or {}
    dataset = str(cfg.get("dataset") or "").strip().lower()
    runner = str(cfg.get("runner") or "").strip().lower()
    is_skillsbench = ("skillsbench" in dataset) or ("skills-bench" in dataset) or (runner == "skillsbench")

    if is_skillsbench:
        _append(procmem2skills_root / "benchmarks" / "skillsbench" / "tasks")
        _append(procmem2skills_root / "benchmarks" / "skillsbench")
        _append(Path("/raid/zhiyuan/pm2s/skillsbench/tasks"))
        _append(Path("/raid/zhiyuan/pm2s/skillsbench"))
    else:
        _append(procmem2skills_root / "benchmarks" / "harbor-datasets")

    return [r for r in roots if r.exists()]


def _find_task_source(src_roots: list[Path], task_name: str) -> Path | None:
    for src_root in src_roots:
        direct = src_root / task_name
        if direct.is_dir():
            return direct

        nested_tasks = src_root / "tasks" / task_name
        if nested_tasks.is_dir():
            return nested_tasks

        candidates = sorted(src_root.glob(f"*/{task_name}"))
        if candidates:
            return candidates[0]

    return None


def _copy_task_source(src_roots: list[Path], task_name: str, dst_root: Path) -> Path:
    src = _find_task_source(src_roots, task_name)
    if src is None:
        roots_text = ", ".join(str(p) for p in src_roots)
        raise FileNotFoundError(f"task source not found for {task_name} under roots=[{roots_text}]")
    dst = dst_root / task_name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def _inject_workflow_context(task_dir: Path, block: str) -> None:
    instruction = task_dir / "instruction.md"
    if not instruction.is_file():
        return
    text = instruction.read_text(encoding="utf-8")
    if "<!-- PM2S_WORKFLOW_CONTEXT_BEGIN -->" in text and "<!-- PM2S_WORKFLOW_CONTEXT_END -->" in text:
        text = text.split("<!-- PM2S_WORKFLOW_CONTEXT_BEGIN -->", 1)[0].rstrip() + "\n"
    instruction.write_text(text.rstrip() + "\n\n" + block, encoding="utf-8")


def _compact_procedure_title(form: str) -> str:
    form_slug = _normalize_slug(form, default="procedure")
    if form_slug == "short-plan":
        return "Short Plan Procedure"
    if form_slug == "test-first":
        return "Test-First Procedure"
    if form_slug == "script":
        return "Script-Style Command Recipe"
    return form_slug.replace("-", " ").title()


def _inject_compact_procedure_context(task_dir: Path, form: str, procedure_path: Path) -> None:
    instruction = task_dir / "instruction.md"
    if not instruction.is_file():
        return
    procedure = procedure_path.read_text(encoding="utf-8").strip()
    if not procedure:
        return

    begin = "<!-- PM2S_COMPACT_PROCEDURE_BEGIN -->"
    end = "<!-- PM2S_COMPACT_PROCEDURE_END -->"
    text = instruction.read_text(encoding="utf-8")
    if begin in text and end in text:
        text = text.split(begin, 1)[0].rstrip() + "\n"

    form_slug = _normalize_slug(form, default="procedure")
    block = (
        f"{begin}\n"
        f"## {_compact_procedure_title(form_slug)}\n"
        "The following task-specific procedure was distilled from prior execution workflows. "
        "Use it as guidance, adapt it to the current task state, and do not assume literal commands or paths are already valid.\n\n"
        f"{procedure}\n"
        f"{end}\n"
    )
    instruction.write_text(text.rstrip() + "\n\n" + block, encoding="utf-8")


def _compact_for_memory(value: Any, *, limit: int = 220) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 3)] + "..."


def _workflow_steps_from_trace_steps(step_rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    order = 1
    for raw in step_rows:
        if not isinstance(raw, dict):
            continue
        step_type = str(raw.get("step_type") or "").strip().lower()

        if step_type == "command_execution":
            command = _compact_for_memory(raw.get("command"), limit=260)
            output = _compact_for_memory(raw.get("output"), limit=220)
            exit_code = raw.get("exit_code")
            verification_parts: list[str] = []
            if exit_code is not None:
                verification_parts.append(f"exit_code={exit_code}")
            if output:
                verification_parts.append(f"output={output}")
            out.append(
                {
                    "order": order,
                    "intent": "Execute command and inspect result.",
                    "tool": "shell",
                    "operation": command or "execute_command",
                    "preconditions": [],
                    "verification": "; ".join(verification_parts) if verification_parts else None,
                }
            )
            order += 1
            continue

        if step_type == "agent_message":
            message = _compact_for_memory(raw.get("agent_message"), limit=220)
            if not message:
                continue
            out.append(
                {
                    "order": order,
                    "intent": message,
                    "tool": "assistant",
                    "operation": "agent_message",
                    "preconditions": [],
                    "verification": None,
                }
            )
            order += 1
            continue

        generic = _compact_for_memory(str(raw), limit=220)
        if not generic:
            continue
        out.append(
            {
                "order": order,
                "intent": "Inspect intermediate state.",
                "tool": "unknown",
                "operation": generic,
                "preconditions": [],
                "verification": None,
            }
        )
        order += 1

    if out:
        return out

    return [
        {
            "order": 1,
            "intent": "No structured steps available; rely on current observation.",
            "tool": "unknown",
            "operation": "no_op",
            "preconditions": [],
            "verification": None,
        }
    ]


def _workflow_candidate_from_trace(
    task_name: str,
    rec: dict[str, Any],
    attempt_index: int,
    *,
    hint_mode: str = "with-status",
) -> dict[str, Any]:
    include_status = str(hint_mode).strip().lower() == "with-status"
    trial_name = str(rec.get("trial_name") or "").strip()
    status = str(rec.get("status") or "unknown").strip().lower() or "unknown"
    if status not in {"success", "failure"}:
        status = "unknown"

    raw_steps = _normalize_trace_steps(rec, max_steps=12)
    if raw_steps:
        converted_steps = [
            {
                "step_type": "command_execution",
                "command": s.get("command", ""),
                "output": s.get("result", ""),
                "exit_code": None,
            }
            for s in raw_steps
            if isinstance(s, dict)
        ]
    else:
        converted_steps = list(rec.get("steps") or [])

    steps = _workflow_steps_from_trace_steps(converted_steps)
    objective = _compact_for_memory(task_name, limit=120)
    objective = f"Solve task {objective}" if objective else "Solve task"

    verification: list[str] = []
    reward = rec.get("reward")
    if include_status:
        if isinstance(reward, (int, float)):
            verification.append(f"reward={float(reward):.3f}")
        verification.append(f"status={status}")

    failure_modes: list[str] = []
    exception_type = _compact_for_memory(rec.get("exception_type"), limit=120)
    if include_status and status == "failure" and exception_type:
        failure_modes.append(f"exception_type={exception_type}")

    import json
    fingerprint_payload = {
        "task_name": task_name,
        "trial_name": trial_name,
        "status": status if include_status else None,
        "steps": steps,
    }
    fingerprint = hashlib.sha1(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    task_slug = _normalize_slug(task_name, default="task", max_len=40)
    trial_slug = _normalize_slug(trial_name, default=f"attempt-{attempt_index}", max_len=48)
    workflow_id = f"{task_slug}-{trial_slug}-wf"

    return {
        "workflow_id": workflow_id,
        "source_segment_id": trial_name or f"{task_slug}-attempt-{attempt_index}",
        "objective": objective,
        "trigger": f"When solving task {task_name}.",
        "preconditions": [],
        "steps": steps,
        "verification": verification,
        "failure_modes": failure_modes,
        "fingerprint": fingerprint,
        "metadata": {
            "source": "pm2s_run_context_comparison",
            "trial_name": trial_name,
            "status": status if include_status else None,
            "reward": reward if include_status else None,
            "exception_type": rec.get("exception_type") if include_status else None,
            "hint_mode": "with-status" if include_status else "no-hint",
        },
    }


def _workflow_attempts_from_traces(
    task_name: str,
    traces: list[dict[str, Any]],
    *,
    hint_mode: str = "with-status",
) -> list[dict[str, Any]]:
    include_status = str(hint_mode).strip().lower() == "with-status"
    attempts: list[dict[str, Any]] = []
    for idx, rec in enumerate(traces, start=1):
        if not isinstance(rec, dict):
            continue
        trial_name = str(rec.get("trial_name") or "").strip()
        status = str(rec.get("status") or "unknown").strip().lower() or "unknown"
        if status not in {"success", "failure"}:
            status = "unknown"
        attempt_payload = {
            "attempt_index": idx,
            "episode_id": trial_name or f"{task_name}__attempt_{idx}",
            "task_id": task_name,
            "workflows": [_workflow_candidate_from_trace(task_name, rec, idx, hint_mode=hint_mode)],
        }
        if include_status:
            attempt_payload["status"] = status
        attempts.append(attempt_payload)
    return attempts


def _remove_skill_instruction_hint(task_dir: Path) -> None:
    instruction = task_dir / "instruction.md"
    if not instruction.is_file():
        return

    begin = "<!-- PM2S_SKILL_HINT_BEGIN -->"
    end = "<!-- PM2S_SKILL_HINT_END -->"
    text = instruction.read_text(encoding="utf-8")
    if begin in text and end in text:
        text = text.split(begin, 1)[0].rstrip() + "\n"
        instruction.write_text(text, encoding="utf-8")


def _gemini_settings_dockerfile_snippet(container_skill_root: str) -> str:
    root = str(container_skill_root).rstrip("/")
    settings_json = json.dumps({"includeDirectories": [root]}, separators=(",", ":"))
    return (
        "\n# Allow Gemini CLI file tools to inspect injected skills without task-level hints\n"
        f"RUN mkdir -p /root/.gemini && printf '%s\\n' {shlex.quote(settings_json)} > /root/.gemini/settings.json\n"
    )


def _ensure_dockerfile_copies_skills(
    task_dir: Path,
    *,
    container_skill_root: str,
    agent: str,
) -> None:
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return
    text = dockerfile.read_text(encoding="utf-8")
    normalized = "\n".join(line.strip().lower() for line in text.splitlines())
    root_norm = str(container_skill_root).rstrip("/").lower()
    has_skill_copy = (
        f"copy skills {root_norm}" in normalized
        or f"copy skills/ {root_norm}/" in normalized
    )
    if not has_skill_copy:
        stripped = text.rstrip()
        if stripped.endswith("\\"):
            stripped = stripped[:-1].rstrip()
        text = (
            stripped
            + f"\n\n# Inject task-specific skill for {_normalize_agent_name(agent)}\nCOPY skills {str(container_skill_root).rstrip('/')}\n"
        )
    if _normalize_agent_name(agent) in {"gemini-cli", "gemini"} and "includeDirectories" not in text:
        text += _gemini_settings_dockerfile_snippet(container_skill_root)
    dockerfile.write_text(text, encoding="utf-8")


def _remove_environment_docker_image(task_dir: Path) -> None:
    config_path = task_dir / "task.toml"
    if not config_path.is_file():
        return

    lines = config_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    in_env = False
    changed = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_env = stripped == "[environment]"
            out.append(line)
            continue

        if in_env and stripped.startswith("docker_image") and "=" in stripped:
            changed = True
            continue

        out.append(line)

    if changed:
        config_path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")



def _set_task_agent_home(task_dir: Path, *, agent: str) -> None:
    config_path = task_dir / "task.toml"
    if not config_path.is_file():
        return

    env_updates = _agent_home_env_vars(agent)
    if not env_updates:
        return

    text = config_path.read_text(encoding="utf-8")
    for k in env_updates.keys():
        if f"{k} =" in text:
            return

    if "[agent.env]" in text:
        text = text.rstrip() + "\n" + "".join(f"{k} = \"{v}\"\n" for k, v in env_updates.items())
    else:
        text = text.rstrip() + "\n\n[agent.env]\n" + "".join(f"{k} = \"{v}\"\n" for k, v in env_updates.items())
    config_path.write_text(text, encoding="utf-8")


def _inject_skill_into_task_workspace(
    task_dir: Path,
    skill_md: Path,
    task_name: str,
    *,
    agent: str,
) -> Path:
    """
    Inject one task-specific skill using the same pattern as skillsbench:
    place under environment/skills and ensure Dockerfile copies to agent-specific skills root.
    """
    root = task_dir / "environment" / "skills" / _normalize_slug(task_name, default="task")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "SKILL.md"
    shutil.copy2(skill_md, target)
    container_skill_root = _agent_skill_container_root(agent)
    _ensure_dockerfile_copies_skills(
        task_dir,
        container_skill_root=container_skill_root,
        agent=agent,
    )
    _set_task_agent_home(task_dir, agent=agent)
    _remove_environment_docker_image(task_dir)
    _remove_skill_instruction_hint(task_dir)
    return target


def _inject_skill_pool_into_task_workspace(
    task_dir: Path,
    skill_entries: list[dict[str, str]],
    task_name: str,
    *,
    agent: str,
) -> list[Path]:
    skills_root = task_dir / "environment" / "skills"
    created: list[Path] = []
    seen: set[str] = set()
    if skills_root.exists():
        shutil.rmtree(skills_root)
    for idx, entry in enumerate(skill_entries, start=1):
        skill_md = Path(str(entry.get("skill_md") or "")).resolve()
        if not skill_md.is_file():
            continue
        skill_name = str(entry.get("skill_name") or "").strip() or skill_md.parent.name
        slug = _normalize_slug(skill_name, default=f"skill-{idx}")
        if slug in seen:
            slug = f"{slug}-{idx}"
        seen.add(slug)
        root = skills_root / slug
        root.mkdir(parents=True, exist_ok=True)
        target = root / "SKILL.md"
        shutil.copy2(skill_md, target)
        created.append(target)
    if created:
        container_skill_root = _agent_skill_container_root(agent)
        _ensure_dockerfile_copies_skills(
            task_dir,
            container_skill_root=container_skill_root,
            agent=agent,
        )
        _set_task_agent_home(task_dir, agent=agent)
        _remove_environment_docker_image(task_dir)
        _remove_skill_instruction_hint(task_dir)
    return created


def _discover_result_files(job_dir: Path) -> list[Path]:
    root_result = (job_dir / "result.json").resolve()
    paths: list[Path] = []
    for name in ("result.json", "results.json"):
        for rp in sorted(job_dir.rglob(name)):
            if name == "result.json" and rp.resolve() == root_result:
                continue
            if rp.is_file():
                paths.append(rp.resolve())
    seen: set[Path] = set()
    out: list[Path] = []
    for rp in paths:
        parent = rp.parent.resolve()
        if parent in seen:
            continue
        seen.add(parent)
        out.append(rp)
    return out


def _extract_reward(payload: dict[str, Any]) -> float | None:
    vals = [payload.get("reward"), ((payload.get("verifier_result") or {}).get("rewards") or {}).get("reward")]
    for v in vals:
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _extract_exception_type(payload: dict[str, Any]) -> str | None:
    info = payload.get("exception_info")
    if isinstance(info, dict) and info.get("exception_type"):
        return str(info["exception_type"])
    return None


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _read_latest_harbor_manifest(run_dir: Path) -> dict[str, Any] | None:
    return _load_json_dict(run_dir / "harbor-manifest.json")


def _merge_harbor_job_results(
    result_payloads: list[dict[str, Any]],
    *,
    fallback_eval_key: str = "codex__prepared-tasks",
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "id": str(uuid4()),
        "started_at": None,
        "finished_at": None,
        "n_total_trials": 0,
        "stats": {"n_trials": 0, "n_errors": 0, "evals": {}},
        "trial_results": [],
    }
    if not result_payloads:
        return merged

    started_candidates: list[str] = []
    finished_candidates: list[str] = []
    eval_acc: dict[str, dict[str, Any]] = {}

    def _get_eval_bucket(eval_key: str) -> dict[str, Any]:
        bucket = eval_acc.get(eval_key)
        if bucket is None:
            bucket = {
                "n_trials": 0,
                "n_errors": 0,
                "reward_stats": {},
                "exception_stats": {},
                "metric_weight": 0.0,
                "metric_weighted_sum": 0.0,
            }
            eval_acc[eval_key] = bucket
        return bucket

    first_id = None
    for payload in result_payloads:
        if not isinstance(payload, dict):
            continue
        if first_id is None and payload.get("id"):
            first_id = str(payload.get("id"))
        started_at = str(payload.get("started_at") or "").strip()
        finished_at = str(payload.get("finished_at") or "").strip()
        if started_at:
            started_candidates.append(started_at)
        if finished_at:
            finished_candidates.append(finished_at)

        merged["n_total_trials"] += int(payload.get("n_total_trials") or 0)
        stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
        merged["stats"]["n_trials"] += int(stats.get("n_trials") or 0)
        merged["stats"]["n_errors"] += int(stats.get("n_errors") or 0)

        evals = stats.get("evals") if isinstance(stats.get("evals"), dict) else {}
        if not evals:
            evals = {fallback_eval_key: {}}
        for eval_key, eval_row in evals.items():
            ek = str(eval_key or fallback_eval_key)
            row = eval_row if isinstance(eval_row, dict) else {}
            bucket = _get_eval_bucket(ek)
            row_n_trials = int(row.get("n_trials") or 0)
            bucket["n_trials"] += row_n_trials
            bucket["n_errors"] += int(row.get("n_errors") or 0)

            reward_stats = row.get("reward_stats") if isinstance(row.get("reward_stats"), dict) else {}
            reward_map = reward_stats.get("reward") if isinstance(reward_stats.get("reward"), dict) else {}
            for reward_key, trials in reward_map.items():
                k = str(reward_key)
                dest = bucket["reward_stats"].setdefault(k, [])
                if isinstance(trials, list):
                    dest.extend(str(x) for x in trials)

            exception_stats = row.get("exception_stats") if isinstance(row.get("exception_stats"), dict) else {}
            for exc_key, trials in exception_stats.items():
                k = str(exc_key)
                dest = bucket["exception_stats"].setdefault(k, [])
                if isinstance(trials, list):
                    dest.extend(str(x) for x in trials)

            metrics = row.get("metrics") if isinstance(row.get("metrics"), list) else []
            metric_mean = None
            for m in metrics:
                if isinstance(m, dict) and isinstance(m.get("mean"), (int, float)):
                    metric_mean = float(m.get("mean"))
                    break
            if metric_mean is not None:
                weight = float(row_n_trials if row_n_trials > 0 else 1)
                bucket["metric_weight"] += weight
                bucket["metric_weighted_sum"] += metric_mean * weight

        trial_results = payload.get("trial_results")
        if isinstance(trial_results, list):
            merged["trial_results"].extend(trial_results)

    if first_id is not None:
        merged["id"] = first_id
    if started_candidates:
        merged["started_at"] = min(started_candidates)
    if finished_candidates:
        merged["finished_at"] = max(finished_candidates)

    for eval_key, bucket in eval_acc.items():
        mean_value = 0.0
        if bucket["metric_weight"] > 0:
            mean_value = float(bucket["metric_weighted_sum"]) / float(bucket["metric_weight"])
        merged["stats"]["evals"][eval_key] = {
            "n_trials": int(bucket["n_trials"]),
            "n_errors": int(bucket["n_errors"]),
            "metrics": [{"mean": mean_value}],
            "reward_stats": {"reward": bucket["reward_stats"]},
            "exception_stats": bucket["exception_stats"],
        }

    return merged


def _merge_harbor_job_dirs(
    *,
    run_dir: Path,
    source_job_dirs: list[Path],
    target_jobs_root: Path | None = None,
) -> dict[str, Any]:
    jobs_root = target_jobs_root.resolve() if target_jobs_root is not None else (run_dir / "harbor-jobs").resolve()
    canonical_sources: list[Path] = []
    seen: set[str] = set()
    for p in source_job_dirs:
        rp = p.resolve()
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        if rp.is_dir():
            canonical_sources.append(rp)

    if not canonical_sources:
        return {"status": "skipped_no_source_jobs", "source_job_count": 0}

    jobs_root.mkdir(parents=True, exist_ok=True)
    target_dir = canonical_sources[0]
    if target_dir.parent.resolve() != jobs_root:
        target_dir = jobs_root / target_dir.name
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        shutil.move(str(canonical_sources[0]), str(target_dir))
        canonical_sources[0] = target_dir
    rename_candidate = re.sub(r"-a\d+(?=-gpt-)", "", target_dir.name)
    if rename_candidate and rename_candidate != target_dir.name:
        candidate_path = target_dir.parent / rename_candidate
        if not candidate_path.exists():
            target_dir = target_dir.rename(candidate_path)

    moved_trial_count = 0
    renamed_trial_count = 0
    removed_job_dirs: list[str] = []
    source_result_payloads: list[dict[str, Any]] = []
    target_result = _load_json_dict(target_dir / "result.json")
    if target_result is not None:
        source_result_payloads.append(target_result)

    merge_meta_dir = target_dir / "_merge_meta"
    merge_meta_dir.mkdir(parents=True, exist_ok=True)
    target_job_log = target_dir / "job.log"
    used_names = {child.name for child in target_dir.iterdir()}
    special_files = {"config.json", "result.json", "job.log", "_merge_meta"}

    for src in canonical_sources[1:]:
        src_result = _load_json_dict(src / "result.json")
        if src_result is not None:
            source_result_payloads.append(src_result)

        for meta_name in ("config.json", "result.json", "job.log"):
            meta_src = src / meta_name
            if meta_src.is_file():
                meta_dst = merge_meta_dir / f"{src.name}.{meta_name}"
                shutil.copy2(meta_src, meta_dst)

        src_job_log = src / "job.log"
        if src_job_log.is_file():
            with target_job_log.open("a", encoding="utf-8", errors="replace") as out_f:
                out_f.write(f"\n\n===== MERGED FROM {src.name} =====\n")
                out_f.write(src_job_log.read_text(encoding="utf-8", errors="replace"))

        for child in sorted(src.iterdir()):
            if child.name in special_files:
                continue
            dst_name = child.name
            dst = target_dir / dst_name
            if dst.exists():
                suffix = 1
                while True:
                    candidate = target_dir / f"{dst_name}__m{suffix}"
                    if not candidate.exists():
                        dst = candidate
                        renamed_trial_count += 1
                        break
                    suffix += 1
            shutil.move(str(child), str(dst))
            moved_trial_count += 1
            used_names.add(dst.name)

        shutil.rmtree(src, ignore_errors=True)
        removed_job_dirs.append(str(src))

    merged_result = _merge_harbor_job_results(source_result_payloads)
    _json_dump(target_dir / "result.json", merged_result)

    return {
        "status": "ok",
        "target_job_dir": str(target_dir),
        "source_job_count": len(canonical_sources),
        "removed_job_dirs": removed_job_dirs,
        "moved_trial_count": moved_trial_count,
        "renamed_trial_count": renamed_trial_count,
    }


def _collect_success_rate(run_dir: Path) -> dict[str, Any]:
    jobs_root = run_dir / "harbor-jobs"
    if not jobs_root.is_dir():
        return {"trial_total": 0, "success_total": 0, "success_rate": 0.0}
    job_dirs = sorted([p for p in jobs_root.iterdir() if p.is_dir()])
    if not job_dirs:
        return {"trial_total": 0, "success_total": 0, "success_rate": 0.0}

    total = 0
    success = 0
    per_task: dict[str, dict[str, int]] = {}
    for job_dir in job_dirs:
        for result_path in _discover_result_files(job_dir):
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            total += 1
            task_name = str(payload.get("task_name") or result_path.parent.name.split("__", 1)[0])
            reward = _extract_reward(payload)
            exc = _extract_exception_type(payload)
            ok = bool(reward is not None and reward >= 1.0 and not exc)
            if ok:
                success += 1
            row = per_task.setdefault(task_name, {"total": 0, "success": 0})
            row["total"] += 1
            if ok:
                row["success"] += 1

    for task, row in per_task.items():
        row["success_rate"] = (float(row["success"]) / float(row["total"])) if row["total"] > 0 else 0.0

    return {
        "trial_total": total,
        "success_total": success,
        "success_rate": (float(success) / float(total)) if total > 0 else 0.0,
        "per_task": per_task,
    }


def _build_arm_command(
    *,
    procmem2skills_root: Path,
    benchmark: dict[str, Any],
    run_dir: Path,
    prepared_tasks_dir: Path,
    experiment_id: str,
    model: str,
    agent: str,
    n_attempts: int,
    n_concurrent: int,
    max_steps: int,
    command_timeout_sec: int,
    base_url: str | None,
    agent_env: dict[str, str] | None,
    runner_python: str | None,
    dry_run: bool,
    memory_setting: str = "none",
    workflow_memory_path: Path | None = None,
    workflow_max_attempts: int | None = None,
    workflow_max_workflows_per_attempt: int | None = None,
    workflow_max_steps_per_workflow: int | None = None,
) -> list[str]:
    script = procmem2skills_root / "scripts" / "server" / "run_skillsbench_harbor_experiment.py"
    python_launcher = runner_python or sys.executable
    normalized_memory_setting = str(memory_setting or "none").strip().lower() or "none"

    cmd = [
        python_launcher,
        str(script),
        "--experiment-id",
        experiment_id,
        "--model",
        model,
        "--dataset",
        str(benchmark["dataset"]),
        "--run-dir",
        str(run_dir),
        "--source-mode",
        "path",
        "--skillsbench-path",
        str(prepared_tasks_dir),
        "--memory-setting",
        normalized_memory_setting,
        "--n-attempts",
        str(max(1, n_attempts)),
        "--n-concurrent",
        str(max(1, n_concurrent)),
        "--max-steps",
        str(max(1, max_steps)),
        "--command-timeout-sec",
        str(max(1, command_timeout_sec)),
        "--import-benchmark",
        "terminal-bench",
        "--harness",
        "skills-bench/harness" if str(benchmark.get("runner") or "").strip().lower() == "skillsbench" else "terminal-bench/harness",
    ]

    if normalized_memory_setting == "none":
        cmd.extend(["--native-agent", agent])
    elif normalized_memory_setting == "workflows":
        if workflow_memory_path is None:
            raise RuntimeError("workflow memory mode requires workflow_memory_path")
        cmd.extend(["--workflow-memory-path", str(workflow_memory_path)])
        if workflow_max_attempts is not None and int(workflow_max_attempts) > 0:
            cmd.extend(["--workflow-max-attempts", str(int(workflow_max_attempts))])
        if workflow_max_workflows_per_attempt is not None and int(workflow_max_workflows_per_attempt) > 0:
            cmd.extend(["--workflow-max-workflows-per-attempt", str(int(workflow_max_workflows_per_attempt))])
        if workflow_max_steps_per_workflow is not None and int(workflow_max_steps_per_workflow) > 0:
            cmd.extend(["--workflow-max-steps-per-workflow", str(int(workflow_max_steps_per_workflow))])
    else:
        raise RuntimeError(f"unsupported memory_setting for this runner: {normalized_memory_setting}")

    if base_url:
        cmd.extend(["--base-url", base_url])
    if agent_env:
        for k in sorted(agent_env.keys()):
            v = str(agent_env[k])
            cmd.extend(["--ae", f"{k}={v}"])
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline/workflow/skill context comparison experiments.")
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--skills-root", type=Path, required=True)
    parser.add_argument(
        "--compact-procedure-root",
        type=Path,
        default=None,
        help="Root containing compact procedure artifacts: {condition}/{form}/{task}/procedure.md.",
    )
    parser.add_argument("--skills-manifest", type=Path, default=None)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--task-source-root", type=Path, required=True)
    parser.add_argument("--procmem2skills-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--output-layout",
        choices=["legacy", "normal"],
        default="legacy",
        help="legacy: {output_root}/{agent_model}/pipeline-v1/eval/{ratio}; normal: {output_root}/{benchmark}/{ratio}",
    )
    parser.add_argument(
        "--benchmark-output",
        default=None,
        help="Optional benchmark slug override for --output-layout normal (e.g. skillsbench).",
    )
    parser.add_argument("--provider", choices=sorted(PROVIDER_DEFAULT_KEY_ENV.keys()), required=True)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--m-success", type=int, default=4)
    parser.add_argument("--n-failure", type=int, default=1)
    parser.add_argument("--n-attempts", type=int, default=5)
    parser.add_argument("--attempt-execution-mode", choices=["serial-merge", "direct"], default="serial-merge", help="serial-merge: split into N single-attempt runs then merge; direct: single run with requested n-attempts.")
    parser.add_argument("--n-concurrent", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--command-timeout-sec", type=int, default=1200)
    parser.add_argument("--docker-cleanup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--docker-cleanup-timeout-sec", type=int, default=180)
    parser.add_argument("--docker-cleanup-strict", action="store_true")
    parser.add_argument("--task-name", action="append", default=[])
    parser.add_argument("--run-id", default=None, help="Stable run identifier under eval/{condition}/runs (default: utc timestamp slug).")
    parser.add_argument(
        "--arms",
        default="baseline,workflow,skill",
        help="Comma-separated arms to run: baseline,workflow,skill,short-plan,test-first,script",
    )
    parser.add_argument(
        "--workflow-injection-mode",
        choices=["instruction", "memory"],
        default="instruction",
        help="How to run workflow arm: instruction appending vs native workflow-memory channel.",
    )
    parser.add_argument(
        "--workflow-hint-mode",
        choices=["with-status", "no-hint"],
        default="with-status",
        help="with-status: expose success/failure labels in injected workflow traces; no-hint: hide labels.",
    )
    parser.add_argument(
        "--workflow-memory-max-attempts",
        type=int,
        default=0,
        help="Optional cap forwarded in workflow-memory mode; <=0 means all.",
    )
    parser.add_argument(
        "--workflow-memory-max-workflows-per-attempt",
        type=int,
        default=0,
        help="Optional cap forwarded in workflow-memory mode; <=0 means all.",
    )
    parser.add_argument(
        "--workflow-memory-max-steps-per-workflow",
        type=int,
        default=0,
        help="Optional cap forwarded in workflow-memory mode; <=0 means all.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    provider = str(args.provider).strip().lower()
    api_key_env = args.api_key_env or PROVIDER_DEFAULT_KEY_ENV[provider]
    if provider in {"google", "claude"} and not args.base_url:
        raise RuntimeError(f"provider={provider} requires --base-url")
    provider_env = _prepare_provider_env(provider=provider, api_key_env=api_key_env, base_url=args.base_url)
    provider_env = _resolve_stable_harbor_runtime(
        procmem2skills_root=args.procmem2skills_root.resolve(),
        env=provider_env,
    )

    trace_records = _load_traces(args.trace_root.resolve())
    by_task = _group_traces_by_task(trace_records)
    benchmark_map = _load_benchmark_map(args.benchmark_config.resolve())

    condition_label = f"{max(0, int(args.m_success))}s{max(0, int(args.n_failure))}f"
    transfer_skill_pool_for_selection = _load_transfer_skill_pool_manifest(args.skills_manifest.resolve()) if args.skills_manifest else {}

    model_for_runner = _strip_openai_model_prefix(args.model)

    arms_text = str(args.arms or "").strip()
    requested_arms = [x.strip().lower() for x in arms_text.split(",") if x.strip()]
    allowed_arms = {"baseline", "workflow", "skill", *COMPACT_PROCEDURE_ARMS}
    if not requested_arms:
        requested_arms = ["baseline", "workflow", "skill"]
    invalid = [a for a in requested_arms if a not in allowed_arms]
    if invalid:
        raise RuntimeError(f"invalid --arms values: {invalid}; allowed={sorted(allowed_arms)}")
    seen_arms: set[str] = set()
    arms: list[str] = []
    for a in requested_arms:
        if a in seen_arms:
            continue
        seen_arms.add(a)
        arms.append(a)

    compact_arms = [a for a in arms if a in COMPACT_PROCEDURE_ARMS]
    compact_procedure_root = args.compact_procedure_root.resolve() if args.compact_procedure_root else None
    if compact_arms and compact_procedure_root is None:
        raise RuntimeError("--compact-procedure-root is required when compact procedure arms are requested")

    require_skill_for_selection = (not arms_text) or any(
        part.strip().lower() == "skill" for part in arms
    )

    run_id_raw = str(args.run_id or "").strip()
    if run_id_raw:
        run_id = _normalize_slug(run_id_raw, default="run", max_len=96)
    else:
        run_id = _normalize_slug(f"run-{int(time.time())}-{uuid4().hex[:8]}", default="run", max_len=96)

    selected_tasks: list[str] = []
    task_filter = {t.strip() for t in args.task_name if t.strip()}
    for task_name, row in sorted(by_task.items()):
        if task_filter and task_name not in task_filter:
            continue
        if len(row["success"]) < max(0, int(args.m_success)) or len(row["failure"]) < max(0, int(args.n_failure)):
            continue
        if require_skill_for_selection:
            has_default_skill = _load_skill_for_condition(args.skills_root.resolve(), condition_label, task_name) is not None
            has_transfer_skill_pool = bool(transfer_skill_pool_for_selection.get(task_name))
            if not has_default_skill and not has_transfer_skill_pool:
                continue
        if compact_arms:
            missing_compact = [
                arm
                for arm in compact_arms
                if _load_compact_procedure_for_condition(compact_procedure_root, condition_label, arm, task_name) is None
            ]
            if missing_compact:
                continue
        selected_tasks.append(task_name)

    if not selected_tasks:
        raise RuntimeError("no tasks eligible for comparison (need traces + skill for condition)")

    agent_model = _normalize_slug(f"{args.agent}-{args.model.split('/')[-1]}", default="agent-model")

    by_benchmark_tasks: dict[str, list[str]] = {}
    for task_name in selected_tasks:
        bid = str(by_task[task_name].get("benchmark") or "unknown-benchmark")
        by_benchmark_tasks.setdefault(bid, []).append(task_name)

    output_layout = str(args.output_layout or "legacy").strip().lower()
    if output_layout not in {"legacy", "normal"}:
        raise RuntimeError(f"invalid --output-layout: {args.output_layout}")

    output_slug_set = {
        _benchmark_output_slug(
            benchmark_id=bid,
            benchmark_cfg=benchmark_map.get(bid),
        )
        for bid in by_benchmark_tasks.keys()
    }
    output_slug_set = {x for x in output_slug_set if x}

    if output_layout == "normal":
        benchmark_output_slug = str(args.benchmark_output or "").strip()
        if not benchmark_output_slug:
            if len(output_slug_set) == 1:
                benchmark_output_slug = next(iter(output_slug_set))
            elif len(output_slug_set) > 1:
                benchmark_output_slug = "mixed"
            else:
                benchmark_output_slug = "unknown"
        eval_condition_root = args.output_root.resolve() / benchmark_output_slug / condition_label
    else:
        pipeline_root = args.output_root.resolve() / agent_model / "pipeline-v1"
        eval_condition_root = pipeline_root / "eval" / condition_label

    eval_root = eval_condition_root / "runs" / run_id

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "condition": condition_label,
        "run_id": run_id,
        "eval_run_root": str(eval_root),
        "task_count": len(selected_tasks),
        "tasks": selected_tasks,
        "procmem2skills_root": str(args.procmem2skills_root.resolve()),
        "dry_run": bool(args.dry_run),
        "arms_requested": arms,
        "output_layout": output_layout,
        "workflow_injection_mode": str(args.workflow_injection_mode),
        "workflow_hint_mode": str(args.workflow_hint_mode),
        "skills_manifest": str(args.skills_manifest.resolve()) if args.skills_manifest else None,
        "compact_procedure_root": str(compact_procedure_root) if compact_procedure_root else None,
        "docker_cleanup": {
            "enabled": bool(args.docker_cleanup),
            "timeout_sec": int(args.docker_cleanup_timeout_sec),
            "strict": bool(args.docker_cleanup_strict),
        },
        "arms": {},
    }

    overall_run_failed = False
    for arm in arms:
        arm_entry: dict[str, Any] = {"benchmarks": {}, "overall": {}}
        total_trials = 0
        total_success = 0
        transfer_skill_pool = _load_transfer_skill_pool_manifest(args.skills_manifest.resolve()) if (arm == "skill" and args.skills_manifest) else {}

        for benchmark_id, task_names in sorted(by_benchmark_tasks.items()):
            bench = benchmark_map.get(benchmark_id)
            if bench is None:
                arm_entry["benchmarks"][benchmark_id] = {"status": "skipped_unknown_benchmark", "task_count": len(task_names)}
                continue

            prepared_root = eval_root / arm / benchmark_id / "prepared-tasks"
            if prepared_root.exists():
                shutil.rmtree(prepared_root)
            prepared_root.mkdir(parents=True, exist_ok=True)

            available_tasks: list[str] = []
            skipped_missing_task_source: list[str] = []
            src_roots = _candidate_task_source_roots(
                base_root=args.task_source_root.resolve(),
                benchmark_cfg=bench,
                procmem2skills_root=args.procmem2skills_root.resolve(),
            )
            for task_name in task_names:
                if _find_task_source(src_roots, task_name) is None:
                    skipped_missing_task_source.append(task_name)
                    continue
                available_tasks.append(task_name)

            if not available_tasks:
                arm_entry["benchmarks"][benchmark_id] = {
                    "status": "skipped_missing_task_source",
                    "task_count": 0,
                    "requested_task_count": len(task_names),
                    "skipped_missing_task_source_count": len(skipped_missing_task_source),
                    "skipped_missing_task_source": skipped_missing_task_source,
                    "trial_total": 0,
                    "success_total": 0,
                    "success_rate": 0.0,
                    "per_task": {},
                }
                continue

            skill_index_entries: list[dict[str, str]] = []
            compact_procedure_entries: list[dict[str, str]] = []
            workflow_memory_payload: dict[str, list[dict[str, Any]]] = {}

            for task_name in available_tasks:
                task_dir = _copy_task_source(src_roots, task_name, prepared_root)
                if arm == "workflow":
                    succ = by_task[task_name]["success"][: max(0, int(args.m_success))]
                    fail = by_task[task_name]["failure"][: max(0, int(args.n_failure))]
                    selected_traces = [*succ, *fail]
                    if args.workflow_injection_mode == "instruction":
                        _inject_workflow_context(
                            task_dir,
                            _workflow_block(task_name, selected_traces, hint_mode=str(args.workflow_hint_mode)),
                        )
                    else:
                        attempts = _workflow_attempts_from_traces(
                            task_name,
                            selected_traces,
                            hint_mode=str(args.workflow_hint_mode),
                        )
                        if attempts:
                            workflow_memory_payload[task_name] = attempts
                elif arm == "skill":
                    transfer_entries = transfer_skill_pool.get(task_name) or []
                    if transfer_entries:
                        injected_skills = _inject_skill_pool_into_task_workspace(task_dir, transfer_entries, task_name, agent=args.agent)
                        skill_index_entries.append({
                            "task_name": task_name,
                            "source_skills": [str(entry.get("skill_md") or "") for entry in transfer_entries],
                            "injected_skills": [str(path) for path in injected_skills],
                            "container_skill_root": _agent_skill_container_root(args.agent),
                        })
                    else:
                        skill_md = _load_skill_for_condition(args.skills_root.resolve(), condition_label, task_name)
                        if skill_md is not None:
                            injected_skill = _inject_skill_into_task_workspace(task_dir, skill_md, task_name, agent=args.agent)
                            skill_index_entries.append({
                                "task_name": task_name,
                                "source_skill": str(skill_md),
                                "injected_skill": str(injected_skill),
                                "container_skill_root": _agent_skill_container_root(args.agent),
                            })
                elif arm in COMPACT_PROCEDURE_ARMS:
                    procedure_path = _load_compact_procedure_for_condition(
                        compact_procedure_root,
                        condition_label,
                        arm,
                        task_name,
                    )
                    if procedure_path is None:
                        raise RuntimeError(f"missing compact procedure for arm={arm} condition={condition_label} task={task_name}")
                    _inject_compact_procedure_context(task_dir, arm, procedure_path)
                    compact_procedure_entries.append({
                        "task_name": task_name,
                        "form": arm,
                        "source_procedure": str(procedure_path),
                    })

            workflow_memory_path: Path | None = None
            memory_setting = "none"
            workflow_max_attempts_arg: int | None = None
            workflow_max_workflows_per_attempt_arg: int | None = None
            workflow_max_steps_per_workflow_arg: int | None = None
            if arm == "workflow" and args.workflow_injection_mode == "memory":
                if not workflow_memory_payload:
                    raise RuntimeError(f"workflow memory payload is empty for benchmark={benchmark_id}")
                workflow_memory_path = eval_root / arm / benchmark_id / "workflow-memory" / "grouped_workflow_memory.json"
                _json_dump(workflow_memory_path, workflow_memory_payload)
                memory_setting = "workflows"
                workflow_max_attempts_arg = int(args.workflow_memory_max_attempts)
                workflow_max_workflows_per_attempt_arg = int(args.workflow_memory_max_workflows_per_attempt)
                workflow_max_steps_per_workflow_arg = int(args.workflow_memory_max_steps_per_workflow)

            run_dir = eval_root / arm / benchmark_id / "results"
            exp_id = _normalize_slug(f"pm2s-{arm}-{benchmark_id}-{int(time.time())}", default="pm2s", max_len=56)
            # Keep codex-attempt timeout aligned with harness command timeout.
            # The wrapper default is 240s; long trials get truncated otherwise.
            agent_env_overrides: dict[str, str] = {
                "HARBOR_CODEX_ATTEMPT_TIMEOUT_SEC": str(max(300, int(args.command_timeout_sec))),
            }
            # Agent-specific runtime env (e.g., Gemini trust gate, per-agent home).
            # For skill arm, GEMINI_CLI_HOME/CODEX_HOME also controls skill discovery roots.
            agent_env_overrides.update(_agent_home_env_vars(args.agent))

            # For gemini-cli, explicitly forward GEMINI_API_KEY into Harbor agent env.
            if _normalize_agent_name(args.agent) in {"gemini-cli", "gemini"}:
                gemini_key = str(provider_env.get("GEMINI_API_KEY") or "").strip()
                if gemini_key:
                    agent_env_overrides["GEMINI_API_KEY"] = gemini_key

            # Claude Code OAuth must use the refreshable credentials file inside
            # Harbor containers. Do not forward CLAUDE_CODE_OAUTH_TOKEN: it is a
            # short-lived access token and causes long multi-task runs to degrade
            # into uniform 401 failures after expiry. The Harbor claude-code
            # agent uploads CLAUDE_CODE_CREDENTIALS_PATH during setup.
            if _normalize_agent_name(args.agent) in {"claude-code", "claude"}:
                provider_env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
                claude_credentials_path = str(
                    provider_env.get("CLAUDE_CODE_CREDENTIALS_PATH")
                    or provider_env.get("CLAUDE_CODE_CREDENTIALS_FILE")
                    or ""
                ).strip()
                if claude_credentials_path:
                    provider_env["CLAUDE_CODE_CREDENTIALS_PATH"] = claude_credentials_path
                # Let the Harbor claude-code agent choose its container-local
                # CLAUDE_CONFIG_DIR so it can use the uploaded credentials.
                provider_env.pop("CLAUDE_CONFIG_DIR", None)

            requested_attempts = max(1, int(args.n_attempts))
            attempt_mode = str(args.attempt_execution_mode or "serial-merge").strip().lower()
            direct_mode = (attempt_mode == "direct")
            attempt_strategy = "direct_single_run_with_requested_n_attempts" if direct_mode else ("single_run" if requested_attempts == 1 else "serial_runs_with_single_attempt")
            plan_count = 1 if direct_mode else requested_attempts
            n_attempts_per_run = requested_attempts if direct_mode else 1
            attempt_plan: list[dict[str, Any]] = []
            for attempt_idx in range(1, plan_count + 1):
                attempt_exp_id = exp_id
                if (not direct_mode) and requested_attempts > 1:
                    attempt_exp_id = _normalize_slug(
                        f"{exp_id}-a{attempt_idx}",
                        default="pm2s",
                        max_len=56,
                    )
                cmd = _build_arm_command(
                    procmem2skills_root=args.procmem2skills_root.resolve(),
                    benchmark=bench,
                    run_dir=run_dir,
                    prepared_tasks_dir=prepared_root,
                    experiment_id=attempt_exp_id,
                    model=model_for_runner,
                    agent=args.agent,
                    n_attempts=n_attempts_per_run,
                    n_concurrent=args.n_concurrent,
                    max_steps=args.max_steps,
                    command_timeout_sec=args.command_timeout_sec,
                    base_url=args.base_url,
                    agent_env=agent_env_overrides,
                    runner_python=provider_env.get("PROCMEM_BENCHMARK_PYTHON"),
                    dry_run=bool(args.dry_run),
                    memory_setting=memory_setting,
                    workflow_memory_path=workflow_memory_path,
                    workflow_max_attempts=workflow_max_attempts_arg,
                    workflow_max_workflows_per_attempt=workflow_max_workflows_per_attempt_arg,
                    workflow_max_steps_per_workflow=workflow_max_steps_per_workflow_arg,
                )
                attempt_plan.append(
                    {
                        "attempt_index": attempt_idx,
                        "experiment_id": attempt_exp_id,
                        "command": cmd,
                    }
                )

            run_manifest = {
                "arm": arm,
                "benchmark": benchmark_id,
                "dataset": bench.get("dataset"),
                "task_count": len(available_tasks),
                "requested_task_count": len(task_names),
                "skipped_missing_task_source_count": len(skipped_missing_task_source),
                "skipped_missing_task_source": skipped_missing_task_source,
                "prepared_tasks_dir": str(prepared_root),
                "run_dir": str(run_dir),
                "command": _redact_command_for_manifest(attempt_plan[0]["command"]) if attempt_plan else [],
                "commands": [_redact_command_for_manifest(row["command"]) for row in attempt_plan],
                "requested_n_attempts": requested_attempts,
                "execution_n_attempts_per_run": n_attempts_per_run,
                "attempt_strategy": attempt_strategy,
                "status": "dry_run" if args.dry_run else "planned",
            }
            if arm == "skill":
                run_manifest["agent_env_overrides"] = _redact_agent_env(agent_env_overrides)
                run_manifest["skill_injection_mode"] = f"skillsbench_style_env_skills_copy_to_{_normalize_agent_name(args.agent)}"
                run_manifest["skill_index_entries"] = skill_index_entries
            if arm == "workflow":
                run_manifest["workflow_injection_mode"] = str(args.workflow_injection_mode)
                run_manifest["workflow_hint_mode"] = str(args.workflow_hint_mode)
                if workflow_memory_path is not None:
                    run_manifest["workflow_memory_path"] = str(workflow_memory_path)
                    run_manifest["workflow_memory_task_count"] = len(workflow_memory_payload)
            if arm in COMPACT_PROCEDURE_ARMS:
                run_manifest["compact_procedure_injection_mode"] = "instruction_append"
                run_manifest["compact_procedure_entries"] = compact_procedure_entries

            if args.dry_run:
                metrics = {"trial_total": 0, "success_total": 0, "success_rate": 0.0, "per_task": {}}
            else:
                attempt_results: list[dict[str, Any]] = []
                had_failed_run = False
                cleanup_had_error = False
                source_job_dirs: list[Path] = []
                attempt_run_dirs: list[Path] = []
                for row in attempt_plan:
                    attempt_run_dir = run_dir / "_attempt_runs" / f"attempt-{int(row['attempt_index']):02d}"
                    attempt_run_dir.mkdir(parents=True, exist_ok=True)
                    attempt_run_dirs.append(attempt_run_dir)
                    attempt_cmd = list(row["command"])
                    if "--run-dir" in attempt_cmd:
                        ridx = attempt_cmd.index("--run-dir")
                        if ridx + 1 < len(attempt_cmd):
                            attempt_cmd[ridx + 1] = str(attempt_run_dir)
                    else:
                        attempt_cmd.extend(["--run-dir", str(attempt_run_dir)])
                    completed = subprocess.run(
                        attempt_cmd,
                        cwd=args.procmem2skills_root.resolve(),
                        env=provider_env,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    run_ok = completed.returncode == 0
                    redacted_file_count = _redact_sensitive_files(attempt_run_dir, provider_env)
                    attempt_result: dict[str, Any] = {
                        "attempt_index": row["attempt_index"],
                        "experiment_id": row["experiment_id"],
                        "run_dir": str(attempt_run_dir),
                        "return_code": int(completed.returncode),
                        "status": "ok" if run_ok else "failed",
                        "stdout_tail": _redact_sensitive_text((completed.stdout or "")[-3000:], provider_env),
                        "stderr_tail": _redact_sensitive_text((completed.stderr or "")[-3000:], provider_env),
                        "redacted_sensitive_file_count": redacted_file_count,
                    }
                    attempt_results.append(attempt_result)
                    if not run_ok:
                        had_failed_run = True

                    harbor_manifest = _read_latest_harbor_manifest(attempt_run_dir)
                    if harbor_manifest is not None:
                        attempt_result["harbor_manifest_path"] = str(attempt_run_dir / "harbor-manifest.json")
                        job_dir_text = str(harbor_manifest.get("job_dir") or "").strip()
                        if job_dir_text:
                            job_dir_path = Path(job_dir_text)
                            attempt_result["harbor_job_dir"] = str(job_dir_path)
                            if job_dir_path.is_dir():
                                source_job_dirs.append(job_dir_path)

                    # Clean between serial runs to avoid accumulated Docker side effects.
                    if int(row["attempt_index"]) < len(attempt_plan):
                        inter_cleanup = _cleanup_docker(
                            enabled=bool(args.docker_cleanup),
                            timeout_sec=int(args.docker_cleanup_timeout_sec),
                            strict=False,
                        )
                        attempt_result["inter_attempt_docker_cleanup"] = inter_cleanup
                        if bool(inter_cleanup.get("had_error")):
                            cleanup_had_error = True

                run_manifest["attempt_results"] = attempt_results
                run_manifest["return_code"] = 0 if not had_failed_run else 1
                if attempt_results:
                    run_manifest["stdout_tail"] = attempt_results[-1].get("stdout_tail", "")
                    run_manifest["stderr_tail"] = attempt_results[-1].get("stderr_tail", "")
                run_manifest["status"] = "ok" if not had_failed_run else "failed"
                if cleanup_had_error:
                    run_manifest["inter_attempt_cleanup_had_error"] = True

                if len(attempt_plan) > 1:
                    if not source_job_dirs:
                        for ard in attempt_run_dirs:
                            jobs_root = ard / "harbor-jobs"
                            if jobs_root.is_dir():
                                source_job_dirs.extend(sorted([p for p in jobs_root.iterdir() if p.is_dir()]))
                    merge_info = _merge_harbor_job_dirs(
                        run_dir=run_dir,
                        source_job_dirs=source_job_dirs,
                        target_jobs_root=run_dir / "harbor-jobs",
                    )
                    run_manifest["harbor_merge"] = merge_info
                else:
                    single_attempt_root = attempt_run_dirs[0] if attempt_run_dirs else None
                    if single_attempt_root is not None:
                        src_jobs = single_attempt_root / "harbor-jobs"
                        source_job_dirs = sorted([p for p in src_jobs.iterdir() if p.is_dir()]) if src_jobs.is_dir() else []
                        merge_info = _merge_harbor_job_dirs(
                            run_dir=run_dir,
                            source_job_dirs=source_job_dirs,
                            target_jobs_root=run_dir / "harbor-jobs",
                        ) if source_job_dirs else {"status": "skipped_no_source_jobs", "source_job_count": 0}
                        run_manifest["harbor_merge"] = merge_info

                # Always collect metrics from raw result files for skillsbench.
                # Even when importer exits non-zero, Harbor may already have complete trial outputs.
                metrics = _collect_success_rate(run_dir)

            cleanup_info = _cleanup_docker(
                enabled=bool(args.docker_cleanup),
                timeout_sec=int(args.docker_cleanup_timeout_sec),
                strict=bool(args.docker_cleanup_strict),
            )
            run_manifest["docker_cleanup"] = cleanup_info
            if bool(args.docker_cleanup_strict) and bool(cleanup_info.get("had_error")):
                run_manifest["status"] = "failed"
                run_manifest["docker_cleanup_strict_failure"] = True

            run_manifest["metrics"] = metrics
            _json_dump(run_dir / "comparison_run_manifest.json", run_manifest)
            if str(run_manifest.get("status") or "").strip().lower() == "failed":
                overall_run_failed = True

            benchmark_entry = {
                "status": run_manifest["status"],
                "docker_cleanup_status": (run_manifest.get("docker_cleanup") or {}).get("status"),
                "run_dir": str(run_dir),
                "task_count": len(available_tasks),
                "requested_task_count": len(task_names),
                "skipped_missing_task_source_count": len(skipped_missing_task_source),
                "skipped_missing_task_source": skipped_missing_task_source,
                **metrics,
            }
            if arm == "workflow":
                benchmark_entry["workflow_injection_mode"] = str(args.workflow_injection_mode)
                benchmark_entry["workflow_hint_mode"] = str(args.workflow_hint_mode)
                if workflow_memory_path is not None:
                    benchmark_entry["workflow_memory_path"] = str(workflow_memory_path)
                    benchmark_entry["workflow_memory_task_count"] = len(workflow_memory_payload)
            arm_entry["benchmarks"][benchmark_id] = benchmark_entry

            total_trials += int(metrics.get("trial_total") or 0)
            total_success += int(metrics.get("success_total") or 0)

        arm_entry["overall"] = {
            "trial_total": total_trials,
            "success_total": total_success,
            "success_rate": (float(total_success) / float(total_trials)) if total_trials > 0 else 0.0,
        }
        report["arms"][arm] = arm_entry

    baseline_rate = float(report["arms"].get("baseline", {}).get("overall", {}).get("success_rate", 0.0))
    workflow_rate = float(report["arms"].get("workflow", {}).get("overall", {}).get("success_rate", 0.0))
    skill_rate = float(report["arms"].get("skill", {}).get("overall", {}).get("success_rate", 0.0))
    report["deltas"] = {
        "workflow_minus_baseline": workflow_rate - baseline_rate,
        "skill_minus_baseline": skill_rate - baseline_rate,
        "skill_minus_workflow": skill_rate - workflow_rate,
    }

    report_path = eval_root / "comparison_report.json"
    _json_dump(report_path, report)

    latest_payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "condition": condition_label,
        "run_id": run_id,
        "report_path": str(report_path),
        "run_root": str(eval_root),
    }
    _json_dump(eval_condition_root / "latest.json", latest_payload)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if overall_run_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
