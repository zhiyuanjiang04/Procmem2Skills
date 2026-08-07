#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CLEAN_ROOT = Path(os.environ.get("PM2S_CLEAN_ROOT", str(Path(__file__).resolve().parents[1])))
DEFAULT_RESULTS_ROOT = Path(os.environ.get("PM2S_RESULTS_ROOT", str(CLEAN_ROOT / "outputs")))


@dataclass
class TrialRecord:
    task_name: str
    trial_name: str
    status: str  # success|failure|runtime_error
    reward: float | None
    exception_type: str | None
    result_json: Path
    codex_txt: Path | None
    trajectory_json: Path | None
    gemini_cli_txt: Path | None
    source_root: str  # raw|collect
    mtime: float


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_reward(payload: dict[str, Any]) -> float | None:
    candidates = [
        payload.get("reward"),
        ((payload.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
        (payload.get("agent_result") or {}).get("score"),
    ]
    for value in candidates:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_exception_type(payload: dict[str, Any]) -> str | None:
    info = payload.get("exception_info")
    if isinstance(info, dict):
        et = info.get("exception_type")
        if et:
            return str(et)
    return None


def _discover_codex_file(trial_dir: Path) -> Path | None:
    preferred = trial_dir / "agent" / "codex.txt"
    if preferred.is_file():
        return preferred.resolve()
    for p in sorted(trial_dir.rglob("codex.txt")):
        if p.is_file():
            return p.resolve()
    return None


def _discover_trajectory_file(trial_dir: Path) -> Path | None:
    preferred = trial_dir / "agent" / "trajectory.json"
    if preferred.is_file():
        return preferred.resolve()
    for p in sorted(trial_dir.rglob("trajectory.json")):
        if p.is_file():
            return p.resolve()
    return None


def _discover_gemini_cli_file(trial_dir: Path) -> Path | None:
    preferred = trial_dir / "agent" / "gemini-cli.txt"
    if preferred.is_file():
        return preferred.resolve()
    for p in sorted(trial_dir.rglob("gemini-cli.txt")):
        if p.is_file():
            return p.resolve()
    return None


def _trial_status(success: bool, evidence_exists: bool) -> str:
    if success:
        return "success"
    if evidence_exists:
        return "failure"
    return "runtime_error"


def _load_qualified_tasks(status_log: Path) -> set[str]:
    tasks: set[str] = set()
    for raw in status_log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts: dict[str, str] = {}
        for kv in line.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                parts[k] = v
        task = parts.get("task")
        if not task:
            continue
        need_s = int(parts.get("need_success", "0") or "0")
        need_f = int(parts.get("need_failure", "0") or "0")
        if need_s == 0 and need_f == 0:
            tasks.add(task)
    return tasks


def _scan_records(root: Path, source_root: str) -> list[TrialRecord]:
    rows: list[TrialRecord] = []
    if not root.exists():
        return rows

    for result_path in root.rglob("result.json"):
        payload = _read_json(result_path)
        if payload is None:
            continue

        task_name = payload.get("task_name")
        trial_name = payload.get("trial_name")
        if not isinstance(task_name, str) or not isinstance(trial_name, str):
            continue

        reward = _extract_reward(payload)
        exception_type = _extract_exception_type(payload)

        trial_dir = result_path.parent
        codex_txt = _discover_codex_file(trial_dir)
        trajectory_json = _discover_trajectory_file(trial_dir)
        gemini_cli_txt = _discover_gemini_cli_file(trial_dir)

        success = bool(reward is not None and reward >= 1.0 and not exception_type)
        evidence_exists = (codex_txt is not None) or (trajectory_json is not None) or (gemini_cli_txt is not None)
        status = _trial_status(success=success, evidence_exists=evidence_exists)

        try:
            mtime = result_path.stat().st_mtime
        except Exception:
            mtime = 0.0

        rows.append(
            TrialRecord(
                task_name=task_name,
                trial_name=trial_name,
                status=status,
                reward=reward,
                exception_type=exception_type,
                result_json=result_path.resolve(),
                codex_txt=codex_txt,
                trajectory_json=trajectory_json,
                gemini_cli_txt=gemini_cli_txt,
                source_root=source_root,
                mtime=mtime,
            )
        )

    return rows


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    keep = max(16, max_chars - 12)
    return text[:keep].rstrip() + " [TRUNCATED]"


def _parse_codex_workflow(codex_path: Path, max_output_chars: int) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for raw in codex_path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("type") != "item.completed":
            continue
        item = obj.get("item")
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "agent_message":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                steps.append(
                    {
                        "step_index": len(steps) + 1,
                        "step_type": "agent_message",
                        "agent_message": text.strip(),
                    }
                )

        elif item_type == "command_execution":
            cmd = item.get("command")
            if not isinstance(cmd, str) or not cmd.strip():
                continue
            output = item.get("aggregated_output")
            if not isinstance(output, str):
                output = ""
            out = _truncate_text(output.strip(), max_output_chars)

            step: dict[str, Any] = {
                "step_index": len(steps) + 1,
                "step_type": "command_execution",
                "command": cmd.strip(),
                "output": out,
            }
            if isinstance(item.get("exit_code"), int):
                step["exit_code"] = int(item["exit_code"])
            if isinstance(item.get("status"), str):
                step["exec_status"] = item["status"].strip()
            steps.append(step)

    return steps


def _read_json_any(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _extract_exit_code(text: str) -> int | None:
    match = re.search(r"(?:^|\n)Exit Code:\s*(-?\d+)\b", str(text or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _parse_trajectory_workflow(trajectory_path: Path, max_output_chars: int) -> list[dict[str, Any]]:
    payload = _read_json_any(trajectory_path)
    if not isinstance(payload, dict):
        return []

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        return []

    steps: list[dict[str, Any]] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            continue

        source = str(raw_step.get("source") or "").strip().lower()
        message = raw_step.get("message")
        if source in {"agent", "assistant"} and isinstance(message, str) and message.strip():
            steps.append(
                {
                    "step_index": len(steps) + 1,
                    "step_type": "agent_message",
                    "agent_message": message.strip(),
                }
            )

        results_by_call: dict[str, str] = {}
        observation = raw_step.get("observation")
        if isinstance(observation, dict):
            results = observation.get("results")
            if isinstance(results, list):
                for idx, result in enumerate(results):
                    if not isinstance(result, dict):
                        continue
                    key = str(result.get("source_call_id") or idx)
                    content = result.get("content")
                    if isinstance(content, str):
                        results_by_call[key] = content

        tool_calls = raw_step.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue

        for idx, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                continue

            call_id = str(call.get("tool_call_id") or idx)
            function_name = str(call.get("function_name") or call.get("name") or "").strip()
            arguments = call.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {"raw": arguments}
            if not isinstance(arguments, dict):
                arguments = {}

            command = arguments.get("command")
            if not isinstance(command, str) or not command.strip():
                command = function_name
                if arguments:
                    command = f"{command} {json.dumps(arguments, ensure_ascii=False, sort_keys=True)}".strip()
            if not command:
                continue

            output = results_by_call.get(call_id)
            if output is None and len(results_by_call) == 1:
                output = next(iter(results_by_call.values()))
            if not isinstance(output, str):
                output = ""

            out = _truncate_text(output.strip(), max_output_chars)
            step: dict[str, Any] = {
                "step_index": len(steps) + 1,
                "step_type": "command_execution",
                "command": command.strip(),
                "output": out,
            }
            exit_code = _extract_exit_code(output)
            if exit_code is not None:
                step["exit_code"] = exit_code
                step["exec_status"] = "completed" if exit_code == 0 else "failed"
            steps.append(step)

    return steps


def _parse_gemini_cli_workflow(gemini_path: Path, max_output_chars: int) -> list[dict[str, Any]]:
    text = gemini_path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text)
    text = re.sub(r"Both GOOGLE_API_KEY and GEMINI_API_KEY are set\. Using GOOGLE_API_KEY\.", "", text)
    text = re.sub(r"Warning: 256-color support not detected\.[^\n]*", "", text)
    text = re.sub(r"YOLO mode is enabled\.[^\n]*", "", text)
    text = re.sub(r"Ripgrep is not available\.[^\n]*", "", text)
    text = re.sub(r"AQ[A-Za-z0-9_-]{20,}", "AQ<redacted>", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    return [
        {
            "step_index": 1,
            "step_type": "agent_message",
            "agent_message": _truncate_text(text, max_output_chars),
        }
    ]


def _parse_trial_workflow(rec: TrialRecord, max_output_chars: int) -> tuple[list[dict[str, Any]], str | None]:
    if rec.codex_txt is not None and rec.codex_txt.is_file():
        steps = _parse_codex_workflow(rec.codex_txt, max_output_chars=max_output_chars)
        if steps:
            return steps, "codex_txt"
    if rec.trajectory_json is not None and rec.trajectory_json.is_file():
        steps = _parse_trajectory_workflow(rec.trajectory_json, max_output_chars=max_output_chars)
        if steps:
            return steps, "trajectory_json"
        steps = _parse_gemini_cli_workflow(rec.trajectory_json, max_output_chars=max_output_chars)
        if steps:
            return steps, "trajectory_text"
    if rec.gemini_cli_txt is not None and rec.gemini_cli_txt.is_file():
        steps = _parse_gemini_cli_workflow(rec.gemini_cli_txt, max_output_chars=max_output_chars)
        if steps:
            return steps, "gemini_cli_txt"
    return [], None


def _build_key(rec: TrialRecord) -> str:
    return f"{rec.task_name}::{rec.trial_name}::{rec.result_json}"


def _dedup(records: list[TrialRecord]) -> list[TrialRecord]:
    best: dict[str, TrialRecord] = {}
    for rec in records:
        key = _build_key(rec)
        prev = best.get(key)
        if prev is None or rec.mtime > prev.mtime:
            best[key] = rec
    return list(best.values())


def _pick(records: list[TrialRecord], need: int, rng: random.Random) -> list[TrialRecord]:
    if need <= 0:
        return []
    if len(records) <= need:
        return list(records)
    return rng.sample(records, need)


PATH_RE = re.compile(r"(?<![A-Za-z0-9_])(?:/[A-Za-z0-9._~@%+:-]+)+")
TRIAL_ID_RE = re.compile(r"\b([a-z0-9-]+__)[A-Za-z0-9]{6,}\b")
HOST_PORT_RE = re.compile(r"\b(localhost|127\.0\.0\.1|0\.0\.0\.0):([1-9][0-9]{1,4})\b")
FLAG_PORT_RE = re.compile(r"(?i)(--port(?:=|\s+)|-p\s+|port(?:\s+|=))([1-9][0-9]{1,4})\b")


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def _is_likely_progress_message(msg: str) -> bool:
    m = msg.lower()
    progress_markers = [
        "still running",
        "still in progress",
        "waiting",
        "installation is in progress",
        "installing",
        "i'm waiting",
        "i am waiting",
        "no blockers so far",
        "once it completes",
        "next i",
        "then i",
    ]
    return any(k in m for k in progress_markers)


def _replace_paths(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        low = token.lower()
        if "/app" in low:
            return "{workdir_path}"
        if "/tmp" in low:
            return "{tmp_path}"
        if "/root" in low or "/home" in low:
            return "{home_path}"
        return "{abs_path}"

    return PATH_RE.sub(repl, text)


def _replace_trial_ids(text: str) -> str:
    return TRIAL_ID_RE.sub(r"\1{trial_id}", text)


def _replace_ports(text: str) -> str:
    def host_repl(match: re.Match[str]) -> str:
        host = match.group(1)
        p = int(match.group(2))
        if 1 <= p <= 65535:
            return f"{host}:{{port}}"
        return match.group(0)

    def flag_repl(match: re.Match[str]) -> str:
        prefix = match.group(1)
        p = int(match.group(2))
        if 1 <= p <= 65535:
            return f"{prefix}{{port}}"
        return match.group(0)

    out = HOST_PORT_RE.sub(host_repl, text)
    out = FLAG_PORT_RE.sub(flag_repl, out)
    return out


def _clean_text(text: str) -> str:
    s = text
    s = _replace_paths(s)
    s = _replace_trial_ids(s)
    s = _replace_ports(s)
    return s


def _clean_workflow_steps(steps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    cleaned: list[dict[str, Any]] = []
    removed_consecutive_agent_msgs = 0

    for st in steps:
        t = st.get("step_type")
        if t == "command_execution":
            cmd = _clean_text(str(st.get("command") or ""))
            out = _clean_text(str(st.get("output") or ""))
            c = dict(st)
            c["command"] = cmd
            c["output"] = out
            cleaned.append(c)
        elif t == "agent_message":
            msg = _clean_text(str(st.get("agent_message") or ""))
            msg_sig = _normalize_whitespace(msg)
            if cleaned and cleaned[-1].get("step_type") == "agent_message":
                prev_msg = _normalize_whitespace(str(cleaned[-1].get("agent_message") or ""))
                if msg_sig == prev_msg or _is_likely_progress_message(msg):
                    removed_consecutive_agent_msgs += 1
                    continue
            if _is_likely_progress_message(msg):
                removed_consecutive_agent_msgs += 1
                continue
            c = dict(st)
            c["agent_message"] = msg
            cleaned.append(c)
        else:
            cleaned.append(dict(st))

    for i, st in enumerate(cleaned, start=1):
        st["step_index"] = i

    stats = {
        "removed_consecutive_agent_messages": removed_consecutive_agent_msgs,
        "input_step_count": len(steps),
        "output_step_count": len(cleaned),
    }
    return cleaned, stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export task workflows from status-qualified mixed traces.")
    p.add_argument(
        "--status-log",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "terminal-bench-2" / "mixed-collect" / "status.log",
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "terminal-bench-2" / "raw",
    )
    p.add_argument(
        "--collect-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "terminal-bench-2" / "mixed-collect",
    )
    p.add_argument("--success-per-task", type=int, default=5)
    p.add_argument("--failure-per-task", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-output-chars", type=int, default=4000)
    p.add_argument("--enable-cleaning", action="store_true", help="Enable workflow cleaning and placeholder rewriting.")
    p.add_argument(
        "--workflow-out",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "terminal-bench-2" / "mixed-collect" / "workflows_5s5f_from_status.json",
    )
    p.add_argument(
        "--metadata-out",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "terminal-bench-2" / "mixed-collect" / "workflows_5s5f_from_status.meta.json",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    status_log = args.status_log.resolve()
    raw_root = args.raw_root.resolve()
    collect_root = args.collect_root.resolve()
    workflow_out = args.workflow_out.resolve()
    metadata_out = args.metadata_out.resolve()

    if not status_log.is_file():
        raise RuntimeError(f"status.log not found: {status_log}")
    if not raw_root.exists():
        raise RuntimeError(f"raw root not found: {raw_root}")
    if not collect_root.exists():
        raise RuntimeError(f"collect root not found: {collect_root}")

    qualified_tasks = _load_qualified_tasks(status_log)

    raw_records = _scan_records(raw_root, "raw")
    collect_records = _scan_records(collect_root, "collect")
    all_records = _dedup([*raw_records, *collect_records])

    by_task: dict[str, dict[str, list[TrialRecord]]] = {}
    for rec in all_records:
        if rec.task_name not in qualified_tasks:
            continue
        bucket = by_task.setdefault(rec.task_name, {"success": [], "failure": []})
        if rec.status == "success":
            bucket["success"].append(rec)
        elif rec.status == "failure":
            bucket["failure"].append(rec)

    rng = random.Random(int(args.seed))

    workflows_payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": {
            "task_filter": "status.log entries with need_success=0 and need_failure=0",
            "sampling": "random",
            "seed": int(args.seed),
            "success_per_task": int(args.success_per_task),
            "failure_per_task": int(args.failure_per_task),
        },
        "cleaning": {
            "enabled": bool(args.enable_cleaning),
            "rules": {
                "drop_consecutive_progress_agent_message": True,
                "replace_absolute_paths": True,
                "replace_trial_ids": True,
                "replace_ports": True,
            },
        },
        "tasks": [],
    }

    metadata: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "status_log": str(status_log),
            "raw_root": str(raw_root),
            "collect_root": str(collect_root),
        },
        "rules": {
            "qualified_task_rule": "need_success==0 and need_failure==0",
            "trial_status_rule": "success if reward>=1 and no exception_type; else failure if any trajectory evidence exists; else runtime_error",
            "workflow_step_rule": "prefer codex.txt parser; fallback to trajectory.json parser using agent message and tool call observations",
            "output_truncation": int(args.max_output_chars),
            "sampling": "random",
            "seed": int(args.seed),
            "success_per_task": int(args.success_per_task),
            "failure_per_task": int(args.failure_per_task),
            "insufficient_pool_behavior": "skip_task",
            "cleaning_enabled": bool(args.enable_cleaning),
        },
        "scan_counts": {
            "raw_records": len(raw_records),
            "collect_records": len(collect_records),
            "dedup_total_records": len(all_records),
            "qualified_tasks_from_status": len(qualified_tasks),
        },
        "cleaning_stats": {
            "trials_cleaned": 0,
            "removed_consecutive_agent_messages": 0,
            "input_steps_total": 0,
            "output_steps_total": 0,
        },
        "selected_task_count": 0,
        "skipped_tasks": [],
        "selected_trials": [],
    }

    for task_name in sorted(qualified_tasks):
        bucket = by_task.get(task_name, {"success": [], "failure": []})
        succ_pool = list(bucket.get("success", []))
        fail_pool = list(bucket.get("failure", []))

        if len(succ_pool) < int(args.success_per_task) or len(fail_pool) < int(args.failure_per_task):
            metadata["skipped_tasks"].append(
                {
                    "task_name": task_name,
                    "reason": "insufficient_success_or_failure_pool",
                    "success_pool": len(succ_pool),
                    "failure_pool": len(fail_pool),
                    "required_success": int(args.success_per_task),
                    "required_failure": int(args.failure_per_task),
                }
            )
            continue

        picked_success = _pick(succ_pool, int(args.success_per_task), rng)
        picked_failure = _pick(fail_pool, int(args.failure_per_task), rng)
        picked = [*picked_success, *picked_failure]

        task_trials: list[dict[str, Any]] = []
        for rec in picked:
            steps, workflow_source = _parse_trial_workflow(rec, max_output_chars=int(args.max_output_chars))
            if not steps:
                continue

            clean_info: dict[str, Any] | None = None
            if args.enable_cleaning:
                cleaned_steps, cstats = _clean_workflow_steps(steps)
                clean_info = cstats
                steps = cleaned_steps
                metadata["cleaning_stats"]["trials_cleaned"] += 1
                metadata["cleaning_stats"]["removed_consecutive_agent_messages"] += int(cstats["removed_consecutive_agent_messages"])
                metadata["cleaning_stats"]["input_steps_total"] += int(cstats["input_step_count"])
                metadata["cleaning_stats"]["output_steps_total"] += int(cstats["output_step_count"])

            trial_payload = {
                "task_name": rec.task_name,
                "trial_name": rec.trial_name,
                "status": rec.status,  # success | failure
                "exception_type": rec.exception_type,
                "reward": rec.reward,
                "source_root": rec.source_root,
                "result_json": str(rec.result_json),
                "codex_txt": str(rec.codex_txt) if rec.codex_txt else None,
                "trajectory_json": str(rec.trajectory_json) if rec.trajectory_json else None,
                "gemini_cli_txt": str(rec.gemini_cli_txt) if rec.gemini_cli_txt else None,
                "workflow_source": workflow_source,
                "workflow": steps,
            }
            if clean_info is not None:
                trial_payload["cleaning"] = clean_info

            task_trials.append(trial_payload)
            metadata["selected_trials"].append(
                {
                    "task_name": rec.task_name,
                    "trial_name": rec.trial_name,
                    "status": rec.status,
                    "source_root": rec.source_root,
                    "result_json": str(rec.result_json),
                    "codex_txt": str(rec.codex_txt) if rec.codex_txt else None,
                    "trajectory_json": str(rec.trajectory_json) if rec.trajectory_json else None,
                    "gemini_cli_txt": str(rec.gemini_cli_txt) if rec.gemini_cli_txt else None,
                    "workflow_source": workflow_source,
                    "workflow_step_count": len(steps),
                }
            )

        s_cnt = sum(1 for t in task_trials if t.get("status") == "success")
        f_cnt = sum(1 for t in task_trials if t.get("status") == "failure")
        if s_cnt < int(args.success_per_task) or f_cnt < int(args.failure_per_task):
            metadata["skipped_tasks"].append(
                {
                    "task_name": task_name,
                    "reason": "selected_trials_missing_or_unparseable_workflow",
                    "picked_success": len(picked_success),
                    "picked_failure": len(picked_failure),
                    "parsed_success": s_cnt,
                    "parsed_failure": f_cnt,
                }
            )
            metadata["selected_trials"] = [x for x in metadata["selected_trials"] if x["task_name"] != task_name]
            continue

        workflows_payload["tasks"].append(
            {
                "task_name": task_name,
                "required_success": int(args.success_per_task),
                "required_failure": int(args.failure_per_task),
                "trials": task_trials,
            }
        )

    metadata["selected_task_count"] = len(workflows_payload["tasks"])

    workflow_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    workflow_out.write_text(json.dumps(workflows_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_out.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "workflow_out": str(workflow_out),
                "metadata_out": str(metadata_out),
                "qualified_tasks": len(qualified_tasks),
                "selected_tasks": metadata["selected_task_count"],
                "skipped_tasks": len(metadata["skipped_tasks"]),
                "selected_trials": len(metadata["selected_trials"]),
                "cleaning_enabled": bool(args.enable_cleaning),
                "cleaning_stats": metadata.get("cleaning_stats"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
