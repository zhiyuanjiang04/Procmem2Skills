#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROVIDER_DEFAULT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GOOGLE_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "uniapi": "UNIAPI_API_KEY",
}


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


def _normalize_workflow_steps_to_steps(workflow_steps: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(workflow_steps, list):
        return out
    for step in workflow_steps:
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("step_type") or "").strip().lower()
        if step_type == "command_execution":
            out.append(
                {
                    "command": str(step.get("command") or ""),
                    "result": str(step.get("output") or ""),
                    "exit_code": step.get("exit_code"),
                    "exec_status": step.get("exec_status"),
                }
            )
        elif step_type == "agent_message":
            msg = step.get("agent_message")
            if isinstance(msg, str) and msg.strip():
                out.append({"agent_message": msg.strip()})
    return out


def _load_trace_records(workflow_file: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(workflow_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to parse workflow input: {workflow_file}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"workflow input must be a JSON object: {workflow_file}")

    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise RuntimeError(f"workflow input missing 'tasks' list: {workflow_file}")

    rows: list[dict[str, Any]] = []
    for task_row in tasks:
        if not isinstance(task_row, dict):
            continue
        task_name = str(task_row.get("task_name") or "").strip()
        if not task_name:
            continue
        trials = task_row.get("trials")
        if not isinstance(trials, list):
            continue
        for trial in trials:
            if not isinstance(trial, dict):
                continue
            trial_name = str(trial.get("trial_name") or "").strip()
            if not trial_name:
                continue
            rec = dict(trial)
            rec["task_name"] = task_name
            rec["trial_name"] = trial_name
            rec["status"] = str(rec.get("status") or "runtime_error").strip().lower()
            if "steps" not in rec:
                rec["steps"] = _normalize_workflow_steps_to_steps(rec.get("workflow"))
            rows.append(rec)
    return rows


def _record_for_prompt(rec: dict[str, Any], *, include_status: bool) -> dict[str, Any]:
    # Keep only fields needed for induction; drop heavy raw payloads.
    out: dict[str, Any] = {
        "task_name": str(rec.get("task_name") or ""),
        "trial_name": str(rec.get("trial_name") or ""),
    }
    if include_status:
        out["status"] = str(rec.get("status") or "runtime_error")
        exc = rec.get("exception_type")
        if exc is not None and str(exc).strip():
            out["exception_type"] = str(exc)
    steps = rec.get("steps")
    out["steps"] = steps if isinstance(steps, list) else []
    return out


def _build_user_message(records: list[dict[str, Any]], *, hint_mode: str) -> str:
    include_status = hint_mode == "with-status"
    lines = ["Here are N execution traces for a task. Each trace shows the agent's actions and observations.", ""]
    for idx, rec in enumerate(records, start=1):
        prompt_rec = _record_for_prompt(rec, include_status=include_status)
        if include_status:
            status = str(prompt_rec.get("status") or "runtime_error").upper()
            if status == "RUNTIME_ERROR":
                status = "FAILURE"
            lines.append(f"[Trace {idx} - {status}]")
        else:
            lines.append(f"[Trace {idx}]")
        lines.append(json.dumps(prompt_rec, ensure_ascii=False, indent=2))
        lines.append("")
    lines.append("Generate a reusable skill from these traces.")
    return "\n".join(lines)


def _trim_text(value: str, max_chars: int) -> str:
    text = str(value or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _ensure_skill_frontmatter(skill_content: str, expected_name: str) -> tuple[str, dict[str, Any]]:
    expected = _normalize_slug(expected_name, default="task-skill", max_len=80)
    text = str(skill_content or "").strip()
    if not text:
        text = "# Skill\n"

    lines = text.splitlines()
    info: dict[str, Any] = {
        "expected_name": expected,
        "had_frontmatter": False,
        "had_name_field": False,
        "original_name": None,
        "changed": False,
    }

    if lines and lines[0].strip() == "---":
        end_idx = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i
                break
        if end_idx is not None:
            info["had_frontmatter"] = True
            fm_lines = lines[1:end_idx]
            body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
            new_fm: list[str] = []
            name_found = False
            for line in fm_lines:
                if re.match(r"^\s*name\s*:", line):
                    name_found = True
                    info["had_name_field"] = True
                    original = line.split(":", 1)[1].strip().strip('"').strip("'")
                    info["original_name"] = original
                    if original != expected:
                        info["changed"] = True
                    new_fm.append(f"name: {expected}")
                else:
                    new_fm.append(line)
            if not name_found:
                new_fm.insert(0, f"name: {expected}")
                info["changed"] = True
            new_text = "---\n" + "\n".join(new_fm).rstrip() + "\n---\n\n"
            if body:
                new_text += body.rstrip() + "\n"
            return new_text, info

    info["changed"] = True
    rebuilt = f"---\nname: {expected}\ndescription: Generated skill for {expected}.\n---\n\n{text.rstrip()}\n"
    return rebuilt, info


def _compress_record_for_prompt(record: dict[str, Any], *, max_steps: int, max_cmd_chars: int, max_result_chars: int, max_agent_msg_chars: int) -> dict[str, Any]:
    rec = dict(record)
    steps_in = rec.get("steps")
    if not isinstance(steps_in, list):
        return rec

    kept = steps_in[: max(0, int(max_steps))]
    steps_out: list[dict[str, Any]] = []
    for st in kept:
        if not isinstance(st, dict):
            continue
        out: dict[str, Any] = {}
        cmd = st.get("command")
        if cmd is not None:
            out["command"] = _trim_text(str(cmd), max_cmd_chars)
        res = st.get("result")
        if res is not None:
            out["result"] = _trim_text(str(res), max_result_chars)
        msg = st.get("agent_message")
        if msg is not None:
            out["agent_message"] = _trim_text(str(msg), max_agent_msg_chars)
        if "exit_code" in st:
            out["exit_code"] = st.get("exit_code")
        if "exec_status" in st:
            out["exec_status"] = st.get("exec_status")
        if out:
            steps_out.append(out)

    rec["steps"] = steps_out
    rec["_prompt_compression"] = {
        "original_steps": len(steps_in),
        "kept_steps": len(steps_out),
        "max_steps": max_steps,
        "max_cmd_chars": max_cmd_chars,
        "max_result_chars": max_result_chars,
        "max_agent_msg_chars": max_agent_msg_chars,
    }
    return rec


def _compress_records_for_prompt(records: list[dict[str, Any]], *, max_steps: int, max_cmd_chars: int, max_result_chars: int, max_agent_msg_chars: int) -> list[dict[str, Any]]:
    return [
        _compress_record_for_prompt(
            r,
            max_steps=max_steps,
            max_cmd_chars=max_cmd_chars,
            max_result_chars=max_result_chars,
            max_agent_msg_chars=max_agent_msg_chars,
        )
        for r in records
    ]


def _parse_conditions(raw: str) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for token in [x.strip() for x in raw.split(",") if x.strip()]:
        m = re.fullmatch(r"(\d+)s(\d+)f", token)
        if not m:
            raise RuntimeError(f"invalid condition token: {token}")
        s = int(m.group(1))
        f = int(m.group(2))
        if s + f <= 0:
            raise RuntimeError(f"empty condition not allowed: {token}")
        out.append((s, f, token))
    if not out:
        raise RuntimeError("no skill conditions parsed")
    return out


def _infer_benchmark_from_path(workflow_input: Path) -> str:
    raw = str(workflow_input).lower()
    candidates = [
        ("terminalbenchpro", ["terminal-bench-pro", "terminalbenchpro", "tbpro"]),
        ("terminalbench2", ["terminal-bench-2", "terminal-bench-2-0", "terminalbench2", "tb2"]),
        ("skillsbench", ["skillsbench", "skills-bench", "sb"]),
    ]
    for name, keys in candidates:
        if any(k in raw for k in keys):
            return name
    return "unknown-benchmark"


def _resolve_benchmark_name(args: argparse.Namespace, workflow_input: Path) -> str:
    bench = str(getattr(args, "benchmark", "") or "").strip()
    if bench:
        return _normalize_slug(bench, default="unknown-benchmark", max_len=64)
    sub = str(getattr(args, "skills_subdir", "") or "").strip()
    if sub:
        return _normalize_slug(sub, default="unknown-benchmark", max_len=64)
    return _infer_benchmark_from_path(workflow_input)


def _pick_records(successes: list[dict[str, Any]], failures: list[dict[str, Any]], s_need: int, f_need: int, seed: int, salt: str) -> list[dict[str, Any]]:
    digest = hashlib.sha256(f"{seed}:{salt}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    picked: list[dict[str, Any]] = []
    if s_need > 0:
        picked.extend(rng.sample(successes, s_need))
    if f_need > 0:
        picked.extend(rng.sample(failures, f_need))
    rng.shuffle(picked)
    return picked


def _provider_headers(provider: str, api_key: str) -> dict[str, str]:
    if provider == "claude":
        return {"x-api-key": api_key, "Authorization": f"Bearer {api_key}"}
    return {"Authorization": f"Bearer {api_key}"}


def _chat_completion(*, base_url: str, provider: str, api_key: str, model: str, system_prompt: str, user_message: str, timeout_sec: int = 120, max_output_tokens: int = 8192) -> str:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "pm2s-skill-generator/1.1",
        **_provider_headers(provider, api_key),
    }

    model_l = str(model or "").lower()
    use_responses = provider == "openai" and ("codex" in model_l or model_l.startswith("gpt-5"))

    if use_responses:
        url = base_url.rstrip("/") + "/responses"
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_message}]},
            ],
            "temperature": 0.2,
            "max_output_tokens": max(256, int(max_output_tokens)),
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(f"LLM HTTPError {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM URLError: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(f"LLM timeout: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM invalid JSON response: {raw[:500]}") from exc

        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        chunks: list[str] = []
        output = data.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "message":
                    content = item.get("content")
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and isinstance(part.get("text"), str):
                                chunks.append(part.get("text") or "")
                elif isinstance(item.get("text"), str):
                    chunks.append(item.get("text") or "")

        merged = "".join(chunks).strip()
        if merged:
            return merged
        raise RuntimeError(f"LLM response missing output text: {raw[:500]}")

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.2,
        "max_tokens": max(256, int(max_output_tokens)),
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"LLM HTTPError {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM URLError: {exc}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"LLM timeout: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM invalid JSON response: {raw[:500]}") from exc

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice0 = choices[0] if isinstance(choices[0], dict) else {}
        msg = choice0.get("message") if isinstance(choice0, dict) else None
        finish_reason = str(choice0.get("finish_reason") or "") if isinstance(choice0, dict) else ""
        if isinstance(msg, dict) and isinstance(msg.get("content"), str) and msg.get("content").strip():
            return msg["content"]
        if finish_reason == "length":
            raise RuntimeError(f"LLM finish_reason=length with empty content: {raw[:500]}")
    raise RuntimeError(f"LLM response missing choices/message: {raw[:500]}")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate skills from workflow JSON records.")
    parser.add_argument("--workflow-input", type=Path, required=True, help="Single workflow JSON file (tasks[]/trials[]).")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--skills-subdir",
        default="",
        help="Optional extra subdir under pipeline-v1/skills (e.g. skillsbench).",
    )
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=sorted(PROVIDER_DEFAULT_KEY_ENV.keys()), required=True)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--skill-creator-model", required=True)
    parser.add_argument("--system-prompt-file", type=Path, required=True)
    parser.add_argument(
        "--hint-mode",
        choices=["with-status", "no-hint"],
        default="with-status",
        help="with-status: include per-trace success/failure labels; no-hint: hide status and let model infer.",
    )
    parser.add_argument(
        "--benchmark",
        default="",
        help="Benchmark name for output layout, e.g. skillsbench / terminalbench2 / terminalbenchpro.",
    )
    parser.add_argument("--conditions", default="5s0f,4s1f,3s2f,2s3f,1s4f,0s5f")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Skip generation when target SKILL.md already exists.")
    parser.add_argument("--max-prompt-steps-per-trace", type=int, default=120)
    parser.add_argument("--max-prompt-command-chars", type=int, default=600)
    parser.add_argument("--max-prompt-result-chars", type=int, default=2400)
    parser.add_argument("--max-prompt-agent-message-chars", type=int, default=1200)
    parser.add_argument("--chat-timeout-sec", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-sec", type=float, default=3.0)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workflow_input = args.workflow_input.resolve()
    if not workflow_input.is_file():
        raise RuntimeError(f"workflow-input missing: {workflow_input}")

    provider = str(args.provider).strip().lower()
    api_key_env = args.api_key_env or PROVIDER_DEFAULT_KEY_ENV[provider]
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"missing API key env: {api_key_env}")

    if provider in {"google", "claude"} and not args.base_url:
        raise RuntimeError(f"provider={provider} requires --base-url")

    default_base_by_provider = {
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "uniapi": "https://api.uniapi.io/v1",
    }
    default_base = default_base_by_provider.get(provider, "https://api.openai.com/v1")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("UNIAPI_BASE_URL") or default_base

    system_prompt = args.system_prompt_file.resolve().read_text(encoding="utf-8")
    conditions = _parse_conditions(args.conditions)

    records = _load_trace_records(workflow_input)
    if not records:
        raise RuntimeError(f"no usable workflow trials found in: {workflow_input}")
    by_task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for rec in records:
        task = str(rec.get("task_name") or "").strip()
        if not task:
            continue
        status = str(rec.get("status") or "runtime_error").strip().lower()
        if status not in {"success", "failure", "runtime_error"}:
            status = "runtime_error"
        bucket = by_task.setdefault(task, {"success": [], "failure": []})
        if status == "success":
            bucket["success"].append(rec)
        elif status == "failure":
            bucket["failure"].append(rec)

    benchmark_name = _resolve_benchmark_name(args, workflow_input)

    if str(args.hint_mode).strip().lower() == "no-hint":
        skills_root = args.output_root.resolve() / "skills" / "no-hint" / benchmark_name
    else:
        agent_model = _normalize_slug(f"{args.agent}-{args.model.split('/')[-1]}", default="agent-model")
        pipeline_root = args.output_root.resolve() / agent_model / "pipeline-v1"
        skills_root = pipeline_root / "skills"
        if str(args.skills_subdir).strip():
            skills_root = skills_root / _normalize_slug(str(args.skills_subdir).strip(), default="skills")

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workflow_input": str(workflow_input),
        "skills_root": str(skills_root),
        "hint_mode": str(args.hint_mode),
        "benchmark": benchmark_name,
        "system_prompt_file": str(args.system_prompt_file.resolve()),
        "conditions": [c[2] for c in conditions],
        "cross_task": "TODO_NOT_IMPLEMENTED",
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
        "max_prompt_steps_per_trace": int(args.max_prompt_steps_per_trace),
        "max_prompt_command_chars": int(args.max_prompt_command_chars),
        "max_prompt_result_chars": int(args.max_prompt_result_chars),
        "max_prompt_agent_message_chars": int(args.max_prompt_agent_message_chars),
        "chat_timeout_sec": int(args.chat_timeout_sec),
        "max_retries": int(args.max_retries),
        "retry_backoff_sec": float(args.retry_backoff_sec),
        "max_output_tokens": int(args.max_output_tokens),
        "tasks": {},
    }

    for task_name in sorted(by_task.keys()):
        succ = by_task[task_name]["success"]
        fail = by_task[task_name]["failure"]
        task_log: dict[str, Any] = {
            "success_pool": len(succ),
            "failure_pool": len(fail),
            "conditions": {},
        }
        for s_need, f_need, label in conditions:
            cond_log: dict[str, Any] = {
                "required_success": s_need,
                "required_failure": f_need,
                "generated": False,
            }
            if len(succ) < s_need or len(fail) < f_need:
                cond_log["status"] = "insufficient_traces"
                task_log["conditions"][label] = cond_log
                continue

            picked = _pick_records(succ, fail, s_need, f_need, args.seed, salt=f"{task_name}:{label}")
            user_message = _build_user_message(picked, hint_mode=str(args.hint_mode))

            skill_dir = skills_root / label / _normalize_slug(task_name, default="task")
            skill_path = skill_dir / "SKILL.md"
            manifest_path = skill_dir / "generation_manifest.json"

            if args.resume and skill_path.is_file():
                cond_log["status"] = "skipped_resume_exists"
                cond_log["generated"] = False
                cond_log["resumed"] = True
                cond_log["skill_path"] = str(skill_path)
                if manifest_path.is_file():
                    cond_log["manifest_path"] = str(manifest_path)
                task_log["conditions"][label] = cond_log
                continue

            base_steps = max(1, int(args.max_prompt_steps_per_trace))
            base_cmd_chars = max(50, int(args.max_prompt_command_chars))
            base_result_chars = max(100, int(args.max_prompt_result_chars))
            base_agent_msg_chars = max(80, int(args.max_prompt_agent_message_chars))

            current_steps = base_steps
            current_cmd_chars = base_cmd_chars
            current_result_chars = base_result_chars
            current_agent_msg_chars = base_agent_msg_chars

            # Always cap trace payload for prompt stability.
            used_records = _compress_records_for_prompt(
                picked,
                max_steps=current_steps,
                max_cmd_chars=current_cmd_chars,
                max_result_chars=current_result_chars,
                max_agent_msg_chars=current_agent_msg_chars,
            )
            used_message = _build_user_message(used_records, hint_mode=str(args.hint_mode))
            compress_attempted = True

            if args.dry_run:
                skill_content = (
                    "---\n"
                    f"name: {_normalize_slug(task_name, default='task-skill')}\n"
                    "description: dry-run placeholder\n"
                    "---\n\n"
                    f"# {_normalize_slug(task_name, default='Task Skill')}\n\n"
                    "Dry run only; no LLM call executed.\n"
                )
            else:
                last_error = ""
                attempt_used = 0
                for attempt in range(1, max(1, int(args.max_retries)) + 1):
                    attempt_used = attempt
                    try:
                        skill_content = _chat_completion(
                            base_url=base_url,
                            provider=provider,
                            api_key=api_key,
                            model=args.skill_creator_model,
                            system_prompt=system_prompt,
                            user_message=used_message,
                            timeout_sec=max(1, int(args.chat_timeout_sec)),
                            max_output_tokens=max(256, int(args.max_output_tokens)),
                        )
                        last_error = ""
                        break
                    except RuntimeError as exc:
                        last_error = str(exc)
                        ll = last_error.lower()
                        is_ctx = ("maximum context length" in ll) or ("400000 tokens" in ll) or ("finish_reason=length" in ll) or (("missing choices/message" in ll) and ("finish_reason\":\"length" in ll))
                        if is_ctx:
                            factor = 0.55 if ("finish_reason=length" in ll or "missing choices/message" in ll) else 0.7
                            next_steps = max(4, int(current_steps * factor))
                            next_cmd_chars = max(30, int(current_cmd_chars * factor))
                            next_result_chars = max(120, int(current_result_chars * factor))
                            next_agent_msg_chars = max(60, int(current_agent_msg_chars * factor))

                            if (
                                next_steps,
                                next_cmd_chars,
                                next_result_chars,
                                next_agent_msg_chars,
                            ) == (
                                current_steps,
                                current_cmd_chars,
                                current_result_chars,
                                current_agent_msg_chars,
                            ):
                                next_steps = max(4, current_steps - 4)
                                next_cmd_chars = max(30, current_cmd_chars - 20)
                                next_result_chars = max(120, current_result_chars - 80)
                                next_agent_msg_chars = max(60, current_agent_msg_chars - 30)

                            current_steps = next_steps
                            current_cmd_chars = next_cmd_chars
                            current_result_chars = next_result_chars
                            current_agent_msg_chars = next_agent_msg_chars

                            used_records = _compress_records_for_prompt(
                                picked,
                                max_steps=current_steps,
                                max_cmd_chars=current_cmd_chars,
                                max_result_chars=current_result_chars,
                                max_agent_msg_chars=current_agent_msg_chars,
                            )
                            used_message = _build_user_message(used_records, hint_mode=str(args.hint_mode))
                            compress_attempted = True
                        if attempt < max(1, int(args.max_retries)):
                            time.sleep(max(0.0, float(args.retry_backoff_sec)) * attempt)
                            continue
                        cond_log["status"] = "failed_after_retries"
                        cond_log["generated"] = False
                        cond_log["resumed"] = False
                        cond_log["attempt_count"] = int(attempt_used)
                        cond_log["last_error"] = last_error[:1000]
                        cond_log["prompt_chars"] = int(len(used_message))
                        cond_log["compression_applied"] = bool(compress_attempted)
                        cond_log["compression_limits"] = {
                            "max_steps": int(current_steps),
                            "max_command_chars": int(current_cmd_chars),
                            "max_result_chars": int(current_result_chars),
                            "max_agent_message_chars": int(current_agent_msg_chars),
                        }
                        task_log["conditions"][label] = cond_log
                        break
                if cond_log.get("status") == "failed_after_retries":
                    continue
                cond_log["attempt_count"] = int(attempt_used)

            skill_dir.mkdir(parents=True, exist_ok=True)
            skill_content, frontmatter_info = _ensure_skill_frontmatter(skill_content, skill_dir.name)
            skill_path.write_text(skill_content, encoding="utf-8")
            cond_manifest = {
                "task_name": task_name,
                "condition": label,
                "picked_trials": [str(r.get("trial_name") or "") for r in picked],
                "used_trials": [str(r.get("trial_name") or "") for r in used_records],
                "skill_path": str(skill_path),
                "dry_run": bool(args.dry_run),
                "resume": bool(args.resume),
                "prompt_chars": len(used_message),
                "compression_applied": bool(compress_attempted),
                "max_prompt_steps_per_trace": int(args.max_prompt_steps_per_trace),
                "max_prompt_command_chars": int(args.max_prompt_command_chars),
                "max_prompt_result_chars": int(args.max_prompt_result_chars),
                "max_prompt_agent_message_chars": int(args.max_prompt_agent_message_chars),
                "max_output_tokens": int(args.max_output_tokens),
                "effective_compression_limits": {
                    "max_steps": int(current_steps),
                    "max_command_chars": int(current_cmd_chars),
                    "max_result_chars": int(current_result_chars),
                    "max_agent_message_chars": int(current_agent_msg_chars),
                },
                "hint_mode": str(args.hint_mode),
                "benchmark": benchmark_name,
                "system_prompt_file": str(args.system_prompt_file.resolve()),
                "frontmatter": frontmatter_info,
            }
            _json_dump(manifest_path, cond_manifest)

            cond_log["status"] = "ok"
            cond_log["generated"] = True
            cond_log["resumed"] = False
            cond_log["skill_path"] = str(skill_path)
            cond_log["manifest_path"] = str(manifest_path)
            cond_log["prompt_chars"] = int(len(used_message))
            cond_log["compression_applied"] = bool(compress_attempted)
            cond_log["compression_limits"] = {
                "max_steps": int(current_steps),
                "max_command_chars": int(current_cmd_chars),
                "max_result_chars": int(current_result_chars),
                "max_agent_message_chars": int(current_agent_msg_chars),
            }
            cond_log["frontmatter"] = frontmatter_info
            task_log["conditions"][label] = cond_log

        summary["tasks"][task_name] = task_log

    summary_path = skills_root / "generation_summary.json"
    _json_dump(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
