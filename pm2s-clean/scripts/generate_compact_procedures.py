#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import generate_skills as skillgen


FORM_SPECS: dict[str, dict[str, str]] = {
    "short-plan": {
        "title": "Short Plan",
        "instruction": (
            "Write a concise pre-execution plan with 3-5 numbered steps. The plan should be based only on the "
            "task instruction and should help the future agent organize its work before execution. Keep it much "
            "less detailed than a SKILL.md file: no preconditions, no failure-mode section, no verification recipe, "
            "no examples, no code blocks, and no hidden implementation hints. Use broad execution stages such as "
            "inspect, identify, plan, implement, validate, and iterate. The plan may restate requirements already "
            "present in the task instruction, but it must not add solution facts that are not in the instruction."
        ),
    },
    "test-first": {
        "title": "Test-First Procedure",
        "instruction": (
            "Distill the workflows into a compact validation-first procedure. The procedure must make the agent identify "
            "the expected success condition, locate or create a lightweight validation check, run it before or during "
            "implementation, and rerun it before final submission. It may describe what should be validated, but it must "
            "not describe how to implement the solution. Do not include concrete commands, source files, patches, "
            "constants, package names, exact expected outputs, or known fixes from the source workflow."
        ),
    },
    "script": {
        "title": "Reusable Script-Style Command Recipe",
        "instruction": (
            "Distill the workflows into a reusable script-style command recipe. Preserve reusable command patterns, "
            "script skeletons, validation commands, and file-inspection idioms that could help solve the task family. "
            "Do not create a one-shot solution script. Do not hard-code source-run-specific parameters, absolute paths, "
            "final answers, magic constants, row counts, filenames that are not part of the task interface, or values "
            "copied from the successful run. Use placeholders such as <input_path>, <output_path>, <test_command>, "
            "or <target_file> when a value must be adapted to the current task. Explicitly tell the agent to inspect "
            "the current workspace before running or adapting any command."
        ),
    },
}

GENERATION_SOURCES = {"workflow", "instruction"}


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


def _parse_forms(raw: str) -> list[str]:
    forms = []
    for token in [x.strip().lower() for x in str(raw or "").split(",") if x.strip()]:
        slug = _normalize_slug(token, default="form")
        if slug not in FORM_SPECS:
            raise RuntimeError(f"invalid form={token!r}; allowed={sorted(FORM_SPECS)}")
        if slug not in forms:
            forms.append(slug)
    if not forms:
        raise RuntimeError("at least one form is required")
    return forms


def _build_system_prompt(form: str, *, source: str) -> str:
    spec = FORM_SPECS[form]
    if source == "workflow":
        if form == "short-plan":
            source_policy = (
                "You are a compact procedure generator. Given one or more execution traces from an agent attempting "
                "the same task, produce a short reusable plan that captures only the high-level process.\n"
                "Use the traces only as evidence about generic process order, not as a source of solution facts. "
                "Do not summarize the solution. Do not copy trace-specific discoveries. Do not preserve domain nouns "
                "from the traces unless they are already present in the benchmark task instruction."
            )
        else:
            source_policy = (
                "You are given multiple agent execution traces for the same benchmark task. Your goal is to abstract "
                "high-level, transferable procedural patterns from these traces.\n"
                "Do not summarize the solution. Do not copy trace-specific discoveries. Instead, identify reusable "
                "execution stages that would remain valid for another agent attempting the same task from scratch."
            )
    else:
        if form == "short-plan":
            source_policy = (
                "You are a compact pre-planning assistant. You will see only the benchmark task instruction. "
                "Write a short plan that a future agent can read before starting the task.\n"
                "Base the plan only on the task instruction. Do not use prior workflow knowledge, hidden solution "
                "traces, or assumptions about the correct answer."
            )
        else:
            source_policy = (
                "You will see only the task instruction. Base the guidance only on that instruction. "
                "Do not use prior workflow knowledge, hidden solution traces, or assumptions about the correct answer."
            )
    return (
        "You write compact procedural guidance for future agent runs.\n"
        "The output will be appended to a benchmark task instruction, not installed as a SKILL.md file.\n"
        "Keep the output concise, operational, and reusable. Do not mention this prompt.\n"
        "The guidance is a lightweight procedural baseline, not a direct answer.\n"
        f"{source_policy}\n\n"
        f"Target form: {spec['title']}.\n"
        f"{spec['instruction']}\n\n"
        "Output requirements:\n"
        "- Write Markdown only.\n"
        "- Start with a short heading matching the target form.\n"
        "- Prefer numbered steps or short bullet lists.\n"
        "- Produce exactly one compact procedure, not multiple alternatives.\n"
        "- Do not include fenced code unless the target form is script and the code is a reusable skeleton with placeholders.\n"
        "- Do not include final task answers, hidden diagnoses, known fixes, exact patches, exact source-run "
        "discoveries, or values that are not explicitly present in the task instruction.\n"
        "- Do not invent exact filenames, paths, package/archive names, flags, shell commands, code symbols, "
        "function names, signatures, numeric thresholds, magic constants, or literal output strings beyond those "
        "already specified by the task instruction.\n"
        "- If a detail would let an agent skip exploration or debugging, replace it with an abstract phrase such as "
        "relevant file, appropriate command, necessary dependency, required configuration, validation check, or "
        "expected output format.\n"
        "- For short-plan outputs specifically, do not include tool-specific command recipes, trace-discovered bug "
        "causes, workaround descriptions, implementation tactics, or step names that reveal the concrete fix.\n"
        "- For short-plan outputs specifically, prefer generic words such as task, workspace, component, requirement, "
        "approach, change, artifact, validation check, failure, and result. Avoid domain-specific nouns copied from "
        "the traces.\n"
        "- Before finalizing, check every sentence: if it tells the future agent something it would otherwise need "
        "to discover by inspecting, testing, or debugging the current environment, rewrite it as a more abstract "
        "transferable step."
    )


def _build_user_message(task_name: str, records: list[dict[str, Any]], *, form: str, hint_mode: str) -> str:
    base = skillgen._build_user_message(records, hint_mode=hint_mode)
    if form == "short-plan":
        request = (
            "Output only the short plan in this format:\n\n"
            "# Short Plan\n\n"
            "1. {{high-level step 1}}\n"
            "2. {{high-level step 2}}\n"
            "3. {{high-level step 3}}\n\n"
            "Use 3-5 numbered steps. Each step must be one concise sentence. The plan should be general enough "
            "to guide a future agent on the same task type, but it must not solve the task for the agent. "
            "Prefer a generic stage vocabulary: inspect, identify, plan, implement, validate, iterate."
        )
    elif form == "test-first":
        request = "Output only a test-first procedure: 3-5 high-level validation checkpoints, one concise sentence each."
    else:
        request = "Output only the requested compact procedure."
    return (
        f"Task name: {task_name}\n"
        f"Requested compact procedure form: {form}\n\n"
        f"{base}\n\n"
        f"{request}"
    )


def _find_instruction_path(task_source_root: Path, task_name: str) -> Path | None:
    direct = task_source_root / task_name / "instruction.md"
    if direct.is_file():
        return direct
    matches = sorted(task_source_root.glob(f"*/{task_name}/instruction.md"))
    if matches:
        return matches[0]
    return None


def _load_task_instruction(task_source_root: Path, task_name: str, *, max_chars: int) -> tuple[str, Path]:
    path = _find_instruction_path(task_source_root, task_name)
    if path is None:
        raise RuntimeError(f"missing instruction.md for task={task_name!r} under {task_source_root}")
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n...[truncated]"
    return text, path


def _build_instruction_user_message(task_name: str, instruction: str, *, form: str) -> str:
    if form == "short-plan":
        request = (
            "Read the task instruction and write a short pre-execution plan for the agent.\n"
            "Output only this format:\n\n"
            "# Short Plan\n\n"
            "1. {{step 1}}\n"
            "2. {{step 2}}\n"
            "3. {{step 3}}\n\n"
            "Use 3-5 numbered steps. Each step must be one concise sentence. Do not solve the task, do not add "
            "facts absent from the instruction, and do not include commands, code, exact file edits, final answers, "
            "or hidden implementation details."
        )
    else:
        request = (
            "Generate only the requested compact procedure. Base it only on the task instruction above. "
            "Do not use prior workflow knowledge, hidden solution traces, or assumptions about the correct answer."
        )
    return (
        f"Task name: {task_name}\n"
        f"Requested compact procedure form: {form}\n\n"
        "Task instruction:\n"
        "-----\n"
        f"{instruction}\n"
        "-----\n\n"
        f"{request}"
    )


def _fallback_procedure(task_name: str, form: str) -> str:
    if form == "short-plan":
        return (
            "## Short Plan\n\n"
            "1. Inspect the task instruction and current workspace before making changes.\n"
            "2. Identify the expected deliverable, relevant files, and available validation command.\n"
            "3. Make the smallest task-specific change needed to produce the deliverable.\n"
            "4. Run the relevant validation or sanity check.\n"
            "5. Revise based on observed failures before final submission.\n"
        )
    if form == "test-first":
        return (
            "## Test-First Procedure\n\n"
            "1. Identify the expected success condition and output format before editing.\n"
            "2. Locate an existing test/checker or create a lightweight validation check.\n"
            "3. Run the check once to understand the current failure or baseline state.\n"
            "4. Implement the smallest fix or artifact generation step.\n"
            "5. Rerun the same check and fix any mismatch before final submission.\n"
        )
    return (
        "## Reusable Script-Style Command Recipe\n\n"
        "- Inspect the current workspace before adapting any command.\n"
        "- Use placeholders such as `<input_path>`, `<output_path>`, and `<target_file>` for task-specific values.\n"
        "- Prefer small scripts that parse inputs, produce the requested artifact, and run a separate validation command.\n"
        "- Do not hard-code final answers or source-run-specific constants.\n"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compact procedure baselines.")
    parser.add_argument("--workflow-input", type=Path, required=True, help="Workflow JSON used for the task list and, in workflow mode, source records.")
    parser.add_argument("--source", choices=sorted(GENERATION_SOURCES), default="workflow", help="Use workflow records or task instructions as generation input.")
    parser.add_argument("--task-source-root", type=Path, default=None, help="Benchmark task source root used by --source instruction.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root for compact procedure artifacts.")
    parser.add_argument("--benchmark", default="terminal-bench-2")
    parser.add_argument("--conditions", default="5s0f")
    parser.add_argument("--forms", default="short-plan,test-first,script")
    parser.add_argument("--task-name", action="append", default=[], help="Optional task name filter; may be repeated.")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=sorted(skillgen.PROVIDER_DEFAULT_KEY_ENV.keys()), required=True)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--procedure-creator-model", required=True)
    parser.add_argument("--hint-mode", choices=["with-status", "no-hint"], default="no-hint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-prompt-steps-per-trace", type=int, default=80)
    parser.add_argument("--max-prompt-command-chars", type=int, default=500)
    parser.add_argument("--max-prompt-result-chars", type=int, default=1600)
    parser.add_argument("--max-prompt-agent-message-chars", type=int, default=900)
    parser.add_argument("--max-instruction-chars", type=int, default=12000)
    parser.add_argument("--chat-timeout-sec", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-sec", type=float, default=3.0)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    forms = _parse_forms(args.forms)
    conditions = skillgen._parse_conditions(args.conditions)
    source = str(args.source).strip().lower()
    if source not in GENERATION_SOURCES:
        raise RuntimeError(f"invalid source={source!r}; allowed={sorted(GENERATION_SOURCES)}")
    task_source_root = args.task_source_root.resolve() if args.task_source_root else None
    if source == "instruction" and task_source_root is None:
        raise RuntimeError("--source instruction requires --task-source-root")
    records = skillgen._load_trace_records(args.workflow_input.resolve())
    by_task: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_task.setdefault(str(rec.get("task_name") or ""), []).append(rec)
    by_task = {k: v for k, v in by_task.items() if k}
    task_filter = {str(x).strip() for x in (args.task_name or []) if str(x).strip()}
    if task_filter:
        by_task = {k: v for k, v in by_task.items() if k in task_filter}
        missing = sorted(task_filter - set(by_task))
        if missing:
            raise RuntimeError(f"task-name filter did not match workflow records: {missing}")

    provider = str(args.provider).strip().lower()
    api_key_env = args.api_key_env or skillgen.PROVIDER_DEFAULT_KEY_ENV[provider]
    api_key = ""
    base_url = args.base_url
    if not args.dry_run:
        import os

        api_key = str(os.environ.get(api_key_env) or "").strip()
        if not api_key:
            raise RuntimeError(f"missing key env: {api_key_env}")
        if not base_url:
            if provider in {"openrouter", "uniapi"}:
                base_url = "https://openrouter.ai/api/v1" if provider == "openrouter" else "https://api.uniapi.io/v1"
            elif provider in {"google", "claude"}:
                raise RuntimeError(f"provider={provider} requires --base-url")

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workflow_input": str(args.workflow_input.resolve()),
        "source": source,
        "task_source_root": str(task_source_root) if task_source_root else None,
        "output_root": str(args.output_root.resolve()),
        "benchmark": str(args.benchmark),
        "conditions": [label for _s, _f, label in conditions],
        "forms": forms,
        "agent": args.agent,
        "model": args.model,
        "procedure_creator_model": args.procedure_creator_model,
        "provider": provider,
        "api_key_env": api_key_env,
        "dry_run": bool(args.dry_run),
        "tasks": {},
    }

    for task_name, task_records in sorted(by_task.items()):
        task_log: dict[str, Any] = {}
        for success_n, failure_n, condition_label in conditions:
            successes = [r for r in task_records if str(r.get("status") or "").lower() == "success"]
            failures = [r for r in task_records if str(r.get("status") or "").lower() == "failure"]
            if len(successes) < success_n or len(failures) < failure_n:
                continue
            selected = [*successes[:success_n], *failures[:failure_n]]
            instruction_text = ""
            instruction_path: Path | None = None
            if source == "workflow":
                selected = skillgen._compress_records_for_prompt(
                    selected,
                    max_steps=max(1, int(args.max_prompt_steps_per_trace)),
                    max_cmd_chars=max(50, int(args.max_prompt_command_chars)),
                    max_result_chars=max(100, int(args.max_prompt_result_chars)),
                    max_agent_msg_chars=max(80, int(args.max_prompt_agent_message_chars)),
                )
            else:
                assert task_source_root is not None
                instruction_text, instruction_path = _load_task_instruction(
                    task_source_root,
                    task_name,
                    max_chars=max(0, int(args.max_instruction_chars)),
                )
            for form in forms:
                task_slug = _normalize_slug(task_name, default="task")
                out_dir = args.output_root.resolve() / condition_label / form / task_slug
                out_path = out_dir / "procedure.md"
                manifest_path = out_dir / "generation_manifest.json"
                if args.resume and out_path.is_file():
                    task_log[f"{condition_label}/{form}"] = {"status": "skipped_existing", "procedure_path": str(out_path)}
                    continue
                if source == "workflow":
                    user_message = _build_user_message(task_name, selected, form=form, hint_mode=str(args.hint_mode))
                else:
                    user_message = _build_instruction_user_message(task_name, instruction_text, form=form)
                system_prompt = _build_system_prompt(form, source=source)
                if args.dry_run:
                    content = _fallback_procedure(task_name, form)
                    status = "dry_run"
                else:
                    content = ""
                    last_error = None
                    for attempt in range(1, max(1, int(args.max_retries)) + 1):
                        try:
                            content = skillgen._chat_completion(
                                base_url=str(base_url),
                                provider=provider,
                                api_key=api_key,
                                model=args.procedure_creator_model,
                                system_prompt=system_prompt,
                                user_message=user_message,
                                timeout_sec=int(args.chat_timeout_sec),
                                max_output_tokens=int(args.max_output_tokens),
                            ).strip()
                            if content:
                                break
                        except Exception as exc:  # pragma: no cover - network/API behavior
                            last_error = f"{type(exc).__name__}: {exc}"
                            import time

                            time.sleep(max(0.0, float(args.retry_backoff_sec)) * attempt)
                    if not content:
                        raise RuntimeError(f"failed to generate {form} for {task_name}: {last_error}")
                    status = "generated"

                out_dir.mkdir(parents=True, exist_ok=True)
                out_path.write_text(content.rstrip() + "\n", encoding="utf-8")
                _json_dump(
                    manifest_path,
                    {
                        "task_name": task_name,
                        "condition": condition_label,
                        "form": form,
                        "status": status,
                        "procedure_path": str(out_path),
                        "prompt_chars": len(user_message),
                        "source": source,
                        "selected_trace_count": len(selected) if source == "workflow" else 0,
                        "source_trials": [str(r.get("trial_name") or "") for r in selected] if source == "workflow" else [],
                        "instruction_path": str(instruction_path) if instruction_path else None,
                        "system_prompt": system_prompt,
                    },
                )
                task_log[f"{condition_label}/{form}"] = {"status": status, "procedure_path": str(out_path)}
        summary["tasks"][task_name] = task_log

    _json_dump(args.output_root.resolve() / "generation_summary.json", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
