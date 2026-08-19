#!/usr/bin/env python3
"""Agent skill-selection evaluation over retrieval candidate pools."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_ROOT,
    agent_model_slug,
    f1_score,
    iter_jsonl,
    normalize_slug,
    retrieval_run_dir,
    setting_from_rows,
    write_json,
    write_jsonl,
    write_latest_pointer,
)

CLEAN_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run skill retrieval evaluation for a candidate pool.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--candidate-pool", type=Path, default=None)
    parser.add_argument("--skills-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--agent", default="agent-pick")
    parser.add_argument("--agent-mode", choices=["chat", "command", "harbor"], default="command")
    parser.add_argument("--agent-command", default="")
    parser.add_argument("--provider", choices=["openai", "openrouter", "google", "claude", "uniapi"], default="google")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--pm2s-root", type=Path, default=CLEAN_ROOT)
    parser.add_argument(
        "--procmem2skills-root",
        type=Path,
        default=Path(os.environ.get("PROCMEM2SKILLS_ROOT", "/raid/zhiyuan/procmem2skills")),
    )
    parser.add_argument("--benchmark-config", type=Path, default=CLEAN_ROOT / "configs" / "benchmarks.json")
    parser.add_argument(
        "--task-source-root",
        type=Path,
        default=Path(os.environ.get("TASK_SOURCE_ROOT", "/raid/zhiyuan/procmem2skills/benchmarks/harbor-datasets")),
    )
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--n-concurrent", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--command-timeout-sec", type=int, default=300)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source_to_skill_text(source: str) -> str:
    raw = str(source or "")
    text = raw.strip()
    if not raw:
        return ""
    path = Path(text).expanduser()
    if path.is_file():
        return clean_prompt_text(path.read_text(encoding="utf-8", errors="replace"))
    return clean_prompt_text(raw)


def clean_prompt_text(text: str) -> str:
    return str(text or "").replace("\x00", "")


def load_task_instruction_for_manifest(benchmark: str, task_name: str) -> str:
    benchmark = str(benchmark or "").strip().lower()
    task_name = str(task_name or "").strip()
    if not task_name:
        return ""
    candidates: list[Path] = []
    if benchmark in {"skillsbench", "skills-bench"}:
        candidates.append(Path("/raid/zhiyuan/procmem2skills/benchmarks/skillsbench/tasks") / task_name / "instruction.md")
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace").strip()
    return ""


def skill_summary(skill_md: str, max_chars: int = 700) -> str:
    text = str(skill_md or "")
    summary_parts: list[str] = []
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            frontmatter = text[3:end].strip().splitlines()
            capture_desc = False
            desc_lines: list[str] = []
            for line in frontmatter:
                stripped = line.strip()
                if stripped.startswith("name:") or stripped.startswith("displayName:"):
                    summary_parts.append(stripped)
                if stripped.startswith("description:"):
                    capture_desc = True
                    desc = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                    if desc and desc not in {">", ">-", "|", "|-"}:
                        desc_lines.append(desc)
                    continue
                if capture_desc:
                    if line.startswith((" ", "\t")) and stripped:
                        desc_lines.append(stripped.strip('"').strip("'"))
                    else:
                        capture_desc = False
            if desc_lines:
                summary_parts.append("description: " + " ".join(desc_lines))
    if not summary_parts:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("---"):
                summary_parts.append(stripped)
            if len(" ".join(summary_parts)) >= max_chars:
                break
    return " ".join(summary_parts)[:max_chars]


def load_manifest_rows(manifest_path: Path, *, benchmark: str = "") -> list[dict]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(raw_rows, list):
        raise RuntimeError(f"manifest missing rows: {manifest_path}")

    setting = setting_from_rows([], manifest_path.resolve())
    out: list[dict] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        task_name = str(raw.get("task_name") or "").strip()
        candidates: list[dict] = []
        for idx, ref in enumerate(raw.get("neighbors") or [], start=1):
            if not isinstance(ref, dict):
                continue
            skill_name = str(ref.get("neighbor_task_name") or ref.get("skill_slug") or f"skill-{idx}").strip()
            skill_slug = str(ref.get("skill_slug") or skill_name).strip()
            skill_text = source_to_skill_text(str(ref.get("source_skill_md") or ""))
            if not skill_name or not skill_text:
                continue
            candidates.append(
                {
                    "skill_name": skill_name,
                    "skill_slug": skill_slug,
                    "skill_md": skill_text,
                    "description": "",
                    "role": str(ref.get("role") or ""),
                }
            )
        if not candidates:
            continue
        gt_names = [str(c["skill_name"]) for c in candidates if str(c.get("role") or "").lower() == "gt"]
        out.append(
            {
                "benchmark": benchmark or setting.get("benchmark") or "skillsbench",
                "task_name": task_name,
                "task_description": str(raw.get("task_description") or raw.get("instruction") or load_task_instruction_for_manifest(benchmark or setting.get("benchmark") or "skillsbench", task_name) or task_name),
                "pool_size": setting.get("pool_size", len(candidates)),
                "noise_mode": setting.get("noise_mode", "unknown"),
                "seed": setting.get("seed", 0),
                "gt_skill_names": gt_names,
                "candidate_skills": candidates,
                "source_manifest": str(manifest_path.resolve()),
            }
        )
    return out


def prepare_agent_pick_workspace(workspace: Path, row: dict) -> Path:
    if workspace.exists():
        shutil.rmtree(workspace)
    skills_root = workspace / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for idx, skill in enumerate(row.get("candidate_skills") or [], start=1):
        name = str(skill.get("skill_slug") or skill.get("skill_name") or f"skill-{idx}")
        slug = normalize_slug(name, default=f"skill-{idx}")
        if slug in seen:
            slug = f"{slug}-{idx}"
        seen.add(slug)
        skill_dir = skills_root / slug
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(str(skill.get("skill_md") or ""), encoding="utf-8")
    (workspace / "task.md").write_text(str(row.get("task_description") or row.get("task_name") or ""), encoding="utf-8")
    return workspace


def default_key_env(provider: str) -> str:
    return {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "google": "GOOGLE_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "uniapi": "UNIAPI_API_KEY",
    }[provider]


def default_base_url(provider: str) -> str:
    return {
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "google": "https://generativelanguage.googleapis.com/v1beta/openai",
        "claude": "",
        "uniapi": "https://api.uniapi.io/v1",
    }[provider]


def chat_completion(*, base_url: str, api_key: str, model: str, system_prompt: str, user_message: str, timeout_sec: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
        "max_tokens": 128,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "pm2s-skill-retrieval/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"LLM HTTPError {exc.code}: {detail[:500]}") from exc
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]
    raise RuntimeError(f"LLM response missing message: {str(data)[:500]}")


def build_prompt(row: dict) -> tuple[str, str]:
    system_prompt = (
        "You are evaluating which skill an agent should activate for a task. "
        "Choose exactly one candidate skill. Do not solve the task. "
        "Output only: <tool_call name=\"skill\"><name>SKILL_NAME</name></tool_call>"
    )
    lines = [
        "Task:",
        str(row.get("task_description") or row.get("task_name") or "").strip(),
        "",
        "Candidate skills:",
    ]
    for idx, skill in enumerate(row.get("candidate_skills") or [], start=1):
        lines.append(f"{idx}. name: {skill.get('skill_name')}")
        desc = str(skill.get("description") or "").strip()
        if desc:
            lines.append(f"   description: {desc[:1200]}")
    lines.append("")
    lines.append("Return only the XML tool call.")
    return system_prompt, "\n".join(lines)


def build_agent_pick_prompt(row: dict) -> str:
    lines = [
        "You are running a skill retrieval evaluation.",
        "Do not solve the task and do not modify task files.",
        "Choose any number of candidate skills that would help solve the task.",
        "Choose zero only if none of the candidate skills are useful.",
        "Return only strict JSON in this shape: {\"skills\": [\"skill-name\"]}",
        "Use skill names exactly as listed below.",
        "",
        "Candidate skills:",
    ]
    for idx, skill in enumerate(row.get("candidate_skills") or [], start=1):
        name = str(skill.get("skill_name") or skill.get("skill_slug") or f"skill-{idx}").strip()
        summary = skill_summary(str(skill.get("skill_md") or ""), max_chars=700)
        if summary:
            lines.append(f"{idx}. {name}: {summary}")
        else:
            lines.append(f"{idx}. {name}")
    lines.extend([
        "",
        "Task:",
        str(row.get("task_description") or row.get("task_name") or "").strip(),
    ])
    return "\n".join(lines)


def default_agent_command(agent: str, model: str, prompt: str) -> list[str]:
    name = str(agent or "").strip().lower()
    model_leaf = str(model or "").rsplit("/", 1)[-1]
    if name in {"gemini", "gemini-cli"}:
        return [
            "bash",
            "-lc",
            (
                'if [ -s "$HOME/.nvm/nvm.sh" ]; then . "$HOME/.nvm/nvm.sh"; fi; '
                'if ! command -v gemini >/dev/null 2>&1; then '
                'echo "gemini CLI not found on host; install it or pass --agent-command" >&2; exit 127; '
                "fi; "
                'gemini --yolo --model="$0" --prompt="$1"'
            ),
            model_leaf,
            prompt,
        ]
    if name in {"codex", "codex-cli"}:
        return ["codex", "exec", "--model", model, "--skip-git-repo-check", "--", prompt]
    if name in {"claude", "claude-code"}:
        return ["claude", "--verbose", "--output-format=stream-json", "--model", model, "--print"]
    raise RuntimeError(f"no default command for agent={agent}; pass --agent-command")


def replace_command_placeholders(text: str, *, model: str, prompt_file: Path, prompt: str) -> str:
    model_leaf = str(model or "").rsplit("/", 1)[-1]
    return (
        text.replace("{model_leaf}", model_leaf)
        .replace("{model}", str(model))
        .replace("{prompt_file}", str(prompt_file))
        .replace("{prompt}", prompt)
    )


def expand_command_template(template: str, *, model: str, prompt_file: Path, prompt: str) -> list[str]:
    import shlex

    parts = shlex.split(template)
    return [replace_command_placeholders(p, model=model, prompt_file=prompt_file, prompt=prompt) for p in parts]


def run_agent_pick_command(
    *,
    row: dict,
    workspace: Path,
    agent: str,
    model: str,
    agent_command: str,
    timeout_sec: int,
) -> str:
    prompt_file = workspace / "prompt.md"
    prompt = clean_prompt_text(build_agent_pick_prompt(row))
    prompt_file.write_text(prompt, encoding="utf-8")
    use_stdin = (not agent_command.strip()) and str(agent or "").strip().lower() in {"claude", "claude-code"}
    cmd = (
        expand_command_template(agent_command, model=model, prompt_file=prompt_file, prompt=prompt)
        if agent_command.strip()
        else default_agent_command(agent, model, prompt)
    )
    proc = subprocess.run(
        cmd,
        cwd=workspace,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        input=prompt if use_stdin else None,
        timeout=max(1, timeout_sec),
        check=False,
    )
    if proc.returncode != 0:
        stdout_tail = (proc.stdout or "")[-2000:]
        stderr_tail = (proc.stderr or "")[-2000:]
        raise RuntimeError(f"agent command failed rc={proc.returncode}: stdout_tail={stdout_tail} stderr_tail={stderr_tail}")
    if proc.stderr.strip():
        return proc.stdout.rstrip() + "\n\n[stderr]\n" + proc.stderr.rstrip()
    return proc.stdout


def parse_pick(text: str) -> str:
    m = re.search(r"<tool_call\s+name=[\"']skill[\"']>\s*<name>\s*(.*?)\s*</name>\s*</tool_call>", text, flags=re.I | re.S)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    m = re.search(r"<name>\s*(.*?)\s*</name>", text, flags=re.I | re.S)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def parse_picked_skills(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    candidates: list[str] = []

    # Stream-json agents (Claude Code/Codex) may include metadata fields named
    # "skills" in initialization events. Only parse model message/result text.
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if not isinstance(event, dict):
            continue

        if event.get("type") == "result" and isinstance(event.get("result"), str):
            candidates.insert(0, event["result"].strip())
            continue

        msg = event.get("message") if isinstance(event.get("message"), dict) else None
        if event.get("type") == "assistant" and msg:
            parts = msg.get("content")
            if isinstance(parts, list):
                for part in parts:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        candidates.insert(0, part["text"].strip())
            elif isinstance(parts, str):
                candidates.insert(0, parts.strip())
            continue

        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        text_value = ""
        if item.get("type") == "agent_message":
            text_value = str(item.get("text") or "").strip()
        elif event.get("type") == "agent_message":
            text_value = str(event.get("text") or "").strip()
        if text_value:
            candidates.insert(0, text_value)

    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.I | re.S)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    candidates.append(raw)

    seen_candidates: set[str] = set()
    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if not candidate or candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            value = payload.get("skills") or payload.get("skill_names") or payload.get("picked_skills")
            if isinstance(value, list):
                return [re.sub(r"\s+", " ", str(x)).strip() for x in value if str(x).strip()]
            if isinstance(value, str) and value.strip():
                return [re.sub(r"\s+", " ", value).strip()]
        if isinstance(payload, list):
            return [re.sub(r"\s+", " ", str(x)).strip() for x in payload if str(x).strip()]

    for candidate in candidates:
        for match in re.finditer(r'{\s*"skills"\s*:\s*\[[^\]]*\]\s*}', candidate, flags=re.S):
            try:
                payload = json.loads(match.group(0))
            except Exception:
                continue
            value = payload.get("skills")
            if isinstance(value, list):
                return [re.sub(r"\s+", " ", str(x)).strip() for x in value if str(x).strip()]

    xml_pick = parse_pick(raw)
    if xml_pick:
        return [xml_pick]
    names = re.findall(r"<name>\s*(.*?)\s*</name>", raw, flags=re.I | re.S)
    return [re.sub(r"\s+", " ", x).strip() for x in names if x.strip()]

def canonicalize_picked_skills(picked: list[str], row: dict) -> list[str]:
    alias_to_name: dict[str, str] = {}
    for skill in row.get("candidate_skills") or []:
        name = str(skill.get("skill_name") or "").strip()
        slug = str(skill.get("skill_slug") or "").strip()
        if not name:
            continue
        aliases = {name, normalize_slug(name), slug, normalize_slug(slug)}
        for alias in aliases:
            if str(alias or "").strip():
                alias_to_name[str(alias).strip().lower()] = name
    out: list[str] = []
    seen: set[str] = set()
    for item in picked:
        key = str(item or "").strip().lower()
        canonical = alias_to_name.get(key) or alias_to_name.get(normalize_slug(key).lower()) or str(item).strip()
        if canonical and canonical.lower() not in seen:
            out.append(canonical)
            seen.add(canonical.lower())
    return out


def gt_metrics(picked: str, gt_names: list[str]) -> tuple[bool, float, float]:
    gt = {x.strip().lower() for x in gt_names if str(x).strip()}
    if not picked:
        return False, 0.0, 0.0
    hit = picked.strip().lower() in gt
    precision = 1.0 if hit else 0.0
    recall = (1.0 / max(1, len(gt))) if hit else 0.0
    return hit, precision, recall


def gt_set_metrics(picked: list[str], gt_names: list[str]) -> tuple[bool, float, float]:
    picked_set = {str(x).strip().lower() for x in picked if str(x).strip()}
    gt = {str(x).strip().lower() for x in gt_names if str(x).strip()}
    if not picked_set or not gt:
        return False, 0.0, 0.0
    inter = picked_set & gt
    precision = len(inter) / len(picked_set)
    recall = len(inter) / len(gt)
    return bool(inter), precision, recall


def load_context_runner(pm2s_root: Path):
    scripts_dir = pm2s_root.resolve() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import run_context_comparison as rcc  # type: ignore

    return rcc


def materialize_harbor_skill_entries(base_dir: Path, row: dict) -> list[dict[str, str]]:
    task_slug = normalize_slug(str(row.get("task_name") or "task"), default="task")
    task_root = base_dir / task_slug
    if task_root.exists():
        shutil.rmtree(task_root)
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for idx, skill in enumerate(row.get("candidate_skills") or [], start=1):
        name = str(skill.get("skill_name") or skill.get("skill_slug") or f"skill-{idx}").strip()
        slug = normalize_slug(str(skill.get("skill_slug") or name), default=f"skill-{idx}")
        if slug in seen:
            slug = f"{slug}-{idx}"
        seen.add(slug)
        skill_dir = task_root / slug
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(str(skill.get("skill_md") or ""), encoding="utf-8")
        entries.append(
            {
                "skill_name": name,
                "skill_slug": slug,
                "skill_md": str(skill_path),
                "role": str(skill.get("role") or ""),
            }
        )
    return entries


def build_harbor_pick_instruction(row: dict, *, container_skill_root: str) -> str:
    candidate_names = [str(s.get("skill_name") or "") for s in row.get("candidate_skills") or [] if str(s.get("skill_name") or "").strip()]
    return "\n".join(
        [
            "You are running a skill retrieval evaluation.",
            "Do not solve the task. Do not create, edit, or validate task answer files.",
            f"The candidate skills are installed under `{container_skill_root}`.",
            "Inspect the candidate skill directories and choose any number of skills that would help solve the task.",
            "Choose zero only if none of the installed skills are useful.",
            "Return only strict JSON in this shape: {\"skills\": [\"skill-name\"]}",
            "Use skill names exactly as listed below.",
            "",
            "Candidate skill names:",
            *[f"- {name}" for name in candidate_names],
            "",
            "Task:",
            str(row.get("task_description") or row.get("task_name") or "").strip(),
        ]
    ).rstrip() + "\n"


def prepare_harbor_pick_tasks(
    *,
    rows: list[dict],
    out_dir: Path,
    benchmark: str,
    agent: str,
    pm2s_root: Path,
    procmem2skills_root: Path,
    benchmark_config: Path,
    task_source_root: Path,
) -> tuple[Path, list[dict[str, Any]], dict[str, dict]]:
    rcc = load_context_runner(pm2s_root)
    benchmark_map = rcc._load_benchmark_map(benchmark_config.resolve())
    bench = benchmark_map.get(benchmark)
    if bench is None:
        raise RuntimeError(f"benchmark not found in config: {benchmark}")
    prepared_root = out_dir / "harbor" / "prepared-tasks"
    source_skills_root = out_dir / "harbor" / "source-skills"
    if prepared_root.exists():
        shutil.rmtree(prepared_root)
    if source_skills_root.exists():
        shutil.rmtree(source_skills_root)
    prepared_root.mkdir(parents=True, exist_ok=True)

    src_roots = rcc._candidate_task_source_roots(
        base_root=task_source_root.resolve(),
        benchmark_cfg=bench,
        procmem2skills_root=procmem2skills_root.resolve(),
    )
    container_skill_root = rcc._agent_skill_container_root(agent)
    prepared_rows: list[dict[str, Any]] = []
    for row in rows:
        task_name = str(row.get("task_name") or "").strip()
        if not task_name:
            continue
        task_dir = rcc._copy_task_source(src_roots, task_name, prepared_root)
        original_instruction = ""
        instruction_path = task_dir / "instruction.md"
        if instruction_path.is_file():
            original_instruction = instruction_path.read_text(encoding="utf-8", errors="replace").strip()
        entries = materialize_harbor_skill_entries(source_skills_root, row)
        injected = rcc._inject_skill_pool_into_task_workspace(task_dir, entries, task_name, agent=agent)
        prompt_row = dict(row)
        if original_instruction:
            prompt_row["task_description"] = original_instruction
        instruction_path.write_text(
            build_harbor_pick_instruction(prompt_row, container_skill_root=container_skill_root),
            encoding="utf-8",
        )
        prepared_rows.append(
            {
                "task_name": task_name,
                "task_dir": str(task_dir),
                "source_skill_count": len(entries),
                "injected_skills": [str(p) for p in injected],
                "container_skill_root": container_skill_root,
            }
        )
    if not prepared_rows:
        raise RuntimeError("no Harbor pick tasks prepared")
    return prepared_root, prepared_rows, bench


def _read_harbor_agent_text(trial_dir: Path, agent: str) -> str:
    agent_slug = normalize_slug(agent, default="agent")
    candidates = [trial_dir / "agent" / f"{agent}.txt", trial_dir / "agent" / f"{agent_slug}.txt"]
    agent_dir = trial_dir / "agent"
    if agent_dir.is_dir():
        candidates.extend(sorted(agent_dir.glob("*.txt")))
    seen: set[Path] = set()
    chunks: list[str] = []
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            chunks.append(text)
    return "\n".join(chunks).strip()


def find_harbor_agent_records(run_dir: Path, agent: str) -> list[dict[str, str]]:
    # Preserve repeated Harbor attempts as separate retrieval samples.
    records: list[dict[str, str]] = []
    trial_dirs: list[Path] = []
    seen_dirs: set[Path] = set()
    for result_path in sorted(run_dir.rglob("result.json")):
        trial_dir = result_path.parent
        if trial_dir not in seen_dirs:
            seen_dirs.add(trial_dir)
            trial_dirs.append(trial_dir)

    if not trial_dirs:
        candidates = list(run_dir.rglob(f"agent/{agent}.txt")) + list(run_dir.rglob(f"agent/{normalize_slug(agent, default='agent')}.txt"))
        candidates += list(run_dir.rglob("agent/*.txt"))
        for path in sorted(set(candidates)):
            if path.is_file():
                trial_dirs.append(path.parent.parent)

    per_task_count: dict[str, int] = {}
    for trial_dir in trial_dirs:
        task_name = trial_dir.name.split("__", 1)[0]
        if not task_name:
            continue
        per_task_count[task_name] = per_task_count.get(task_name, 0) + 1
        records.append(
            {
                "task_name": task_name,
                "attempt_index": str(per_task_count[task_name]),
                "trial_dir": str(trial_dir),
                "raw_llm_output": _read_harbor_agent_text(trial_dir, agent),
            }
        )
    return records


def run_harbor_agent_pick(
    *,
    args: argparse.Namespace,
    rows: list[dict],
    out_dir: Path,
    benchmark: str,
    base_url: str,
    api_key_env: str,
) -> tuple[list[dict], dict[str, Any]]:
    rcc = load_context_runner(args.pm2s_root)
    benchmark_config = args.benchmark_config or (args.pm2s_root / "configs" / "benchmarks.json")
    prepared_root, prepared_rows, bench = prepare_harbor_pick_tasks(
        rows=rows,
        out_dir=out_dir,
        benchmark=benchmark,
        agent=args.agent,
        pm2s_root=args.pm2s_root,
        procmem2skills_root=args.procmem2skills_root,
        benchmark_config=benchmark_config,
        task_source_root=args.task_source_root,
    )
    harbor_run_dir = out_dir / "harbor" / "results"
    try:
        provider_env = rcc._prepare_provider_env(provider=str(args.provider).lower(), api_key_env=api_key_env, base_url=base_url)
    except RuntimeError:
        if not args.dry_run:
            raise
        provider_env = dict(os.environ)
        if base_url:
            provider_env.setdefault("OPENAI_BASE_URL", base_url)
            provider_env.setdefault("OPENROUTER_BASE_URL", base_url)
    provider_env = rcc._resolve_stable_harbor_runtime(
        procmem2skills_root=args.procmem2skills_root.resolve(),
        env=provider_env,
    )
    agent_name_norm = rcc._normalize_agent_name(args.agent)
    use_codex_oauth = agent_name_norm == "codex" and bool(
        str(provider_env.get("HARBOR_CODEX_AUTH_JSON_PATH") or os.environ.get("HARBOR_CODEX_AUTH_JSON_PATH") or "").strip()
    )
    if use_codex_oauth:
        provider_env.pop("OPENAI_BASE_URL", None)
        provider_env.pop("OPENROUTER_BASE_URL", None)
        base_url = ""
    agent_env = {"HARBOR_CODEX_ATTEMPT_TIMEOUT_SEC": str(max(60, int(args.command_timeout_sec)))}
    agent_env.update(rcc._agent_home_env_vars(args.agent))
    if agent_name_norm in {"gemini-cli", "gemini"}:
        gemini_key = str(provider_env.get("GEMINI_API_KEY") or "").strip()
        if gemini_key:
            agent_env["GEMINI_API_KEY"] = gemini_key

    cmd = rcc._build_arm_command(
        procmem2skills_root=args.procmem2skills_root.resolve(),
        benchmark=bench,
        run_dir=harbor_run_dir,
        prepared_tasks_dir=prepared_root,
        experiment_id=normalize_slug(f"skill-retrieval-pick-{benchmark}-{int(time.time())}", default="skill-retrieval")[:56].rstrip("-"),
        model=rcc._strip_openai_model_prefix(args.model),
        agent=args.agent,
        n_attempts=max(1, int(args.n_attempts)),
        n_concurrent=max(1, int(args.n_concurrent)),
        max_steps=max(1, int(args.max_steps)),
        command_timeout_sec=max(1, int(args.command_timeout_sec)),
        base_url=base_url,
        agent_env=agent_env,
        runner_python=provider_env.get("PROCMEM_BENCHMARK_PYTHON"),
        dry_run=bool(args.dry_run),
        memory_setting="none",
    )

    harbor_meta: dict[str, Any] = {
        "prepared_tasks_dir": str(prepared_root),
        "prepared_task_count": len(prepared_rows),
        "prepared_rows": prepared_rows,
        "harbor_run_dir": str(harbor_run_dir),
        "harbor_command": cmd,
        "harbor_dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        return [], harbor_meta

    completed = subprocess.run(
        cmd,
        cwd=args.procmem2skills_root.resolve(),
        env=provider_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    harbor_meta.update(
        {
            "harbor_return_code": int(completed.returncode),
            "harbor_stdout_tail": (completed.stdout or "")[-3000:],
            "harbor_stderr_tail": (completed.stderr or "")[-3000:],
        }
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Harbor agent-pick run failed rc={completed.returncode}: {(completed.stderr or '')[-1000:]}")

    records = find_harbor_agent_records(harbor_run_dir, args.agent)
    by_task = {str(row.get("task_name") or ""): row for row in rows}
    results: list[dict] = []
    if records:
        iterable = records
    else:
        iterable = [
            {"task_name": str(row.get("task_name") or ""), "attempt_index": "1", "trial_dir": "", "raw_llm_output": ""}
            for row in rows
        ]
    for record in iterable:
        task_name = str(record.get("task_name") or "")
        row = by_task.get(task_name)
        if row is None:
            continue
        raw = str(record.get("raw_llm_output") or "")
        picked_skills = canonicalize_picked_skills(parse_picked_skills(raw), row)
        hit, precision, recall = gt_set_metrics(picked_skills, list(row.get("gt_skill_names") or []))
        results.append(
            {
                "benchmark": row.get("benchmark"),
                "task_name": row.get("task_name"),
                "attempt_index": int(record.get("attempt_index") or 1),
                "trial_dir": record.get("trial_dir") or "",
                "pool_size": row.get("pool_size"),
                "noise_mode": row.get("noise_mode"),
                "seed": row.get("seed"),
                "gt_skill_names": row.get("gt_skill_names"),
                "candidate_skill_names": [s.get("skill_name") for s in row.get("candidate_skills") or []],
                "picked_skill": picked_skills[0] if picked_skills else "",
                "picked_skills": picked_skills,
                "hit": hit,
                "precision": precision,
                "recall": recall,
                "refusal": not bool(picked_skills),
                "raw_llm_output": raw,
                "error": "" if raw else "no agent log parsed",
            }
        )
    return results, harbor_meta


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.skills_manifest is None and args.candidate_pool is None:
        raise RuntimeError("pass either --skills-manifest or --candidate-pool")
    if args.skills_manifest is not None:
        rows = load_manifest_rows(args.skills_manifest.resolve(), benchmark=args.benchmark.strip())
        source_path = args.skills_manifest.resolve()
    else:
        rows = list(iter_jsonl(args.candidate_pool.resolve()))
        source_path = args.candidate_pool.resolve()
    if args.task_limit:
        rows = rows[: args.task_limit]
    if not rows:
        raise RuntimeError(f"empty retrieval input: {source_path}")

    setting = setting_from_rows(rows, source_path)
    benchmark = args.benchmark.strip() or str(setting["benchmark"])
    run_id = args.run_id.strip() or f"{normalize_slug(args.model)}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    if args.output_root:
        out_dir = args.output_root.resolve() / run_id
        setting_dir = args.output_root.resolve()
    else:
        setting_dir = retrieval_run_dir(
            root=root,
            benchmark=benchmark,
            agent=args.agent,
            model=args.model,
            arm="agent_pick",
            noise_mode=str(setting["noise_mode"]),
            pool_size=setting["pool_size"],
            seed=setting["seed"],
            run_id=".",
        ).parent
        out_dir = setting_dir / normalize_slug(run_id, default="run")
    result_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"

    api_key_env = args.api_key_env or default_key_env(args.provider)
    api_key = os.environ.get(api_key_env, "")
    if args.agent_mode == "chat" and not args.dry_run and not api_key:
        raise RuntimeError(f"missing API key env: {api_key_env}")
    base_url = args.base_url or default_base_url(args.provider)

    harbor_meta: dict[str, Any] = {}
    if args.agent_mode == "harbor":
        results, harbor_meta = run_harbor_agent_pick(
            args=args,
            rows=rows,
            out_dir=out_dir,
            benchmark=benchmark,
            base_url=base_url,
            api_key_env=api_key_env,
        )
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def evaluate_one(row: dict, attempt_index: int, row_index: int) -> dict:
            raw = ""
            error = ""
            if args.dry_run:
                gt_names = list(row.get("gt_skill_names") or [])
                raw = json.dumps({"skills": gt_names[:1]}, ensure_ascii=False)
            elif args.agent_mode == "command":
                for retry_index in range(1, max(1, args.max_retries) + 1):
                    try:
                        task_slug = normalize_slug(
                            str(row.get("task_name") or f"task-{row_index + 1}"),
                            default=f"task-{row_index + 1}",
                        )
                        workspace = prepare_agent_pick_workspace(out_dir / "workspaces" / f"attempt-{attempt_index}" / task_slug, row)
                        raw = run_agent_pick_command(
                            row=row,
                            workspace=workspace,
                            agent=args.agent,
                            model=args.model,
                            agent_command=args.agent_command,
                            timeout_sec=max(1, args.timeout_sec),
                        )
                        error = ""
                        break
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        if retry_index < max(1, args.max_retries):
                            time.sleep(2 * retry_index)
            else:
                system_prompt, user_message = build_prompt(row)
                for retry_index in range(1, max(1, args.max_retries) + 1):
                    try:
                        raw = chat_completion(
                            base_url=base_url,
                            api_key=api_key,
                            model=args.model,
                            system_prompt=system_prompt,
                            user_message=user_message,
                            timeout_sec=max(1, args.timeout_sec),
                        )
                        error = ""
                        break
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        if retry_index < max(1, args.max_retries):
                            time.sleep(2 * retry_index)

            picked_skills = canonicalize_picked_skills(parse_picked_skills(raw), row)
            if args.agent_mode == "chat" and len(picked_skills) <= 1:
                picked = picked_skills[0] if picked_skills else parse_pick(raw)
                hit, precision, recall = gt_metrics(picked, list(row.get("gt_skill_names") or []))
                picked_skills = [picked] if picked else []
            else:
                picked = picked_skills[0] if picked_skills else ""
                hit, precision, recall = gt_set_metrics(picked_skills, list(row.get("gt_skill_names") or []))
            return {
                "benchmark": row.get("benchmark"),
                "task_name": row.get("task_name"),
                "attempt_index": attempt_index,
                "pool_size": row.get("pool_size"),
                "noise_mode": row.get("noise_mode"),
                "seed": row.get("seed"),
                "gt_skill_names": row.get("gt_skill_names"),
                "candidate_skill_names": [s.get("skill_name") for s in row.get("candidate_skills") or []],
                "picked_skill": picked,
                "picked_skills": picked_skills,
                "hit": hit,
                "precision": precision,
                "recall": recall,
                "refusal": not bool(picked_skills),
                "raw_llm_output": raw,
                "error": error,
            }

        jobs = [(row, attempt_index, row_index) for attempt_index in range(1, max(1, int(args.n_attempts)) + 1) for row_index, row in enumerate(rows)]
        results = []
        if max(1, int(args.n_concurrent)) == 1:
            for row, attempt_index, row_index in jobs:
                results.append(evaluate_one(row, attempt_index, row_index))
        else:
            with ThreadPoolExecutor(max_workers=max(1, int(args.n_concurrent))) as pool:
                future_to_order = {pool.submit(evaluate_one, row, attempt_index, row_index): order for order, (row, attempt_index, row_index) in enumerate(jobs)}
                ordered: list[tuple[int, dict]] = []
                for fut in as_completed(future_to_order):
                    ordered.append((future_to_order[fut], fut.result()))
                results = [item for _, item in sorted(ordered, key=lambda x: x[0])]


    total = len(results)
    hit = sum(1 for r in results if r["hit"])
    refusal = sum(1 for r in results if r["refusal"])
    precision = sum(float(r["precision"]) for r in results) / total if total else 0.0
    recall = sum(float(r["recall"]) for r in results) / total if total else 0.0
    f1 = f1_score(precision, recall)
    write_jsonl(result_path, results)
    write_json(summary_path, {
        "benchmark": benchmark,
        "agent": args.agent,
        "model": args.model,
        "agent_model": agent_model_slug(args.agent, args.model),
        "arm": "agent_pick",
        "noise_mode": setting["noise_mode"],
        "pool_size": setting["pool_size"],
        "seed": setting["seed"],
        "candidate_pool": str(args.candidate_pool.resolve()) if args.candidate_pool else "",
        "skills_manifest": str(args.skills_manifest.resolve()) if args.skills_manifest else "",
        "run_id": run_id,
        "agent_mode": args.agent_mode,
        "agent_command": args.agent_command,
        "provider": args.provider,
        "model": args.model,
        "api_key_env": api_key_env,
        "base_url": base_url,
        "dry_run": bool(args.dry_run),
        "planned_total": len(rows),
        "planned_attempt_total": len(rows) * max(1, int(args.n_attempts)),
        "n_attempts": max(1, int(args.n_attempts)),
        "total": total,
        "hit": hit,
        "hit_at_1": hit / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "refusal": refusal,
        "refusal_rate": refusal / total if total else 0.0,
        "results": str(result_path),
        "harbor": harbor_meta,
    })
    write_latest_pointer(setting_dir, summary_path, run_id)
    print(f"results={result_path}")
    print(f"summary={summary_path}")
    print(f"hit={hit}/{total} ({hit / total if total else 0.0:.4f}) precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} refusal={refusal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
