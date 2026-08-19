#!/usr/bin/env python3
"""Shared utilities for the skill retrieval pipeline."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOT = Path(os.environ.get("PM2S_RETRIEVAL_ROOT", str(Path(__file__).resolve().parents[2] / "retrieval-data")))
DEFAULT_SKILLSBENCH_TASKS = Path(
    os.environ.get("SKILLSBENCH_TASKS_ROOT", "/raid/zhiyuan/pm2s/skillsbench/tasks")
)


def normalize_slug(text: str, default: str = "item") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text or "").strip()).strip("-._")
    return slug or default


def model_leaf(model: str) -> str:
    return str(model or "").strip().rsplit("/", 1)[-1]


def agent_model_slug(agent: str, model: str) -> str:
    return normalize_slug(f"{agent}-{model_leaf(model)}", default="agent-model")


def f1_score(precision: float, recall: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def candidate_setting_from_path(path: Path) -> dict[str, Any]:
    """Infer benchmark/noise/k/seed from candidate pool or execution manifest paths."""
    parts = path.resolve().parts
    setting: dict[str, Any] = {}
    for marker in ("candidate_pools", "execution_manifests"):
        if marker not in parts:
            continue
        idx = parts.index(marker)
        if len(parts) > idx + 1:
            setting["benchmark"] = parts[idx + 1]
        if len(parts) > idx + 2:
            setting["noise_mode"] = parts[idx + 2]
        if len(parts) > idx + 3:
            match = re.fullmatch(r"k(\d+)", parts[idx + 3])
            if match:
                setting["pool_size"] = int(match.group(1))
        if len(parts) > idx + 4:
            match = re.match(r"seed-(\d+)", parts[idx + 4])
            if match:
                setting["seed"] = int(match.group(1))
        break
    return setting


def setting_from_rows(rows: list[dict[str, Any]], fallback_path: Path | None = None) -> dict[str, Any]:
    setting = candidate_setting_from_path(fallback_path) if fallback_path else {}
    if rows:
        first = rows[0]
        for key in ("benchmark", "noise_mode", "pool_size", "seed"):
            value = first.get(key)
            if value is not None and value != "":
                setting[key] = value
    setting.setdefault("benchmark", "skillsbench")
    setting.setdefault("noise_mode", "unknown")
    setting.setdefault("pool_size", 0)
    setting.setdefault("seed", 0)
    return setting


def retrieval_setting_dir(
    *,
    root: Path,
    benchmark: str,
    agent: str,
    model: str,
    arm: str,
    noise_mode: str,
    pool_size: int | str,
    seed: int | str,
) -> Path:
    return (
        root.resolve()
        / "outputs"
        / normalize_slug(benchmark, default="benchmark")
        / agent_model_slug(agent, model)
        / normalize_slug(arm, default="arm")
        / normalize_slug(noise_mode, default="mode")
        / f"k{pool_size}"
        / f"seed-{seed}"
    )


def retrieval_run_dir(
    *,
    root: Path,
    benchmark: str,
    agent: str,
    model: str,
    arm: str,
    noise_mode: str,
    pool_size: int | str,
    seed: int | str,
    run_id: str,
) -> Path:
    return retrieval_setting_dir(
        root=root,
        benchmark=benchmark,
        agent=agent,
        model=model,
        arm=arm,
        noise_mode=noise_mode,
        pool_size=pool_size,
        seed=seed,
    ) / normalize_slug(run_id, default="run")


def write_latest_pointer(setting_dir: Path, summary_path: Path, run_id: str) -> None:
    write_json(
        setting_dir / "latest.json",
        {
            "run_id": run_id,
            "run_dir": str(summary_path.parent.resolve()),
            "summary": str(summary_path.resolve()),
        },
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path, limit: int | None = None) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    if limit is not None and len(text) > limit:
        return text[:limit]
    return text


def parse_frontmatter(text: str) -> dict[str, str]:
    """Small YAML-frontmatter parser for common scalar/block description fields."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    block = text[3:end].strip("\n")
    out: dict[str, str] = {}
    lines = block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if ":" not in line or line.startswith((" ", "\t")):
            i += 1
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value in {"|", ">"}:
            i += 1
            chunks: list[str] = []
            while i < len(lines) and (lines[i].startswith(" ") or lines[i].startswith("\t") or not lines[i].strip()):
                chunks.append(lines[i].strip())
                i += 1
            out[key] = "\n".join(chunks).strip()
            continue
        out[key] = value.strip("\"'")
        i += 1
    return out


def skill_description(skill_md: Path, metadata: dict[str, Any] | None = None, *, max_chars: int = 2000) -> str:
    if metadata:
        desc = str(metadata.get("description") or "").strip()
        if desc:
            return desc[:max_chars]
    text = read_text(skill_md, limit=12000)
    fm = parse_frontmatter(text)
    desc = str(fm.get("description") or "").strip()
    if desc:
        return desc[:max_chars]
    body = re.sub(r"^---.*?---", "", text, flags=re.S).strip()
    body = re.sub(r"\s+", " ", body)
    return body[:max_chars]


def skill_name_from_md(skill_md: Path) -> str:
    text = read_text(skill_md, limit=4000)
    fm = parse_frontmatter(text)
    return str(fm.get("name") or skill_md.parent.name).strip() or skill_md.parent.name


def task_description(task_dir: Path, *, max_chars: int = 6000) -> str:
    instruction = read_text(task_dir / "instruction.md", limit=max_chars)
    if instruction.strip():
        return instruction.strip()
    toml = read_text(task_dir / "task.toml", limit=max_chars)
    if toml.strip():
        return toml.strip()
    return task_dir.name


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text or ""))


def tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", (text or "").lower())
        if tok not in {"the", "and", "for", "with", "that", "this", "from", "into", "using", "use"}
    }


def lexical_similarity(a: str, b: str) -> float:
    ta = tokens(a)
    tb = tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / ((len(ta) * len(tb)) ** 0.5)
