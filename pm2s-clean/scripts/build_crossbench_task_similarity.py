#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CLEAN_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TaskItem:
    task_uid: str
    task_name: str
    benchmark_id: str
    benchmark_slug: str
    source_task_dir: Path
    instruction_text: str


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


def _unique_tasks_from_traces(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for rec in records:
        task_name = str(rec.get("task_name") or "").strip()
        benchmark_id = str(rec.get("benchmark") or "").strip()
        if not task_name or not benchmark_id:
            continue
        key = (benchmark_id, task_name)
        if key not in out:
            out[key] = {"benchmark_id": benchmark_id, "task_name": task_name}
    return out


def _read_instruction(task_dir: Path, *, max_chars: int) -> str:
    instruction = task_dir / "instruction.md"
    if not instruction.is_file():
        return ""
    text = instruction.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _find_task_source_dir(
    *,
    benchmark_id: str,
    task_name: str,
    task_source_root: Path,
    skillsbench_task_root: Path,
) -> Path | None:
    roots: list[Path] = []
    bid = benchmark_id.strip().lower()
    if bid == "skillsbench":
        roots.extend(
            [
                skillsbench_task_root,
                Path("/raid/zhiyuan/procmem2skills/benchmarks/skillsbench/tasks"),
                Path("/raid/zhiyuan/procmem2skills/benchmarks/skillsbench"),
            ]
        )
    else:
        roots.append(task_source_root)

    for root in roots:
        if not root.exists():
            continue
        direct = root / task_name
        if direct.is_dir():
            return direct
        nested = root / "tasks" / task_name
        if nested.is_dir():
            return nested
        hits = sorted(root.glob(f"*/{task_name}"))
        if hits:
            return hits[0]
    return None


def _encode_texts(
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
            "embedding dependencies unavailable. Use /raid/zhiyuan/procmem2skills/.venv-py312/bin/python"
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cross-benchmark task similarity JSON using Qwen embeddings over task name + description."
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=CLEAN_ROOT / "retrieval-data" / "traces",
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
    parser.add_argument(
        "--skillsbench-task-root",
        type=Path,
        default=CLEAN_ROOT / "retrieval-data" / "skillsbench-tasks-no-skills",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-max-length", type=int, default=2048)
    parser.add_argument("--max-desc-chars", type=int, default=4000)
    parser.add_argument("--task-name", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if int(args.top_n) <= 0:
        raise RuntimeError("--top-n must be > 0")

    trace_root = args.trace_root.resolve()
    benchmark_map = _load_benchmark_map(args.benchmark_config.resolve())

    records = _load_traces(trace_root)
    unique = _unique_tasks_from_traces(records)
    task_filter = {t.strip() for t in (args.task_name or []) if t.strip()}

    items: list[TaskItem] = []
    skipped: list[dict[str, str]] = []

    for (benchmark_id, task_name), _ in sorted(unique.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if task_filter and task_name not in task_filter:
            continue

        benchmark_slug = _benchmark_output_slug(benchmark_id=benchmark_id, benchmark_cfg=benchmark_map.get(benchmark_id))
        source_dir = _find_task_source_dir(
            benchmark_id=benchmark_id,
            task_name=task_name,
            task_source_root=args.task_source_root.resolve(),
            skillsbench_task_root=args.skillsbench_task_root.resolve(),
        )
        if source_dir is None:
            skipped.append({"benchmark_id": benchmark_id, "task_name": task_name, "reason": "missing_source_dir"})
            continue

        description = _read_instruction(source_dir, max_chars=max(256, int(args.max_desc_chars)))
        if not description:
            skipped.append({"benchmark_id": benchmark_id, "task_name": task_name, "reason": "missing_instruction"})
            continue

        task_uid = f"{benchmark_id}::{task_name}"
        items.append(
            TaskItem(
                task_uid=task_uid,
                task_name=task_name,
                benchmark_id=benchmark_id,
                benchmark_slug=benchmark_slug,
                source_task_dir=source_dir,
                instruction_text=description,
            )
        )

    if len(items) < 2:
        raise RuntimeError(f"need at least 2 tasks; got {len(items)} usable tasks")

    texts = [f"Task: {it.task_name}\nBenchmark: {it.benchmark_id}\nDescription: {it.instruction_text}" for it in items]
    embeddings = _encode_texts(
        texts,
        model_name=str(args.embedding_model),
        device=str(args.embedding_device),
        batch_size=max(1, int(args.embedding_batch_size)),
        max_length=max(32, int(args.embedding_max_length)),
    )

    n = len(items)
    top_n = min(max(1, int(args.top_n)), n - 1)
    sims: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = _dot(embeddings[i], embeddings[j])
            sims[i][j] = s
            sims[j][i] = s

    rows: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        pairs: list[tuple[int, float]] = []
        for j in range(n):
            if i == j:
                continue
            pairs.append((j, sims[i][j]))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top = pairs[:top_n]

        rows.append(
            {
                "task_uid": item.task_uid,
                "task_name": item.task_name,
                "benchmark_id": item.benchmark_id,
                "benchmark_slug": item.benchmark_slug,
                "source_task_dir": str(item.source_task_dir),
                "description_excerpt": item.instruction_text[:300],
                "similar_tasks": [
                    {
                        "rank": rank,
                        "task_uid": items[j].task_uid,
                        "task_name": items[j].task_name,
                        "benchmark_id": items[j].benchmark_id,
                        "benchmark_slug": items[j].benchmark_slug,
                        "similarity": float(score),
                    }
                    for rank, (j, score) in enumerate(top, start=1)
                ],
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trace_root": str(trace_root),
        "benchmark_config": str(args.benchmark_config.resolve()),
        "embedding_model": str(args.embedding_model),
        "embedding_device": str(args.embedding_device),
        "top_n": top_n,
        "task_count": len(items),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "tasks": rows,
    }

    if not args.dry_run:
        _json_dump(args.output_json.resolve(), payload)

    print(
        json.dumps(
            {
                "task_count": len(items),
                "top_n": top_n,
                "output_json": str(args.output_json.resolve()),
                "skipped_count": len(skipped),
                "dry_run": bool(args.dry_run),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
