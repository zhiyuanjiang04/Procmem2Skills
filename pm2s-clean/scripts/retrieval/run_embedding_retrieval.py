#!/usr/bin/env python3
"""Embedding/similarity-based retrieval over candidate skill pools."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from common import (
    DEFAULT_ROOT,
    f1_score,
    iter_jsonl,
    lexical_similarity,
    normalize_slug,
    retrieval_run_dir,
    setting_from_rows,
    write_json,
    write_jsonl,
    write_latest_pointer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run embedding-based skill retrieval.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--agent", default="embedding")
    parser.add_argument("--model", default="")
    parser.add_argument("--method", choices=["lexical", "embedding", "local-embedding"], default="lexical")
    parser.add_argument("--provider", choices=["openai", "openrouter", "google", "uniapi"], default="openai")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--local-model-path", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--query-instruction",
        default="Given a task instruction, retrieve the most relevant skill description that can help solve it",
    )
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def default_key_env(provider: str) -> str:
    return {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "google": "GOOGLE_API_KEY",
        "uniapi": "UNIAPI_API_KEY",
    }[provider]


def default_base_url(provider: str) -> str:
    return {
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "google": "https://generativelanguage.googleapis.com/v1beta/openai",
        "uniapi": "https://api.uniapi.io/v1",
    }[provider]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def load_cache(path: Path | None) -> dict[str, list[float]]:
    if not path or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def save_cache(path: Path | None, cache: dict[str, list[float]]) -> None:
    if not path:
        return
    write_json(path, cache)


def embed_batch(*, base_url: str, api_key: str, model: str, texts: list[str], timeout_sec: int) -> list[list[float]]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/embeddings",
        data=json.dumps({"model": model, "input": texts}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "pm2s-skill-retrieval-embedding/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"embedding HTTPError {exc.code}: {detail[:500]}") from exc
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise RuntimeError(f"embedding response missing data: {str(payload)[:500]}")
    rows = sorted(rows, key=lambda r: int(r.get("index", 0)) if isinstance(r, dict) else 0)
    out: list[list[float]] = []
    for row in rows:
        emb = row.get("embedding") if isinstance(row, dict) else None
        if not isinstance(emb, list):
            raise RuntimeError(f"embedding row missing vector: {str(row)[:200]}")
        out.append([float(x) for x in emb])
    return out


def ensure_embeddings(args: argparse.Namespace, texts: list[str]) -> dict[str, list[float]]:
    cache = load_cache(args.cache_path)
    missing = [t for t in dict.fromkeys(texts) if t not in cache]
    if not missing:
        return cache

    api_key_env = args.api_key_env or default_key_env(args.provider)
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"missing API key env: {api_key_env}")
    base_url = args.base_url or default_base_url(args.provider)
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        last_error = ""
        for attempt in range(1, max(1, int(args.max_retries)) + 1):
            try:
                vectors = embed_batch(
                    base_url=base_url,
                    api_key=api_key,
                    model=args.embedding_model,
                    texts=batch,
                    timeout_sec=max(1, int(args.timeout_sec)),
                )
                for text, vector in zip(batch, vectors):
                    cache[text] = vector
                save_cache(args.cache_path, cache)
                last_error = ""
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_retries:
                    time.sleep(2 * attempt)
        if last_error:
            raise RuntimeError(last_error)
    return cache


def qwen_query_text(text: str, instruction: str) -> str:
    instruction = instruction.strip()
    if not instruction:
        return text
    return f"Instruct: {instruction}\nQuery: {text}"


def local_model_name(args: argparse.Namespace) -> str:
    if args.local_model_path:
        return str(args.local_model_path)
    return "Qwen/Qwen3-Embedding-0.6B"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def last_token_pool(last_hidden_states, attention_mask):
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def encode_local_embeddings(args: argparse.Namespace, texts: list[str]) -> list[list[float]]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    model_name = local_model_name(args)
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    model.to(device)
    model.eval()

    vectors: list[list[float]] = []
    batch_size = max(1, int(args.batch_size))
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max(1, int(args.max_length)),
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            embeddings = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            vectors.extend(embeddings.detach().cpu().float().tolist())
    return vectors


def ensure_local_embeddings(args: argparse.Namespace, texts: list[str]) -> dict[str, list[float]]:
    cache = load_cache(args.cache_path)
    missing = [t for t in dict.fromkeys(texts) if t not in cache]
    if not missing:
        return cache
    vectors = encode_local_embeddings(args, missing)
    for text, vector in zip(missing, vectors):
        cache[text] = vector
    save_cache(args.cache_path, cache)
    return cache


def gt_metrics(picked: str, gt_names: list[str]) -> tuple[bool, float, float]:
    gt = {x.strip().lower() for x in gt_names if str(x).strip()}
    if not picked:
        return False, 0.0, 0.0
    hit = picked.strip().lower() in gt
    precision = 1.0 if hit else 0.0
    recall = (1.0 / max(1, len(gt))) if hit else 0.0
    return hit, precision, recall


def topk_metrics(picked_names: list[str], gt_names: list[str]) -> tuple[bool, float, float, float, int]:
    gt = {x.strip().lower() for x in gt_names if str(x).strip()}
    picked: list[str] = []
    seen: set[str] = set()
    for name in picked_names:
        key = str(name or "").strip().lower()
        if key and key not in seen:
            picked.append(key)
            seen.add(key)
    if not picked or not gt:
        return False, 0.0, 0.0, 0.0, 0
    true_positive = len(set(picked) & gt)
    precision = true_positive / len(picked)
    recall = true_positive / len(gt)
    return true_positive > 0, precision, recall, f1_score(precision, recall), true_positive


def main() -> int:
    args = parse_args()
    rows = list(iter_jsonl(args.candidate_pool.resolve()))
    if args.task_limit:
        rows = rows[: args.task_limit]
    if not rows:
        raise RuntimeError(f"empty candidate pool: {args.candidate_pool}")

    setting = setting_from_rows(rows, args.candidate_pool.resolve())
    benchmark = args.benchmark.strip() or str(setting["benchmark"])
    model = args.model.strip() or (
        args.embedding_model
        if args.method == "embedding"
        else normalize_slug(local_model_name(args)) if args.method == "local-embedding" else args.method
    )
    top_k = max(1, int(args.top_k))
    run_id = args.run_id.strip() or f"{args.method}-top{top_k}-{normalize_slug(model)}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    if args.output_root:
        out_dir = args.output_root.resolve() / run_id
        setting_dir = args.output_root.resolve()
    else:
        setting_dir = retrieval_run_dir(
            root=args.root,
            benchmark=benchmark,
            agent=args.agent,
            model=model,
            arm="embedding_based",
            noise_mode=str(setting["noise_mode"]),
            pool_size=setting["pool_size"],
            seed=setting["seed"],
            run_id=".",
        ).parent
        out_dir = setting_dir / normalize_slug(run_id, default="run")
    result_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"

    texts: list[str] = []
    if args.method in {"embedding", "local-embedding"}:
        for row in rows:
            query = str(row.get("task_description") or row.get("task_name") or "")
            texts.append(qwen_query_text(query, args.query_instruction) if args.method == "local-embedding" else query)
            for skill in row.get("candidate_skills") or []:
                texts.append(str(skill.get("description") or skill.get("skill_name") or ""))
        embeddings = ensure_local_embeddings(args, texts) if args.method == "local-embedding" else ensure_embeddings(args, texts)
    else:
        embeddings = {}

    results: list[dict] = []
    for row in rows:
        query = str(row.get("task_description") or row.get("task_name") or "")
        query_key = qwen_query_text(query, args.query_instruction) if args.method == "local-embedding" else query
        scored: list[tuple[float, dict]] = []
        for skill in row.get("candidate_skills") or []:
            desc = str(skill.get("description") or skill.get("skill_name") or "")
            if args.method in {"embedding", "local-embedding"}:
                score = cosine(embeddings[query_key], embeddings[desc])
            else:
                score = lexical_similarity(query, desc)
            scored.append((float(score), skill))
        scored.sort(key=lambda pair: (pair[0], str(pair[1].get("skill_name") or "")), reverse=True)
        picked_skill = scored[0][1] if scored else {}
        picked_name = str(picked_skill.get("skill_name") or "")
        top_items = scored[:top_k]
        top_names = [str(item.get("skill_name") or "") for _, item in top_items]
        hit, precision, recall, row_f1, true_positive = topk_metrics(top_names, list(row.get("gt_skill_names") or []))
        row_recall = None if top_k == 1 else recall
        row_f1_out = None if top_k == 1 else row_f1
        results.append(
            {
                "benchmark": row.get("benchmark"),
                "task_name": row.get("task_name"),
                "pool_size": row.get("pool_size"),
                "noise_mode": row.get("noise_mode"),
                "seed": row.get("seed"),
                "method": args.method,
                "top_k": top_k,
                "embedding_model": args.embedding_model if args.method == "embedding" else None,
                "gt_skill_names": row.get("gt_skill_names"),
                "picked_skill": picked_name,
                "picked_role": picked_skill.get("role"),
                "picked_score": scored[0][0] if scored else None,
                "picked_topk": top_names,
                "true_positive": true_positive,
                "hit": hit,
                "precision": precision,
                "recall": row_recall,
                "f1": row_f1_out,
                "top5": [
                    {"skill_name": item.get("skill_name"), "role": item.get("role"), "score": score}
                    for score, item in scored[:5]
                ],
            }
        )

    total = len(results)
    hit = sum(1 for r in results if r["hit"])
    precision = sum(float(r["precision"]) for r in results) / total if total else 0.0
    if top_k == 1:
        recall = None
        f1 = None
    else:
        recall = sum(float(r["recall"]) for r in results if r["recall"] is not None) / total if total else 0.0
        f1 = sum(float(r["f1"]) for r in results if r["f1"] is not None) / total if total else 0.0
    write_jsonl(result_path, results)
    write_json(
        summary_path,
        {
            "benchmark": benchmark,
            "agent": args.agent,
            "model": model,
            "agent_model": normalize_slug(f"{args.agent}-{model.rsplit('/', 1)[-1]}", default="agent-model"),
            "arm": "embedding_based",
            "noise_mode": setting["noise_mode"],
            "pool_size": setting["pool_size"],
            "seed": setting["seed"],
            "candidate_pool": str(args.candidate_pool.resolve()),
            "run_id": run_id,
            "method": args.method,
            "top_k": top_k,
            "provider": args.provider if args.method == "embedding" else "local" if args.method == "local-embedding" else None,
            "embedding_model": args.embedding_model if args.method == "embedding" else local_model_name(args) if args.method == "local-embedding" else None,
            "local_model_path": str(args.local_model_path.resolve()) if args.local_model_path else None,
            "device": args.device if args.method == "local-embedding" else None,
            "max_length": args.max_length if args.method == "local-embedding" else None,
            "query_instruction": args.query_instruction if args.method == "local-embedding" else None,
            "total": total,
            "hit": hit,
            "hit_at_1": (hit / total if total else 0.0) if top_k == 1 else None,
            "hit_at_k": hit / total if total else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "results": str(result_path),
        },
    )
    write_latest_pointer(setting_dir, summary_path, run_id)
    print(f"results={result_path}")
    print(f"summary={summary_path}")
    if top_k == 1:
        print(f"hit@1={hit}/{total} ({hit / total if total else 0.0:.4f}) precision={precision:.4f} recall=NA f1=NA")
    else:
        print(f"hit@{top_k}={hit}/{total} ({hit / total if total else 0.0:.4f}) precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
