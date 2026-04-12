"""Test how instruction text preprocessing affects clustering quality.

Variants:
  - full: original instruction (baseline)
  - first_sentence: only the first sentence
  - stripped: strip common boilerplate ("You are given", "Your task is to", etc.)
  - title_only: use task_id (kebab-case task name) as a proxy for keywords
"""
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).parent
DATA = ROOT / "data"
TASKS_FILE = DATA / "tasks.jsonl"
OUT_FILE = DATA / "text_variants_comparison.json"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
THRESHOLDS = [0.40, 0.45, 0.50, 0.55]

BOILERPLATE = [
    r"^you are given\b",
    r"^you are placed\b",
    r"^your task is to\b",
    r"^your job is to\b",
    r"^your goal is to\b",
    r"^you need to\b",
    r"^you must\b",
    r"^please\b",
    r"^the file [^.]+ ",
    r"^there's a [^.]+ ",
    r"^we have\b",
    r"^i want you to\b",
    r"^i need\b",
    r"^the following\b",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE), flags=re.IGNORECASE | re.MULTILINE)


def first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0] if parts else text


def strip_boilerplate(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = []
    for s in sentences:
        cleaned = BOILERPLATE_RE.sub("", s).strip(" ,;:")
        if cleaned and len(cleaned.split()) >= 3:
            kept.append(cleaned)
    out = " ".join(kept) if kept else text
    return out[:1500]


def title_words(task_id: str) -> str:
    return task_id.replace("-", " ").replace("_", " ")


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def last_token_pool(last_hidden, attention_mask):
    seq_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), seq_lengths]


def embed(texts, tokenizer, model, device, batch_size=2, max_len=1024):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            pooled = last_token_pool(h, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            out.append(pooled.cpu().numpy())
            del h, pooled, enc
            if device == "mps":
                torch.mps.empty_cache()
    return np.vstack(out).astype(np.float32)


def evaluate(emb, task_ids, id_to_meta):
    sim = np.clip(emb @ emb.T, -1.0, 1.0)
    dist = (1.0 - sim).astype(np.float64)
    np.fill_diagonal(dist, 0.0)

    rows = []
    for thr in THRESHOLDS:
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=thr, metric="precomputed", linkage="average"
        ).fit_predict(dist)
        clusters = {}
        for tid, lab in zip(task_ids, labels):
            clusters.setdefault(int(lab), []).append(tid)
        multi = [c for c in clusters.values() if len(c) >= 2]
        singletons = sum(1 for c in clusters.values() if len(c) == 1)
        sizes = sorted([len(v) for v in multi], reverse=True)

        purities, jacc = [], []
        for tids in multi:
            cats = [id_to_meta[t].get("category", "?") or "?" for t in tids]
            purities.append(Counter(cats).most_common(1)[0][1] / len(cats))
            tag_sets = [set(id_to_meta[t].get("tags", []) or []) for t in tids]
            ps = [jaccard(a, b) for a, b in combinations(tag_sets, 2)]
            if ps:
                jacc.append(mean(ps))

        rows.append({
            "threshold": thr,
            "n_multi": len(multi),
            "noise_rate": round(singletons / len(task_ids), 3),
            "max_cluster": sizes[0] if sizes else 0,
            "median_cluster": float(median(sizes)) if sizes else 0,
            "category_purity": round(mean(purities), 3) if purities else 0.0,
            "tag_jaccard": round(mean(jacc), 3) if jacc else 0.0,
        })
    return rows


def main():
    tasks = [json.loads(l) for l in open(TASKS_FILE)]
    task_ids = [r["task_id"] for r in tasks]
    id_to_meta = {r["task_id"]: r for r in tasks}

    variants = {
        "full": [r["instruction"] for r in tasks],
        "first_sentence": [first_sentence(r["instruction"]) for r in tasks],
        "stripped": [strip_boilerplate(r["instruction"]) for r in tasks],
        "title_only": [title_words(r["task_id"]) for r in tasks],
    }

    print("\nText variant length stats:")
    for name, texts in variants.items():
        lens = [len(t) for t in texts]
        print(f"  {name:<15} mean={mean(lens):.0f}  median={median(lens):.0f}  max={max(lens)}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nloading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if device == "mps" else torch.float32).to(device).eval()

    results = {}
    for name, texts in variants.items():
        print(f"\nembedding variant '{name}'...")
        emb = embed(texts, tokenizer, model, device)
        rows = evaluate(emb, task_ids, id_to_meta)
        results[name] = rows
        print(f"  {'thr':>5} {'multi':>5} {'noise%':>7} {'max':>4} {'pur':>5} {'jacc':>5}")
        for r in rows:
            print(f"  {r['threshold']:>5.2f} {r['n_multi']:>5} {r['noise_rate']*100:>6.1f}% "
                  f"{r['max_cluster']:>4} {r['category_purity']:>5.2f} {r['tag_jaccard']:>5.2f}")

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
