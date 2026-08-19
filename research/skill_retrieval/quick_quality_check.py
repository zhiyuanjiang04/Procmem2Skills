"""Lightweight sanity check on the noise pool (no qwen / no sklearn).

1. Pool quality stats: counts, empty desc, desc length, language mix, downloads.
2. BM25 retrieval over skill descriptions for a few SkillsBench tasks, so a human
   can eyeball whether top hits are plausibly related.

This is a sanity check only; the real experiment uses qwen3-embedding.
"""
import glob
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path.home() / "Desktop" / "skill_retrieval"
POOL = ROOT / "noise_pool"
TASKS = Path.home() / "Desktop" / "Procmem2Skills" / "testsets" / "data" / "skillsbench_tasks.jsonl"

CJK = re.compile(r"[一-鿿]")


def tok(s: str) -> list[str]:
    s = (s or "").lower()
    words = re.findall(r"[a-z0-9]+", s)
    cjk = CJK.findall(s)
    return words + cjk


# ── load pool ────────────────────────────────────────────────────────────
metas = []
for p in glob.glob(str(POOL / "*" / "metadata.json")):
    d = json.load(open(p))
    metas.append((d["skill_name"], d.get("description", ""), d.get("downloads", 0)))

N = len(metas)
print(f"=== POOL QUALITY ({N} skills) ===")
descs = [m[1] for m in metas]
empty = sum(1 for d in descs if len(d) < 5)
lens = sorted(len(d) for d in descs)
cjk_docs = sum(1 for d in descs if CJK.search(d))
dls = sorted((m[2] for m in metas), reverse=True)


def pct(arr, q):
    return arr[min(len(arr) - 1, int(len(arr) * q))]


print(f"empty/short desc : {empty} ({100*empty/N:.1f}%)")
print(f"desc len p10/p50/p90 : {pct(lens,.1)}/{pct(lens,.5)}/{pct(lens,.9)} chars")
print(f"contains CJK     : {cjk_docs} ({100*cjk_docs/N:.1f}%)  [rest mostly english]")
print(f"downloads max/med/min(top10k) : {dls[0]}/{dls[len(dls)//2]}/{dls[-1]}")

# ── BM25 ─────────────────────────────────────────────────────────────────
docs = [tok(d) for d in descs]
df = Counter()
for doc in docs:
    for t in set(doc):
        df[t] += 1
idf = {t: math.log(1 + (N - f + 0.5) / (f + 0.5)) for t, f in df.items()}
avgdl = sum(len(d) for d in docs) / max(N, 1)
doc_tf = [Counter(d) for d in docs]
doc_len = [len(d) for d in docs]
K1, B = 1.5, 0.75


def bm25_top(qtoks, k=8):
    scores = []
    qset = set(qtoks)
    for i in range(N):
        tf = doc_tf[i]
        dl = doc_len[i]
        s = 0.0
        for t in qset:
            f = tf.get(t)
            if not f:
                continue
            s += idf.get(t, 0.0) * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / avgdl))
        if s > 0:
            scores.append((s, i))
    scores.sort(reverse=True)
    return scores[:k]


tasks = [json.loads(l) for l in open(TASKS) if l.strip()]
rng = random.Random(42)
picked = rng.sample(tasks, 6)

print("\n=== BM25 TOP-8 PER TASK (sanity: are these plausibly related?) ===")
for t in picked:
    q = tok(t["task_description"])
    top = bm25_top(q, 8)
    print(f"\n### TASK: {t['task_id']}   gt_skills={t['gt_skills']}")
    print(f"    desc: {t['task_description'][:200].strip()}")
    for rank, (sc, i) in enumerate(top, 1):
        name, desc, _ = metas[i]
        print(f"    {rank:2d}. [{sc:5.1f}] {name}: {desc[:90].strip()}")
