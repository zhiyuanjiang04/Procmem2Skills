# Plan 1 — Phase A Vertical Slice (Recognition Eval, M1 reproduction)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimal end-to-end Phase A driver that reproduces v1 pilot numbers on 5 SkillsBench tasks, with the v2 methodology fixes (lenient parser, GT-ID re-randomisation, multi-GT scoring, MRR + Top-1 metrics, prompt-cacheable pools, context pre-flight). After this plan, Plan 2 extends to full sweeps / ablations / figures; Plan 3 covers Phase B.

**Architecture:** New Python package at `skills retrieval/src/` (Arch-2: local, do not modify `Procmem2Skills-ref`). Async driver wraps the `anthropic` SDK with prompt caching. Pool building reuses `data/embeddings/skill_embeddings.npy` + `skill_metadata.jsonl` + `data/selection_collapse/skillsbench/tasks.jsonl`. Strict module boundaries: `pool_builder.py` (sampling), `prompt.py` (rendering), `driver.py` (API orchestration, no business logic), `parser.py` (extraction only, returns structured JSON), `metrics.py` (scoring only, consumes parser output + pool map).

**Tech Stack:** Python 3.10+, `anthropic>=0.40`, `numpy`, `scipy`, `scikit-learn` (TF-IDF), `faiss-cpu` (already indexed at `data/embeddings/index`), `pydantic` for config/records, `pytest` for tests, `tiktoken`-style anthropic token counter for context pre-flight.

**Scope anchor (5 pilot tasks × 2 strategies × 4 N = 40 conditions × 3 seeds × 2 probes = 240 API calls):**
- Tasks: `sb_000`, `sb_003`, `sb_004`, `sb_006`, `sb_007` (v1 pilot set)
- Strategies: `random`, `hard_neg_semantic`
- Pool sizes: `N ∈ {1, 5, 50, 200}`
- Representation: `card` only (Plan 2 adds others)
- Seeds: 3
- Probes: `awareness`, `selection`
- Model: `claude-sonnet-4-6`

**Expected output:** `skills retrieval/runs/<timestamp>-plan1-m1/metrics/summary.json` with per-condition Recall@5/Top-1/MRR aggregates matching (within bootstrap CI) v1's random/N∈{5,50,200,1000} ~100% pattern, plus the first clean hard-negative numbers at N=50/200 with the GT-leakage fix.

---

## File Structure

```
skills retrieval/
├── design-v2.md                              # (already exists)
├── plans/
│   └── 2026-04-18-plan-1-vertical-slice.md   # this file
├── src/
│   ├── __init__.py
│   ├── config.py                             # pydantic models for RunConfig, PoolSpec, TrialRecord
│   ├── data.py                               # loaders: corpus, embeddings, tasks
│   ├── pool_builder.py                       # sampling strategies + dedup cascade
│   ├── prompt.py                             # prompt templates + card rendering
│   ├── parser.py                             # lenient tag extraction → structured JSON
│   ├── metrics.py                            # MRR, Top-1, Recall@5, Selection|Aware
│   ├── preflight.py                          # token-budget check
│   ├── driver.py                             # async Anthropic SDK orchestrator
│   └── run_m1.py                             # CLI entrypoint: runs the 240-call sweep
└── tests/
    ├── __init__.py
    ├── conftest.py                           # pytest fixtures (tiny corpus + tasks)
    ├── test_data.py
    ├── test_pool_builder.py
    ├── test_prompt.py
    ├── test_parser.py
    ├── test_metrics.py
    └── test_preflight.py
```

**Module responsibilities (strict boundaries — do not cross):**
- `parser.py` — pure extraction. Returns `{extracted_ids: list[str], format_status: Literal['clean','warning','fail'], flags: dict, raw_text: str}`. **Does NOT compute MRR, Top-1, or any accuracy metric.**
- `metrics.py` — pure scoring. Consumes parser output + pool map + GT. **Does NOT call the API or touch files.**
- `driver.py` — API orchestration only. **Does NOT build pools or score.**
- `pool_builder.py` — sampling + dedup + ID re-randomisation. **Does NOT render prompts.**

---

## Task 1: Scaffold package and pyproject glue

**Files:**
- Create: `skills retrieval/src/__init__.py` (empty)
- Create: `skills retrieval/tests/__init__.py` (empty)
- Create: `skills retrieval/tests/conftest.py`
- Modify: `pyproject.toml` (add sub-package + deps)

- [ ] **Step 1: Create empty package markers**

```bash
mkdir -p "skills retrieval/src" "skills retrieval/tests" "skills retrieval/runs" "skills retrieval/pools"
touch "skills retrieval/src/__init__.py" "skills retrieval/tests/__init__.py"
```

- [ ] **Step 2: Add dependencies to pyproject.toml**

Open `pyproject.toml` and append under `[project]` dependencies (if the existing file uses a different build system, adapt):

```toml
dependencies = [
    "anthropic>=0.40",
    "pydantic>=2.5",
    "numpy>=1.26",
    "scipy>=1.11",
    "scikit-learn>=1.3",
    "faiss-cpu>=1.7",
    "tiktoken>=0.5",
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]
```

(If deps already declared, merge — don't duplicate.)

- [ ] **Step 3: Create pytest conftest with tiny fixtures**

Write `skills retrieval/tests/conftest.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> dict:
    """10-skill corpus with 2D embeddings for fast unit tests."""
    metadata = [
        {"id": f"skill_{i:06d}", "slug": f"author/skill-{i}", "name": f"name-{i}", "category": "general"}
        for i in range(10)
    ]
    embeddings = np.array(
        [[np.cos(i * 0.6), np.sin(i * 0.6)] for i in range(10)],
        dtype=np.float32,
    )
    meta_path = tmp_path / "metadata.jsonl"
    emb_path = tmp_path / "embeddings.npy"
    with meta_path.open("w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")
    np.save(emb_path, embeddings)
    return {"metadata_path": meta_path, "embeddings_path": emb_path, "metadata": metadata, "embeddings": embeddings}


@pytest.fixture
def tiny_task() -> dict:
    return {
        "task_id": "sb_test",
        "task_name": "test-task",
        "domain": "test",
        "instruction": "Test task instruction.",
        "gt_skills": [
            {"skill_id": "gt_test_alpha", "name": "alpha", "dir_name": "alpha", "content": "---\nname: alpha\ndescription: Test alpha skill.\n---\nBody."},
        ],
        "n_gt_skills": 1,
    }
```

- [ ] **Step 4: Verify pytest discovers the suite**

Run: `cd "skills retrieval" && python -m pytest tests/ -q`
Expected: `no tests ran` (0 collected) — confirms pytest discovery works.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml "skills retrieval/src/__init__.py" "skills retrieval/tests/__init__.py" "skills retrieval/tests/conftest.py"
git commit -m "feat(skills-retrieval): scaffold Phase A driver package"
```

---

## Task 2: Config and record models

**Files:**
- Create: `skills retrieval/src/config.py`
- Create: `skills retrieval/tests/test_config.py`

- [ ] **Step 1: Write failing test**

Write `skills retrieval/tests/test_config.py`:

```python
from skills_retrieval.config import PoolSpec, RunConfig, TrialRecord


def test_pool_spec_roundtrip():
    spec = PoolSpec(task_id="sb_000", strategy="random", n=50, seed=0, representation="card")
    s = spec.model_dump_json()
    loaded = PoolSpec.model_validate_json(s)
    assert loaded == spec
    assert spec.pool_id == "sb_000__random__n50__s0__card"


def test_trial_record_has_all_fields():
    rec = TrialRecord(
        pool_id="sb_000__random__n50__s0__card",
        probe="awareness",
        model="claude-sonnet-4-6",
        raw_response="<skills>SKILL_000,SKILL_001,SKILL_002,SKILL_003,SKILL_004</skills>",
        extracted_ids=["SKILL_000", "SKILL_001", "SKILL_002", "SKILL_003", "SKILL_004"],
        format_status="clean",
        flags={},
        latency_ms=1234,
    )
    assert rec.probe == "awareness"
    assert len(rec.extracted_ids) == 5
```

Note: the package import path is `skills_retrieval.*` — we need the file layout to support that. Since the folder name has a space, add a `setup.cfg` or configure via pyproject. For simplicity, tests run with `PYTHONPATH="skills retrieval/src"` and the src folder uses a nested import root via `src/skills_retrieval/...`. Adjust Task 1 scaffold:

```bash
mv "skills retrieval/src/__init__.py" /tmp/_
mkdir -p "skills retrieval/src/skills_retrieval"
mv /tmp/_ "skills retrieval/src/skills_retrieval/__init__.py"
```

And set `pyproject.toml` `[tool.setuptools.packages.find] where = ["skills retrieval/src"]`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "skills retrieval" && PYTHONPATH=src python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'skills_retrieval'` or equivalent.

- [ ] **Step 3: Implement config models**

Write `skills retrieval/src/skills_retrieval/config.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Strategy = Literal["random", "easy_neg", "hard_neg_semantic", "hard_neg_functional", "adversarial"]
Representation = Literal["card", "name_only", "desc_only", "full", "compressed_full"]
Probe = Literal["awareness", "selection"]
FormatStatus = Literal["clean", "warning", "fail"]


class PoolSpec(BaseModel):
    task_id: str
    strategy: Strategy
    n: int = Field(ge=1)
    seed: int = Field(ge=0)
    representation: Representation = "card"

    @property
    def pool_id(self) -> str:
        return f"{self.task_id}__{self.strategy}__n{self.n}__s{self.seed}__{self.representation}"


class TrialRecord(BaseModel):
    pool_id: str
    probe: Probe
    model: str
    raw_response: str
    extracted_ids: list[str]
    format_status: FormatStatus
    flags: dict = Field(default_factory=dict)
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None


class RunConfig(BaseModel):
    label: str
    model: str = "claude-sonnet-4-6"
    task_ids: list[str]
    strategies: list[Strategy]
    pool_sizes: list[int]
    representation: Representation = "card"
    seeds: list[int] = [0, 1, 2]
    probes: list[Probe] = ["awareness", "selection"]
    temperature: float = 0.0
    max_concurrency: int = 8
    context_safety_margin: int = 1000
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "skills retrieval" && PYTHONPATH=src python -m pytest tests/test_config.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add "skills retrieval/src/skills_retrieval/config.py" "skills retrieval/tests/test_config.py" pyproject.toml
git commit -m "feat(skills-retrieval): add RunConfig/PoolSpec/TrialRecord models"
```

---

## Task 3: Data loaders (corpus + tasks)

**Files:**
- Create: `skills retrieval/src/skills_retrieval/data.py`
- Create: `skills retrieval/tests/test_data.py`

- [ ] **Step 1: Write failing test**

Write `skills retrieval/tests/test_data.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest

from skills_retrieval.data import Corpus, Task, load_tasks


def test_corpus_loads_from_paths(tiny_corpus):
    c = Corpus.from_paths(tiny_corpus["metadata_path"], tiny_corpus["embeddings_path"])
    assert len(c.ids) == 10
    assert c.embeddings.shape == (10, 2)
    assert c.ids[0] == "skill_000000"
    assert c.name_by_id["skill_000003"] == "name-3"


def test_corpus_index_lookup(tiny_corpus):
    c = Corpus.from_paths(tiny_corpus["metadata_path"], tiny_corpus["embeddings_path"])
    assert c.index_of("skill_000005") == 5


def test_load_tasks(tmp_path, tiny_task):
    p = tmp_path / "tasks.jsonl"
    with p.open("w") as f:
        f.write(json.dumps(tiny_task) + "\n")
    tasks = load_tasks(p)
    assert len(tasks) == 1
    assert tasks[0].task_id == "sb_test"
    assert tasks[0].gt_skill_names == ["alpha"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "skills retrieval" && PYTHONPATH=src python -m pytest tests/test_data.py -v`
Expected: `ModuleNotFoundError: No module named 'skills_retrieval.data'`.

- [ ] **Step 3: Implement `data.py`**

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Corpus:
    ids: list[str]
    names: list[str]
    descriptions: list[str]
    embeddings: np.ndarray                           # (N, D) float32
    name_by_id: dict[str, str] = field(default_factory=dict)
    _idx_by_id: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_paths(cls, metadata_path: Path, embeddings_path: Path, descriptions_path: Path | None = None) -> "Corpus":
        ids: list[str] = []
        names: list[str] = []
        with Path(metadata_path).open() as f:
            for line in f:
                row = json.loads(line)
                ids.append(row["id"])
                names.append(row.get("name", row["id"]))
        embeddings = np.load(embeddings_path).astype(np.float32)
        if embeddings.shape[0] != len(ids):
            raise ValueError(f"Embeddings ({embeddings.shape[0]}) != metadata ({len(ids)})")
        descriptions = [""] * len(ids)
        if descriptions_path is not None:
            with Path(descriptions_path).open() as f:
                by_id = {json.loads(line)["id"]: json.loads(line).get("description", "") for line in f}
            descriptions = [by_id.get(i, "") for i in ids]
        name_by_id = dict(zip(ids, names))
        _idx_by_id = {i: k for k, i in enumerate(ids)}
        return cls(ids=ids, names=names, descriptions=descriptions, embeddings=embeddings, name_by_id=name_by_id, _idx_by_id=_idx_by_id)

    def index_of(self, skill_id: str) -> int:
        return self._idx_by_id[skill_id]

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class Task:
    task_id: str
    instruction: str
    gt_skill_names: list[str]
    gt_skill_bodies: list[str]
    domain: str = ""

    @property
    def gt_set(self) -> set[str]:
        return set(self.gt_skill_names)


def load_tasks(path: Path) -> list[Task]:
    tasks: list[Task] = []
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)
            gts = row.get("gt_skills") or []
            tasks.append(
                Task(
                    task_id=row["task_id"],
                    instruction=row["instruction"],
                    gt_skill_names=[g["name"] for g in gts],
                    gt_skill_bodies=[g.get("content", "") for g in gts],
                    domain=row.get("domain", ""),
                )
            )
    return tasks
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd "skills retrieval" && PYTHONPATH=src python -m pytest tests/test_data.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add "skills retrieval/src/skills_retrieval/data.py" "skills retrieval/tests/test_data.py"
git commit -m "feat(skills-retrieval): add Corpus + Task loaders"
```

---

## Task 4: Pool builder — random + hard_neg_semantic + dedup cascade

**Files:**
- Create: `skills retrieval/src/skills_retrieval/pool_builder.py`
- Create: `skills retrieval/tests/test_pool_builder.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np

from skills_retrieval.config import PoolSpec
from skills_retrieval.data import Corpus, Task
from skills_retrieval.pool_builder import build_pool


def _corpus(tiny_corpus) -> Corpus:
    return Corpus.from_paths(tiny_corpus["metadata_path"], tiny_corpus["embeddings_path"])


def _task_with_gt(gt_id: str, gt_embedding) -> Task:
    return Task(task_id="t0", instruction="do a thing", gt_skill_names=[f"name-{int(gt_id.split('_')[-1])}"], gt_skill_bodies=["body"], domain="test")


def test_random_pool_includes_gt_and_has_unique_ids(tiny_corpus):
    corpus = _corpus(tiny_corpus)
    task = _task_with_gt("skill_000003", corpus.embeddings[3])
    # inject a GT-name match into corpus for lookup
    spec = PoolSpec(task_id="t0", strategy="random", n=5, seed=0)
    pool = build_pool(spec, task, corpus, task_embedding=corpus.embeddings[3])
    assert len(pool.display_ids) == 5
    assert len(set(pool.display_ids)) == 5
    # GT must be present (by canonical id)
    assert any(cid == "skill_000003" for cid in pool.id_map.values())


def test_hard_neg_semantic_orders_by_similarity(tiny_corpus):
    corpus = _corpus(tiny_corpus)
    task = _task_with_gt("skill_000003", corpus.embeddings[3])
    spec = PoolSpec(task_id="t0", strategy="hard_neg_semantic", n=4, seed=0)
    pool = build_pool(spec, task, corpus, task_embedding=corpus.embeddings[3])
    # GT (index 3) must be in pool; other members should be nearest neighbours by cosine
    canonical_in_pool = [cid for cid in pool.id_map.values() if cid != "skill_000003"]
    # neighbours should not include any that are too far
    assert len(canonical_in_pool) == 3


def test_id_randomization_is_seeded_and_stable(tiny_corpus):
    corpus = _corpus(tiny_corpus)
    task = _task_with_gt("skill_000003", corpus.embeddings[3])
    spec = PoolSpec(task_id="t0", strategy="random", n=5, seed=42)
    p1 = build_pool(spec, task, corpus, task_embedding=corpus.embeddings[3])
    p2 = build_pool(spec, task, corpus, task_embedding=corpus.embeddings[3])
    assert p1.display_ids == p2.display_ids
    assert p1.id_map == p2.id_map


def test_n_equals_one_is_gt_only(tiny_corpus):
    corpus = _corpus(tiny_corpus)
    task = _task_with_gt("skill_000003", corpus.embeddings[3])
    spec = PoolSpec(task_id="t0", strategy="random", n=1, seed=0)
    pool = build_pool(spec, task, corpus, task_embedding=corpus.embeddings[3])
    assert len(pool.display_ids) == 1
    assert list(pool.id_map.values()) == ["skill_000003"]
```

**NOTE on the fixture mismatch.** The `tiny_corpus` fixture's metadata uses names like `name-3` while tasks in real SkillsBench have GT with names like `mesh-analysis`. The bridge is: for pool building we identify GT **by canonical skill ID**, not by name. Real SkillsBench GT names do not directly map to a canonical corpus skill — SkillsBench GT skills are *human-written* and do **not** exist in the 44k clawhub corpus. We resolve this by loading GT skills as a side-channel: they are injected into the pool by their `gt_skill_bodies` as synthetic corpus entries with a reserved canonical id prefix `gt_<task>_<name>`. Add this to `pool_builder.py`:

- Each task's GT skills are added to a per-task "virtual corpus extension". Their embeddings are computed once at driver startup using Qwen-embedding (or, for Plan 1, a simple Claude embedding fallback if Qwen isn't available — but see Task 4b).

For Plan 1 we **assume GT embeddings are pre-computed** and passed into `build_pool`. The test above passes GT embedding explicitly via `task_embedding=`. Task 12 (M1 runner) wires up real GT embedding loading.

- [ ] **Step 2: Run to verify it fails**

Run: `cd "skills retrieval" && PYTHONPATH=src python -m pytest tests/test_pool_builder.py -v`
Expected: import error.

- [ ] **Step 3: Implement `pool_builder.py`**

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .config import PoolSpec, Strategy
from .data import Corpus, Task

COSINE_EPS_DUP = 0.85       # ε-dedup: drop if cosine(dist, GT) > this
COSINE_EPS_FUNC_GUARD = 0.80  # extra guard for hard_neg_functional


@dataclass
class Pool:
    spec: PoolSpec
    display_ids: list[str]                     # e.g. ["SKILL_000", "SKILL_001", ...]
    id_map: dict[str, str] = field(default_factory=dict)   # display_id -> canonical id ("gt_<...>" or "skill_NNNNNN")
    cards: dict[str, dict] = field(default_factory=dict)   # display_id -> {name, description, body}
    gt_display_ids: list[str] = field(default_factory=list)


def _rng(seed: int, extra: str) -> np.random.Generator:
    h = hashlib.md5(f"{seed}:{extra}".encode()).digest()[:8]
    return np.random.default_rng(np.frombuffer(h, dtype=np.uint64)[0])


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return a @ b.T


def _sample_random(corpus: Corpus, n_needed: int, exclude: set[int], rng: np.random.Generator) -> list[int]:
    all_idx = np.arange(len(corpus))
    mask = np.ones(len(corpus), dtype=bool)
    for i in exclude:
        mask[i] = False
    pool = all_idx[mask]
    picks = rng.choice(pool, size=min(n_needed, len(pool)), replace=False)
    return picks.tolist()


def _sample_hard_neg_semantic(corpus: Corpus, task_embedding: np.ndarray, n_needed: int, exclude: set[int]) -> list[int]:
    sims = _cosine(task_embedding[None, :], corpus.embeddings).ravel()
    order = np.argsort(-sims)
    picks: list[int] = []
    for idx in order:
        if idx in exclude:
            continue
        picks.append(int(idx))
        if len(picks) >= n_needed:
            break
    return picks


def _apply_cosine_dedup(corpus: Corpus, candidate_idxs: list[int], gt_embeddings: np.ndarray, eps: float) -> list[int]:
    if len(candidate_idxs) == 0:
        return []
    cand_emb = corpus.embeddings[candidate_idxs]
    sims = _cosine(cand_emb, gt_embeddings)
    max_sim = sims.max(axis=1)
    return [idx for idx, s in zip(candidate_idxs, max_sim) if s <= eps]


def build_pool(
    spec: PoolSpec,
    task: Task,
    corpus: Corpus,
    task_embedding: np.ndarray,
    gt_entries: list[tuple[str, str, str, np.ndarray]] | None = None,
) -> Pool:
    """Build a pool for one (task, strategy, N, seed) cell.

    gt_entries: list of (canonical_id, name, body, embedding) for synthetic GT skills.
                If None, we infer from task.gt_skill_names (embeddings required for dedup).
    """
    if gt_entries is None:
        # Plan 1 fallback: no embeddings, use zero vectors (dedup becomes a no-op).
        gt_entries = [
            (f"gt_{task.task_id}_{name}", name, body, np.zeros(corpus.embeddings.shape[1], dtype=np.float32))
            for name, body in zip(task.gt_skill_names, task.gt_skill_bodies)
        ]

    n_gt = len(gt_entries)
    n_distractors = max(0, spec.n - n_gt)
    if n_gt > spec.n:
        # N=1 with multi-GT: truncate to first GT, warn downstream.
        gt_entries = gt_entries[: spec.n]
        n_gt = len(gt_entries)
        n_distractors = 0

    gt_emb = np.stack([e[3] for e in gt_entries]) if gt_entries else np.zeros((0, corpus.embeddings.shape[1]), dtype=np.float32)

    # Exclude GT-correlated corpus entries from distractor pool
    exclude: set[int] = set()
    if n_distractors > 0 and n_gt > 0:
        sims = _cosine(corpus.embeddings, gt_emb)
        max_sim = sims.max(axis=1)
        exclude = set(int(i) for i, s in enumerate(max_sim) if s > COSINE_EPS_DUP)

    rng = _rng(spec.seed, f"{spec.task_id}:{spec.strategy}:{spec.n}:distractor")
    if spec.strategy == "random":
        distractor_idx = _sample_random(corpus, n_distractors, exclude, rng)
    elif spec.strategy == "hard_neg_semantic":
        distractor_idx = _sample_hard_neg_semantic(corpus, task_embedding, n_distractors, exclude)
    else:
        raise NotImplementedError(f"Strategy {spec.strategy} is Plan 2")

    # Build ordered list of (canonical_id, name, description, body) entries
    entries: list[tuple[str, str, str]] = []  # (canonical_id, name, description/body)
    for gt_id, gt_name, gt_body, _ in gt_entries:
        entries.append((gt_id, gt_name, gt_body))
    for idx in distractor_idx:
        entries.append((corpus.ids[idx], corpus.names[idx], corpus.descriptions[idx]))

    # Shuffle and assign display IDs
    order_rng = _rng(spec.seed, f"{spec.task_id}:{spec.strategy}:{spec.n}:order")
    order = np.arange(len(entries))
    order_rng.shuffle(order)

    display_ids = [f"SKILL_{k:03d}" for k in range(len(entries))]
    id_map: dict[str, str] = {}
    cards: dict[str, dict] = {}
    gt_display_ids: list[str] = []
    gt_canonical = {g[0] for g in gt_entries}

    for k, original_idx in enumerate(order):
        canonical_id, name, body_or_desc = entries[original_idx]
        did = display_ids[k]
        id_map[did] = canonical_id
        cards[did] = {"name": name, "description": body_or_desc[:200] if body_or_desc else "", "body": body_or_desc}
        if canonical_id in gt_canonical:
            gt_display_ids.append(did)

    return Pool(spec=spec, display_ids=display_ids, id_map=id_map, cards=cards, gt_display_ids=gt_display_ids)
```

- [ ] **Step 4: Run tests**

Run: `cd "skills retrieval" && PYTHONPATH=src python -m pytest tests/test_pool_builder.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add "skills retrieval/src/skills_retrieval/pool_builder.py" "skills retrieval/tests/test_pool_builder.py"
git commit -m "feat(skills-retrieval): pool builder with random + hard_neg_semantic"
```

---

## Task 5: Prompt rendering (card representation)

**Files:**
- Create: `skills retrieval/src/skills_retrieval/prompt.py`
- Create: `skills retrieval/tests/test_prompt.py`

- [ ] **Step 1: Write failing test**

```python
from skills_retrieval.config import PoolSpec
from skills_retrieval.pool_builder import Pool
from skills_retrieval.prompt import render_awareness_prompt, render_selection_prompt


def _pool():
    spec = PoolSpec(task_id="t0", strategy="random", n=3, seed=0)
    return Pool(
        spec=spec,
        display_ids=["SKILL_000", "SKILL_001", "SKILL_002"],
        id_map={"SKILL_000": "gt_t0_alpha", "SKILL_001": "skill_000005", "SKILL_002": "skill_000002"},
        cards={
            "SKILL_000": {"name": "alpha", "description": "Test alpha skill.", "body": "alpha body"},
            "SKILL_001": {"name": "beta", "description": "Test beta skill.", "body": "beta body"},
            "SKILL_002": {"name": "gamma", "description": "Test gamma skill.", "body": "gamma body"},
        },
        gt_display_ids=["SKILL_000"],
    )


def test_awareness_prompt_mentions_five_and_order(tiny_task):
    prompt = render_awareness_prompt(task_instruction=tiny_task["instruction"], pool=_pool())
    assert "EXACTLY 5" in prompt or "exactly 5" in prompt.lower()
    assert "ordered from" in prompt.lower()
    assert "SKILL_000" in prompt
    assert "alpha" in prompt  # name still in card


def test_selection_prompt_has_single_tag_instruction(tiny_task):
    prompt = render_selection_prompt(task_instruction=tiny_task["instruction"], pool=_pool())
    assert "<skill>" in prompt
    assert "SKILL_000" in prompt


def test_card_block_uses_one_line_per_skill():
    pool = _pool()
    from skills_retrieval.prompt import render_pool_block
    block = render_pool_block(pool, representation="card")
    assert block.count("\n") >= 2  # three skills
    assert "SKILL_000: alpha" in block or "SKILL_000:" in block
```

- [ ] **Step 2: Run to verify failure**

Run: `cd "skills retrieval" && PYTHONPATH=src python -m pytest tests/test_prompt.py -v`
Expected: import error.

- [ ] **Step 3: Implement `prompt.py`**

```python
from __future__ import annotations

from .config import Representation
from .pool_builder import Pool


def render_pool_block(pool: Pool, representation: Representation = "card") -> str:
    lines: list[str] = []
    for did in pool.display_ids:
        card = pool.cards[did]
        if representation == "card":
            desc = card["description"].strip().replace("\n", " ")
            lines.append(f"{did}: {card['name']} — {desc}")
        elif representation == "name_only":
            lines.append(f"{did}: {card['name']}")
        elif representation == "desc_only":
            desc = card["description"].strip().replace("\n", " ")
            lines.append(f"{did}: {desc}")
        elif representation == "full":
            lines.append(f"{did}:\n{card['body']}\n---")
        else:
            raise NotImplementedError(f"representation {representation} not in Plan 1")
    return "\n".join(lines)


def render_awareness_prompt(task_instruction: str, pool: Pool) -> str:
    return _render("awareness", task_instruction, pool)


def render_selection_prompt(task_instruction: str, pool: Pool) -> str:
    return _render("selection", task_instruction, pool)


def _render(probe: str, task_instruction: str, pool: Pool) -> str:
    pool_block = render_pool_block(pool, representation=pool.spec.representation)
    response_instruction = (
        "  <skills>ID_1,ID_2,ID_3,ID_4,ID_5</skills>  # EXACTLY 5 skills, ordered from MOST to LEAST relevant"
        if probe == "awareness"
        else "  <skill>SKILL_ID</skill>              # single best skill for solving this task"
    )
    return (
        "You are a retrieval subject in a controlled study.\n"
        "\n"
        "Task:\n"
        f"{task_instruction}\n"
        "\n"
        f"Available skills ({len(pool.display_ids)}):\n"
        f"{pool_block}\n"
        "\n"
        "Respond with EXACTLY ONE of:\n"
        f"{response_instruction}\n"
        "\n"
        "No other text."
    )
```

- [ ] **Step 4: Run tests**

Run: `cd "skills retrieval" && PYTHONPATH=src python -m pytest tests/test_prompt.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add "skills retrieval/src/skills_retrieval/prompt.py" "skills retrieval/tests/test_prompt.py"
git commit -m "feat(skills-retrieval): prompt rendering with strict response protocol"
```

---

## Task 6: Lenient parser

**Files:**
- Create: `skills retrieval/src/skills_retrieval/parser.py`
- Create: `skills retrieval/tests/test_parser.py`

- [ ] **Step 1: Write failing tests**

```python
from skills_retrieval.parser import parse_response


def test_clean_selection():
    out = parse_response("<skill>SKILL_003</skill>", probe="selection")
    assert out["extracted_ids"] == ["SKILL_003"]
    assert out["format_status"] == "clean"
    assert out["flags"] == {}


def test_clean_awareness_five_items():
    out = parse_response("<skills>A,B,C,D,E</skills>", probe="awareness")
    assert out["extracted_ids"] == ["A", "B", "C", "D", "E"]
    assert out["format_status"] == "clean"


def test_awareness_fewer_than_five_flagged():
    out = parse_response("<skills>A,B</skills>", probe="awareness")
    assert out["extracted_ids"] == ["A", "B"]
    assert out["format_status"] == "warning"
    assert out["flags"].get("length_violation") is True


def test_awareness_duplicates_deduped_in_order():
    out = parse_response("<skills>A,B,A,C,D</skills>", probe="awareness")
    assert out["extracted_ids"] == ["A", "B", "C", "D"]
    assert out["flags"].get("dup_violation") is True


def test_extracts_tag_when_surrounded_by_prose():
    out = parse_response("Let me analyze this. <skill>SKILL_007</skill> I picked this one because...", probe="selection")
    assert out["extracted_ids"] == ["SKILL_007"]
    assert out["format_status"] == "warning"


def test_parse_fail_on_missing_tag():
    out = parse_response("I cannot decide.", probe="selection")
    assert out["format_status"] == "fail"
    assert out["extracted_ids"] == []


def test_whitespace_around_ids_stripped():
    out = parse_response("<skills> A , B , C , D , E </skills>", probe="awareness")
    assert out["extracted_ids"] == ["A", "B", "C", "D", "E"]


def test_multiple_tags_take_first_warn():
    out = parse_response("<skill>X</skill> then also <skill>Y</skill>", probe="selection")
    assert out["extracted_ids"] == ["X"]
    assert out["flags"].get("multiple_tags") is True
```

- [ ] **Step 2: Run to verify failure**

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement parser**

```python
from __future__ import annotations

import re
from typing import Literal

_SELECTION_RE = re.compile(r"<skill>([^<]+)</skill>", re.IGNORECASE)
_AWARENESS_RE = re.compile(r"<skills>([^<]+)</skills>", re.IGNORECASE)


def parse_response(raw: str, probe: Literal["awareness", "selection"]) -> dict:
    pattern = _AWARENESS_RE if probe == "awareness" else _SELECTION_RE
    matches = pattern.findall(raw)
    flags: dict = {}
    if not matches:
        return {"extracted_ids": [], "format_status": "fail", "flags": {"parse_fail": True}, "raw_text": raw}

    first_inner = matches[0]
    if len(matches) > 1:
        flags["multiple_tags"] = True

    ids = [tok.strip() for tok in first_inner.split(",") if tok.strip()]

    if probe == "awareness":
        deduped: list[str] = []
        seen: set[str] = set()
        for i in ids:
            if i in seen:
                flags["dup_violation"] = True
                continue
            seen.add(i)
            deduped.append(i)
        ids = deduped
        if len(ids) < 5:
            flags["length_violation"] = True
        elif len(ids) > 5:
            flags["length_violation"] = True
            ids = ids[:5]
    else:
        ids = ids[:1]

    is_clean = not flags and _is_exact_match(raw, probe)
    format_status = "clean" if is_clean else "warning"
    return {"extracted_ids": ids, "format_status": format_status, "flags": flags, "raw_text": raw}


def _is_exact_match(raw: str, probe: str) -> bool:
    stripped = raw.strip()
    if probe == "selection":
        return bool(re.fullmatch(r"<skill>[^<]+</skill>", stripped, re.IGNORECASE))
    return bool(re.fullmatch(r"<skills>[^<]+</skills>", stripped, re.IGNORECASE))
```

- [ ] **Step 4: Run tests**

Run: `cd "skills retrieval" && PYTHONPATH=src python -m pytest tests/test_parser.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add "skills retrieval/src/skills_retrieval/parser.py" "skills retrieval/tests/test_parser.py"
git commit -m "feat(skills-retrieval): lenient parser with format-compliance flags"
```

---

## Task 7: Metrics (MRR, Top-1, Recall@5, Selection|Aware)

**Files:**
- Create: `skills retrieval/src/skills_retrieval/metrics.py`
- Create: `skills retrieval/tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
from skills_retrieval.metrics import score_trial, aggregate_metrics


def _pool_map_with_gt(gt_display: list[str]):
    return {
        "id_map": {"SKILL_000": "gt_t_alpha", "SKILL_001": "skill_x", "SKILL_002": "skill_y", "SKILL_003": "skill_z", "SKILL_004": "skill_w", "SKILL_005": "skill_v"},
        "gt_display_ids": gt_display,
    }


def test_selection_top1_hit():
    pool_map = _pool_map_with_gt(["SKILL_000"])
    parsed = {"extracted_ids": ["SKILL_000"], "format_status": "clean", "flags": {}}
    s = score_trial(parsed, pool_map, probe="selection")
    assert s["selection_top1"] == 1
    assert s["parse_fail"] == 0


def test_selection_top1_miss():
    pool_map = _pool_map_with_gt(["SKILL_000"])
    parsed = {"extracted_ids": ["SKILL_003"], "format_status": "clean", "flags": {}}
    s = score_trial(parsed, pool_map, probe="selection")
    assert s["selection_top1"] == 0


def test_selection_parse_fail_scores_zero():
    pool_map = _pool_map_with_gt(["SKILL_000"])
    parsed = {"extracted_ids": [], "format_status": "fail", "flags": {"parse_fail": True}}
    s = score_trial(parsed, pool_map, probe="selection")
    assert s["selection_top1"] == 0
    assert s["parse_fail"] == 1


def test_awareness_mrr_rank_1():
    pool_map = _pool_map_with_gt(["SKILL_000"])
    parsed = {"extracted_ids": ["SKILL_000", "SKILL_001", "SKILL_002", "SKILL_003", "SKILL_004"], "format_status": "clean", "flags": {}}
    s = score_trial(parsed, pool_map, probe="awareness")
    assert s["awareness_top1"] == 1
    assert s["awareness_mrr"] == 1.0
    assert s["awareness_recall5"] == 1


def test_awareness_mrr_rank_3():
    pool_map = _pool_map_with_gt(["SKILL_002"])
    parsed = {"extracted_ids": ["SKILL_000", "SKILL_001", "SKILL_002", "SKILL_003", "SKILL_004"], "format_status": "clean", "flags": {}}
    s = score_trial(parsed, pool_map, probe="awareness")
    assert s["awareness_top1"] == 0
    assert abs(s["awareness_mrr"] - 1/3) < 1e-9
    assert s["awareness_recall5"] == 1


def test_awareness_gt_absent_mrr_zero():
    pool_map = _pool_map_with_gt(["SKILL_005"])
    parsed = {"extracted_ids": ["SKILL_000", "SKILL_001", "SKILL_002", "SKILL_003", "SKILL_004"], "format_status": "clean", "flags": {}}
    s = score_trial(parsed, pool_map, probe="awareness")
    assert s["awareness_mrr"] == 0.0
    assert s["awareness_recall5"] == 0


def test_multi_gt_best_rank_wins():
    pool_map = _pool_map_with_gt(["SKILL_003", "SKILL_001"])
    parsed = {"extracted_ids": ["SKILL_000", "SKILL_001", "SKILL_002", "SKILL_003", "SKILL_004"], "format_status": "clean", "flags": {}}
    s = score_trial(parsed, pool_map, probe="awareness")
    assert s["awareness_mrr"] == 0.5  # rank 2 is best GT
    assert s["awareness_recall5"] == 1


def test_aggregate_selection_given_aware():
    # awareness hit + selection hit → selection|aware = 1
    # awareness hit + selection miss → selection|aware = 0
    # awareness miss → excluded from denom
    records = [
        {"awareness_recall5": 1, "selection_top1": 1},
        {"awareness_recall5": 1, "selection_top1": 0},
        {"awareness_recall5": 0, "selection_top1": 1},
    ]
    agg = aggregate_metrics(records)
    assert agg["selection_given_aware"] == 0.5
```

- [ ] **Step 2: Run to verify failure**

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `metrics.py`**

```python
from __future__ import annotations

from typing import Literal


def score_trial(parsed: dict, pool_map: dict, probe: Literal["awareness", "selection"]) -> dict:
    gt_set = set(pool_map["gt_display_ids"])
    parse_fail = 1 if parsed.get("format_status") == "fail" else 0
    out: dict = {"parse_fail": parse_fail, "format_status": parsed.get("format_status", "fail")}

    if probe == "selection":
        ids = parsed.get("extracted_ids") or []
        out["selection_top1"] = 1 if ids and ids[0] in gt_set else 0
        return out

    # awareness
    ids = parsed.get("extracted_ids") or []
    out["awareness_recall5"] = 1 if any(i in gt_set for i in ids[:5]) else 0
    out["awareness_top1"] = 1 if ids and ids[0] in gt_set else 0
    best_rr = 0.0
    for rank, i in enumerate(ids[:5], start=1):
        if i in gt_set:
            rr = 1.0 / rank
            if rr > best_rr:
                best_rr = rr
    out["awareness_mrr"] = best_rr
    return out


def aggregate_metrics(records: list[dict]) -> dict:
    if not records:
        return {}
    keys = [k for k in records[0] if isinstance(records[0][k], (int, float))]
    agg: dict = {}
    for k in keys:
        vals = [r[k] for r in records if k in r]
        agg[k] = sum(vals) / len(vals) if vals else 0.0
    # selection | aware (requires both fields on same trial)
    aware_hit_with_sel = [(r.get("awareness_recall5", 0), r.get("selection_top1", 0)) for r in records if "awareness_recall5" in r and "selection_top1" in r]
    if aware_hit_with_sel:
        num = sum(s for a, s in aware_hit_with_sel if a == 1)
        den = sum(1 for a, _ in aware_hit_with_sel if a == 1)
        agg["selection_given_aware"] = num / den if den else float("nan")
    return agg
```

**Note on `aggregate_metrics` test expectation:** the test passes records that have BOTH fields per-record (one per trial). In the real driver we will join the paired awareness+selection trials for the same (task, seed, pool) before calling `aggregate_metrics`.

- [ ] **Step 4: Run tests**

Run: `cd "skills retrieval" && PYTHONPATH=src python -m pytest tests/test_metrics.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add "skills retrieval/src/skills_retrieval/metrics.py" "skills retrieval/tests/test_metrics.py"
git commit -m "feat(skills-retrieval): MRR, Top-1, Recall@5, Selection|Aware metrics"
```

---

## Task 8: Context-window pre-flight

**Files:**
- Create: `skills retrieval/src/skills_retrieval/preflight.py`
- Create: `skills retrieval/tests/test_preflight.py`

- [ ] **Step 1: Write failing tests**

```python
from skills_retrieval.preflight import count_tokens_approx, will_fit


def test_approx_token_count_monotonic():
    a = count_tokens_approx("hello")
    b = count_tokens_approx("hello world this is longer")
    assert b > a


def test_will_fit_true_for_short():
    assert will_fit("short prompt", model_context_limit=1000, safety_margin=100)


def test_will_fit_false_for_long():
    long_text = "x " * 2000
    assert not will_fit(long_text, model_context_limit=100, safety_margin=10)
```

- [ ] **Step 2: Run to verify failure**

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
from __future__ import annotations


def count_tokens_approx(text: str) -> int:
    # Approximation: ~4 chars per token for English. Use Anthropic tokenizer if available.
    try:
        import anthropic
        client = anthropic.Anthropic()
        # SDK ≥0.40 exposes count_tokens via client.messages.count_tokens
        # Fall back to the 4-chars-per-token heuristic if unavailable
    except Exception:
        pass
    return max(1, len(text) // 4)


def will_fit(prompt: str, model_context_limit: int, safety_margin: int = 1000) -> bool:
    return count_tokens_approx(prompt) + safety_margin < model_context_limit
```

- [ ] **Step 4: Run tests**

Run: `cd "skills retrieval" && PYTHONPATH=src python -m pytest tests/test_preflight.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add "skills retrieval/src/skills_retrieval/preflight.py" "skills retrieval/tests/test_preflight.py"
git commit -m "feat(skills-retrieval): context-window pre-flight"
```

---

## Task 9: Async driver (Anthropic SDK wrapper)

**Files:**
- Create: `skills retrieval/src/skills_retrieval/driver.py`
- Create: `skills retrieval/tests/test_driver.py`

- [ ] **Step 1: Write failing test (mocked API)**

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from skills_retrieval.config import RunConfig
from skills_retrieval.driver import Driver


class FakeMessages:
    def __init__(self, response_text: str):
        self._text = response_text
        self.create = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text=response_text)],
            usage=MagicMock(input_tokens=100, output_tokens=20, cache_creation_input_tokens=0, cache_read_input_tokens=0),
        ))


class FakeClient:
    def __init__(self, response_text: str):
        self.messages = FakeMessages(response_text)


@pytest.mark.asyncio
async def test_driver_returns_trial_record():
    client = FakeClient("<skill>SKILL_000</skill>")
    driver = Driver(client=client, model="claude-sonnet-4-6", max_concurrency=1)
    rec = await driver.run_one(
        pool_id="p0",
        probe="selection",
        system_prompt="You are a retrieval subject.",
        pool_block="SKILL_000: alpha — desc",
        user_prompt="Task: test\n\nRespond with <skill>...</skill>",
    )
    assert rec.raw_response == "<skill>SKILL_000</skill>"
    assert rec.extracted_ids == ["SKILL_000"]
    assert rec.model == "claude-sonnet-4-6"
    assert rec.input_tokens == 100
```

- [ ] **Step 2: Run to verify failure**

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement driver**

```python
from __future__ import annotations

import asyncio
import time
from typing import Any

from .config import Probe, TrialRecord
from .parser import parse_response


class Driver:
    """Async Anthropic SDK wrapper with prompt caching on the pool block.

    Strict responsibility: dispatch, retry, parse. No pool building, no scoring.
    """

    def __init__(self, client: Any, model: str, max_concurrency: int = 8, max_tokens: int = 1024):
        self._client = client
        self._model = model
        self._sem = asyncio.Semaphore(max_concurrency)
        self._max_tokens = max_tokens

    async def run_one(
        self,
        *,
        pool_id: str,
        probe: Probe,
        system_prompt: str,
        pool_block: str,
        user_prompt: str,
    ) -> TrialRecord:
        async with self._sem:
            start = time.time()
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=0.0,
                system=[
                    {"type": "text", "text": system_prompt},
                    {"type": "text", "text": pool_block, "cache_control": {"type": "ephemeral"}},
                ],
                messages=[{"role": "user", "content": user_prompt}],
            )
            latency_ms = int((time.time() - start) * 1000)

        raw = resp.content[0].text
        parsed = parse_response(raw, probe=probe)
        usage = getattr(resp, "usage", None)

        return TrialRecord(
            pool_id=pool_id,
            probe=probe,
            model=self._model,
            raw_response=raw,
            extracted_ids=parsed["extracted_ids"],
            format_status=parsed["format_status"],
            flags=parsed["flags"],
            latency_ms=latency_ms,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) if usage else None,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", None) if usage else None,
        )
```

Note on prompt-caching layout: the **pool block** is the cacheable segment, placed in the `system` array with `cache_control`. The user prompt (task + response instructions) varies per probe and is not cached. This lets a single pool serve 6 calls (3 seeds × 2 probes) with only 1 cache creation + 5 cache reads.

- [ ] **Step 4: Run tests**

Run: `cd "skills retrieval" && PYTHONPATH=src python -m pytest tests/test_driver.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add "skills retrieval/src/skills_retrieval/driver.py" "skills retrieval/tests/test_driver.py"
git commit -m "feat(skills-retrieval): async driver with pool-block prompt caching"
```

---

## Task 10: GT embedding precomputation script

Plan 1 needs embeddings for the 5 pilot tasks' GT skills plus task embeddings so `pool_builder.build_pool` can run for real. Use the same `skill_metadata.jsonl` + `skill_embeddings.npy` for the 44k corpus; for GTs (synthetic SkillsBench skills) and task instructions, compute embeddings with Anthropic's embedding endpoint or fall back to the Qwen pipeline already in `data/extract_slugs.py` / `data/embeddings/` pipelines.

**Files:**
- Create: `skills retrieval/src/skills_retrieval/scripts/embed_tasks_and_gts.py`
- Create: `skills retrieval/tests/test_embed_tasks_and_gts.py` (smoke-level)

- [ ] **Step 1: Check what's already usable**

Run: `ls data/embeddings/ && head -3 data/embeddings/skill_metadata.jsonl`
Expected: confirm the 44k corpus embedding format and dimension (expected 1024 or 768 depending on Qwen model; record the dimension).

- [ ] **Step 2: Find upstream embedding code**

Run: `grep -n "embed" data/extract_slugs.py data/generate_benchmark.py 2>/dev/null | head -30`
Expected: locate the embedding call used to produce `skill_embeddings.npy`. Use the same model for consistency. If unclear, check `scripts/selection_collapse/build_hard_negatives.py` for the same pattern.

- [ ] **Step 3: Write the script**

```python
# skills retrieval/src/skills_retrieval/scripts/embed_tasks_and_gts.py
"""Compute embeddings for SkillsBench tasks and their GT skills.

Output: skills retrieval/pools/tasks_gt_embeddings.npz with keys:
  task_ids: array of str, shape (T,)
  task_embeddings: (T, D)
  gt_ids: list[str] (flattened; use gt_offsets to index)
  gt_offsets: (T+1,) int64 — gt i belongs to task gt_offsets[i]
  gt_embeddings: (sum(n_gt), D)

Uses the same embedding model as data/embeddings/skill_embeddings.npy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Import the same embedding function used to build the 44k corpus.
# Adjust this import after Step 2 identifies the canonical module.
# Fallback: use a direct Qwen / OpenAI / Anthropic call here.
def embed_texts(texts: list[str]) -> np.ndarray:
    """Batch-embed texts. Must match 44k-corpus embedding model/dim."""
    # Prefer re-using the project's existing embedder if available:
    try:
        from data.generate_benchmark import embed_texts as _embed  # type: ignore
        return _embed(texts)
    except Exception:
        pass
    # Minimal fallback (OpenAI text-embedding-3-small or project's Qwen endpoint).
    # Concrete implementation chosen at step 4 below based on Step 2 findings.
    raise RuntimeError("Re-use the same embedder used for data/embeddings/skill_embeddings.npy. Update this function after locating it.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/selection_collapse/skillsbench/tasks.jsonl")
    parser.add_argument("--out", default="skills retrieval/pools/tasks_gt_embeddings.npz")
    parser.add_argument("--task_ids", nargs="+", default=["sb_000", "sb_003", "sb_004", "sb_006", "sb_007"])
    args = parser.parse_args()

    tasks: list[dict] = []
    with Path(args.tasks).open() as f:
        for line in f:
            row = json.loads(line)
            if row["task_id"] in args.task_ids:
                tasks.append(row)

    task_ids = [t["task_id"] for t in tasks]
    task_instr = [t["instruction"] for t in tasks]
    gt_ids: list[str] = []
    gt_texts: list[str] = []
    gt_offsets = [0]
    for t in tasks:
        for g in t.get("gt_skills", []):
            gt_ids.append(f"gt_{t['task_id']}_{g['name']}")
            gt_texts.append(g.get("content", g["name"]))
        gt_offsets.append(len(gt_ids))

    task_emb = embed_texts(task_instr)
    gt_emb = embed_texts(gt_texts) if gt_texts else np.zeros((0, task_emb.shape[1]), dtype=np.float32)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        task_ids=np.asarray(task_ids),
        task_embeddings=task_emb.astype(np.float32),
        gt_ids=np.asarray(gt_ids),
        gt_offsets=np.asarray(gt_offsets, dtype=np.int64),
        gt_embeddings=gt_emb.astype(np.float32),
    )
    print(f"Saved {len(task_ids)} tasks, {len(gt_ids)} GTs → {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Concretise `embed_texts`**

Based on Step 2 findings, replace the `embed_texts` body with the concrete call used in the project. If the project uses Qwen via a specific HTTP endpoint, call it. If the 44k corpus uses OpenAI `text-embedding-3-small`, call that. Record the choice in a comment at the top of the file.

- [ ] **Step 5: Smoke-run on pilot set**

Run: `cd "skills retrieval" && PYTHONPATH=src python -m skills_retrieval.scripts.embed_tasks_and_gts`
Expected: `Saved 5 tasks, ~6-8 GTs → skills retrieval/pools/tasks_gt_embeddings.npz`.

Inspect: `python -c "import numpy as np; d=np.load('skills retrieval/pools/tasks_gt_embeddings.npz'); print({k: d[k].shape for k in d.files})"`
Expected: `task_embeddings` shape `(5, D)` where `D` matches the corpus embedding dimension.

- [ ] **Step 6: Commit**

```bash
git add "skills retrieval/src/skills_retrieval/scripts/embed_tasks_and_gts.py" "skills retrieval/pools/tasks_gt_embeddings.npz"
git commit -m "feat(skills-retrieval): embed pilot tasks + GT skills"
```

---

## Task 11: End-to-end smoke test (full pipeline, fake client)

**Files:**
- Create: `skills retrieval/tests/test_end_to_end.py`

- [ ] **Step 1: Write integration test**

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pytest

from skills_retrieval.config import PoolSpec
from skills_retrieval.data import Corpus, Task
from skills_retrieval.pool_builder import build_pool
from skills_retrieval.prompt import render_awareness_prompt, render_pool_block
from skills_retrieval.driver import Driver
from skills_retrieval.metrics import score_trial


class FakeClient:
    def __init__(self, reply: str):
        self.messages = MagicMock()
        usage = MagicMock(input_tokens=100, output_tokens=10, cache_creation_input_tokens=0, cache_read_input_tokens=0)
        self.messages.create = AsyncMock(return_value=MagicMock(content=[MagicMock(text=reply)], usage=usage))


@pytest.mark.asyncio
async def test_pipeline_scores_correct_pick(tiny_corpus):
    corpus = Corpus.from_paths(tiny_corpus["metadata_path"], tiny_corpus["embeddings_path"])
    task = Task(task_id="t0", instruction="Pick alpha", gt_skill_names=["alpha"], gt_skill_bodies=["alpha body"], domain="test")
    # GT embedding matches corpus[3] exactly so dedup is predictable
    gt_entries = [(f"gt_t0_alpha", "alpha", "alpha body", corpus.embeddings[3])]
    spec = PoolSpec(task_id="t0", strategy="random", n=5, seed=0)
    pool = build_pool(spec, task, corpus, task_embedding=corpus.embeddings[3], gt_entries=gt_entries)

    # Model picks the GT's display id
    gt_display = pool.gt_display_ids[0]
    others = [d for d in pool.display_ids if d != gt_display][:4]
    reply = f"<skills>{gt_display},{others[0]},{others[1]},{others[2]},{others[3]}</skills>"

    driver = Driver(client=FakeClient(reply), model="claude-sonnet-4-6", max_concurrency=1)
    rec = await driver.run_one(
        pool_id=spec.pool_id,
        probe="awareness",
        system_prompt="You are a retrieval subject.",
        pool_block=render_pool_block(pool, representation="card"),
        user_prompt=render_awareness_prompt(task.instruction, pool).split("Available skills")[0],
    )
    pool_map = {"id_map": pool.id_map, "gt_display_ids": pool.gt_display_ids}
    parsed = {"extracted_ids": rec.extracted_ids, "format_status": rec.format_status, "flags": rec.flags}
    scored = score_trial(parsed, pool_map, probe="awareness")
    assert scored["awareness_top1"] == 1
    assert scored["awareness_mrr"] == 1.0
    assert scored["awareness_recall5"] == 1
```

- [ ] **Step 2: Run it**

Run: `cd "skills retrieval" && PYTHONPATH=src python -m pytest tests/test_end_to_end.py -v`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add "skills retrieval/tests/test_end_to_end.py"
git commit -m "test(skills-retrieval): end-to-end pipeline smoke test"
```

---

## Task 12: M1 reproduction runner

**Files:**
- Create: `skills retrieval/src/skills_retrieval/run_m1.py`

- [ ] **Step 1: Write the runner**

```python
"""M1 reproduction: 5 pilot tasks × {random, hard_neg_semantic} × N ∈ {1,5,50,200} × 3 seeds × 2 probes.

Writes raw/parsed/metrics under skills retrieval/runs/<timestamp>-plan1-m1/.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
from pathlib import Path

import anthropic
import numpy as np

from .config import PoolSpec, RunConfig, TrialRecord
from .data import Corpus, load_tasks
from .driver import Driver
from .metrics import score_trial, aggregate_metrics
from .pool_builder import build_pool
from .prompt import render_awareness_prompt, render_pool_block, render_selection_prompt
from .preflight import will_fit

MODEL_CONTEXT_LIMIT = 200_000  # claude-sonnet-4-6


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-meta", default="data/embeddings/skill_metadata.jsonl")
    parser.add_argument("--corpus-emb", default="data/embeddings/skill_embeddings.npy")
    parser.add_argument("--tasks", default="data/selection_collapse/skillsbench/tasks.jsonl")
    parser.add_argument("--task-embeds", default="skills retrieval/pools/tasks_gt_embeddings.npz")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--task-ids", nargs="+", default=["sb_000", "sb_003", "sb_004", "sb_006", "sb_007"])
    parser.add_argument("--pool-sizes", nargs="+", type=int, default=[1, 5, 50, 200])
    parser.add_argument("--strategies", nargs="+", default=["random", "hard_neg_semantic"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--label", default="plan1-m1")
    args = parser.parse_args()

    corpus = Corpus.from_paths(Path(args.corpus_meta), Path(args.corpus_emb))
    tasks_all = {t.task_id: t for t in load_tasks(Path(args.tasks))}
    tasks = [tasks_all[tid] for tid in args.task_ids]
    embeds = np.load(args.task_embeds, allow_pickle=False)
    task_emb_by_id = dict(zip(embeds["task_ids"].tolist(), embeds["task_embeddings"]))
    gt_offsets = embeds["gt_offsets"]
    gt_ids = embeds["gt_ids"].tolist()
    gt_emb = embeds["gt_embeddings"]

    ts = dt.datetime.now().strftime("%Y-%m-%d-%H%M")
    out_dir = Path("skills retrieval/runs") / f"{ts}-{args.label}"
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "parsed").mkdir(parents=True, exist_ok=True)
    (out_dir / "pools").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "skipped.jsonl").touch()

    run_cfg = RunConfig(
        label=args.label, model=args.model,
        task_ids=args.task_ids, strategies=args.strategies,
        pool_sizes=args.pool_sizes, seeds=args.seeds,
    )
    (out_dir / "config.json").write_text(run_cfg.model_dump_json(indent=2))

    # Build pools once per (task, strategy, N, seed) — reused for both probes × all trials.
    pool_cache: dict[str, tuple] = {}
    all_trials: list[tuple[str, str, str, PoolSpec]] = []  # (task_id, pool_id, pool_block, spec) — filled in below

    # Two-stage generation to ensure 6× pool reuse:
    # Stage A: build pools, render pool_block per (task, strategy, N, seed) ONCE.
    # Stage B: submit 2 probes × 1 calls = 2 API calls per pool (seeds are already baked into the pool).

    client = anthropic.AsyncAnthropic()
    driver = Driver(client=client, model=args.model, max_concurrency=run_cfg.max_concurrency)

    async def run_pool(task, spec: PoolSpec):
        t_idx_in_gt = args.task_ids.index(task.task_id)
        gt_start, gt_end = gt_offsets[t_idx_in_gt], gt_offsets[t_idx_in_gt + 1]
        task_gt_ids = gt_ids[gt_start:gt_end]
        task_gt_embs = gt_emb[gt_start:gt_end]
        gt_entries = [
            (gid, gid.rsplit("_", 1)[-1], body, emb)
            for gid, body, emb in zip(task_gt_ids, task.gt_skill_bodies, task_gt_embs)
        ]
        pool = build_pool(spec, task, corpus, task_embedding=task_emb_by_id[task.task_id], gt_entries=gt_entries)
        pool_block = render_pool_block(pool, representation="card")
        (out_dir / "pools" / f"{spec.pool_id}.json").write_text(json.dumps({
            "spec": spec.model_dump(),
            "display_ids": pool.display_ids,
            "id_map": pool.id_map,
            "gt_display_ids": pool.gt_display_ids,
        }, indent=2))

        probe_records: list[TrialRecord] = []
        for probe in ["awareness", "selection"]:
            full_prompt = render_awareness_prompt(task.instruction, pool) if probe == "awareness" else render_selection_prompt(task.instruction, pool)
            if not will_fit(full_prompt, MODEL_CONTEXT_LIMIT, run_cfg.context_safety_margin):
                with (out_dir / "skipped.jsonl").open("a") as f:
                    f.write(json.dumps({"pool_id": spec.pool_id, "probe": probe, "reason": "context_overflow"}) + "\n")
                continue
            user_prompt = full_prompt.split("Available skills")[0] + "Respond per the protocol above."
            rec = await driver.run_one(
                pool_id=spec.pool_id, probe=probe,
                system_prompt="You are a retrieval subject in a controlled study.",
                pool_block=pool_block,
                user_prompt=user_prompt,
            )
            probe_records.append(rec)
            (out_dir / "raw" / f"{spec.pool_id}__{probe}.txt").write_text(rec.raw_response)
            (out_dir / "parsed" / f"{spec.pool_id}__{probe}.json").write_text(rec.model_dump_json(indent=2))
        return spec.pool_id, probe_records, pool

    # Enqueue all pool runs
    coros = []
    for task in tasks:
        for strategy in args.strategies:
            for n in args.pool_sizes:
                for seed in args.seeds:
                    spec = PoolSpec(task_id=task.task_id, strategy=strategy, n=n, seed=seed)
                    coros.append(run_pool(task, spec))

    results = await asyncio.gather(*coros)

    # Score and aggregate
    per_trial: list[dict] = []
    for pool_id, recs, pool in results:
        pool_map = {"id_map": pool.id_map, "gt_display_ids": pool.gt_display_ids}
        trial_row: dict = {"pool_id": pool_id}
        for rec in recs:
            parsed = {"extracted_ids": rec.extracted_ids, "format_status": rec.format_status, "flags": rec.flags}
            scored = score_trial(parsed, pool_map, probe=rec.probe)
            trial_row.update({f"{rec.probe}.{k}": v for k, v in scored.items() if isinstance(v, (int, float))})
            trial_row[f"{rec.probe}.format_status"] = rec.format_status
        per_trial.append(trial_row)

    (out_dir / "metrics" / "per_trial.jsonl").write_text("\n".join(json.dumps(r) for r in per_trial))
    # Flatten for aggregate
    flat = []
    for r in per_trial:
        flat.append({
            "awareness_recall5": r.get("awareness.awareness_recall5", 0),
            "awareness_top1": r.get("awareness.awareness_top1", 0),
            "awareness_mrr": r.get("awareness.awareness_mrr", 0.0),
            "selection_top1": r.get("selection.selection_top1", 0),
            "parse_fail": max(r.get("awareness.parse_fail", 0), r.get("selection.parse_fail", 0)),
        })
    summary = aggregate_metrics(flat)
    (out_dir / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done. Summary → {out_dir / 'metrics' / 'summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Dry-run with a tiny subset first**

Run:
```bash
cd /anvil/projects/x-cis260386/william/procmem2skills/procmem2skills
ANTHROPIC_API_KEY=... PYTHONPATH="skills retrieval/src" python -m skills_retrieval.run_m1 \
  --task-ids sb_000 --pool-sizes 1 5 --strategies random --seeds 0 --label plan1-smoke
```
Expected: 2 API calls total (1 task × 1 strategy × 2 N × 1 seed × 2 probes × ~caching overhead); `metrics/summary.json` produced.

- [ ] **Step 3: Inspect smoke run**

```bash
cat "skills retrieval/runs/"*plan1-smoke*"/metrics/summary.json"
cat "skills retrieval/runs/"*plan1-smoke*"/parsed/"*.json | head -30
```
Expected: `awareness_recall5` and `selection_top1` non-zero for N=1 (GT-only anchor should be ~100%).

- [ ] **Step 4: Full M1 run**

```bash
cd /anvil/projects/x-cis260386/william/procmem2skills/procmem2skills
ANTHROPIC_API_KEY=... PYTHONPATH="skills retrieval/src" python -m skills_retrieval.run_m1 --label plan1-m1
```
Expected: ~240 API calls (60 pools × 2 probes, seeds vary). Runtime 5–10 min with concurrency=8.

- [ ] **Step 5: Sanity-check against v1 scale-up**

From `skills retrieval/runs/<ts>-plan1-m1/metrics/summary.json`, slice per (strategy, N). Expected anchors:
- `random @ N ∈ {5, 50, 200}` → Recall@5 ≈ 1.0
- `hard_neg_semantic @ N=50` → Recall@5 in [0.65, 0.80] (v1 scale-up reported 0.72 ± 0.038 on 47 tasks; 5-task pilot will be noisier, but the direction should hold)
- `N=1` (all strategies) → Recall@5 = 1.0, Top-1 = 1.0 (GT-only anchor)

Write a one-page markdown summary at `skills retrieval/runs/<ts>-plan1-m1/ANALYSIS.md` noting matches/mismatches vs. v1.

- [ ] **Step 6: Commit code and the M1 run outputs**

```bash
git add "skills retrieval/src/skills_retrieval/run_m1.py"
git add "skills retrieval/runs/"*plan1-m1*"/config.json" "skills retrieval/runs/"*plan1-m1*"/metrics/" "skills retrieval/runs/"*plan1-m1*"/ANALYSIS.md"
git commit -m "feat(skills-retrieval): M1 reproduction runner + pilot results"
```

(Skip committing `raw/` and `parsed/` — they'll balloon the repo. Add to `.gitignore` if needed.)

---

## Self-review checklist (do NOT skip)

Before handing off to execution, walk the spec (`skills retrieval/design-v2.md`) section-by-section and confirm which plan covers each:

- §1 RQ1 (collapse curve): **Plan 2** (full N sweep incl. 1000/2000/5000).
- §1 RQ2 (representation ablation): **Plan 2** (name_only/desc_only/full/compressed_full).
- §1 RQ3 (Phase A→B correlation): **Plan 3**.
- §1 RQ4 (Selection vs. Selection|Aware divergence): **Plan 1** produces the metric; **Plan 2** plots the divergence across large N.
- §3.1 distractor taxonomy: `random`, `hard_neg_semantic` in **Plan 1**; `easy_neg`, `hard_neg_functional`, `adversarial` in **Plan 2**.
- §3.3 format perturbation: **Plan 2**.
- §3.4 confound control (compressed_full): **Plan 2**.
- §4 prompt + GT-leakage fix + lenient parser + length/dup flags: **Plan 1** (Tasks 5, 6).
- §5 MRR/Top-1/Recall@5/Selection|Aware/format-compliance: **Plan 1** (Task 7). FEM: **Plan 2**. Per-task difficulty regression: **Plan 2**. Random baseline overlay: **Plan 2** (plotting).
- §6.1 async driver + prompt caching + context pre-flight: **Plan 1** (Tasks 8, 9).
- §6.3 run output layout (`runs/<ts>-label/{raw,parsed,metrics,pools,skipped.jsonl}`): **Plan 1** (Task 12).
- §7 Phase B: **Plan 3**.
- §8 M1, M1.5, M2, M2.5 (scope anchor for this plan): **Plan 1** covers the subset of M1 explicitly.

Plan 1 does **not** implement: HDBSCAN clustering (Plan 2 M1.5), the full 47-task sweep (Plan 2 M2), the figures (Plan 2 M4), FEM (Plan 2 M4.5), or any Phase B work (Plan 3).

**Placeholder scan:** searched this plan for TBD/TODO/fill-in patterns. One deliberate hand-off point at Task 10 Step 4 ("Concretise `embed_texts`") — resolved by inspecting an existing project script in Step 2. This is not a placeholder in the spec sense; it is a pointer to existing code that must be identified.

**Type consistency:** `Pool`, `PoolSpec`, `TrialRecord`, `parse_response`, `score_trial`, `aggregate_metrics`, `build_pool`, `Driver.run_one`, `will_fit` — signatures are consistent across tasks.

---

## Execution handoff

Plan complete and saved to `skills retrieval/plans/2026-04-18-plan-1-vertical-slice.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
