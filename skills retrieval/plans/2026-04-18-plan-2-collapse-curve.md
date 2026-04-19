# Plan 2 — Phase A Scale-up: Full Collapse Curve + Figures

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scale Phase A recognition eval from 5 pilot tasks to all 88 SkillsBench tasks, extend pool sizes to include N=1000, compute bootstrap confidence intervals, and generate the collapse-curve and Selection|Aware-divergence figures needed for the paper.

**Architecture:** Reuse all Plan 1 modules (`config`, `data`, `pool_builder`, `prompt`, `parser`, `metrics`, `cli_driver`, `driver`). Add three new modules: `stats.py` (bootstrap CI for proportions), `aggregate.py` (group per-trial records by condition → per-condition summary with CIs), and `plots.py` (matplotlib collapse-curve + divergence plots). Add `run_m2.py` runner that runs the full sweep and invokes aggregation + plotting. Do not change existing module APIs.

**Tech Stack:** Python 3.10+ (existing env at `/anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/python3.10`). New deps: `matplotlib>=3.7`, `pandas>=2.0` (already present for analysis). Bootstrap via `numpy`; no scipy.stats needed.

**Scope anchor:** 88 tasks × {random, hard_neg_semantic} × N ∈ {1, 5, 50, 200, 1000} × card representation × 3 seeds × 2 probes = **5,280 API calls** via `claude -p` (~4–5 hours at concurrency 4, ~3s/call). Keep this in a single `run_m2.py` invocation; resume via `--skip-existing` on re-runs.

**Out of scope (deferred):**
- Representation ablation (name_only, desc_only, full, compressed_full) — Plan 3
- Additional distractor strategies (easy_neg, hard_neg_functional, adversarial) — Plan 4
- Format perturbation, confound control, FEM, difficulty regression — Plan 5
- Phase B end-to-end harbor runs — Plan 6

**Expected output:**
- `skills retrieval/runs/<ts>-plan2-m2/metrics/summary_by_condition.json` — per-(strategy, N) Recall@5/Top-1/MRR/Sel|Aware with 95% bootstrap CIs
- `skills retrieval/runs/<ts>-plan2-m2/figures/collapse_curve.pdf` — x=N (log), y=Recall@5, one line per strategy, shaded CI band
- `skills retrieval/runs/<ts>-plan2-m2/figures/selection_aware_divergence.pdf` — x=N (log), y=Sel|Aware − Awareness_Recall@5
- `skills retrieval/runs/<ts>-plan2-m2/ANALYSIS.md` — written interpretation

---

## File Structure

```
skills retrieval/
├── src/skills_retrieval/
│   ├── stats.py                    # bootstrap CI for rates
│   ├── aggregate.py                # per-trial → per-condition with CIs
│   ├── plots.py                    # matplotlib figures
│   ├── run_m2.py                   # full-sweep runner with --skip-existing
│   └── scripts/
│       └── embed_tasks_and_gts.py  # (existing; Plan 1) — rerun for all 88 tasks
└── tests/
    ├── test_stats.py
    ├── test_aggregate.py
    └── test_plots.py               # lightweight: check file exists + non-empty
```

**Module boundaries:**
- `stats.py` — pure numpy. `bootstrap_rate_ci(values, n_boot=1000, seed=0, ci=0.95) -> (mean, lo, hi)`. No I/O.
- `aggregate.py` — pure transformation. Reads per-trial JSONL, returns nested dict keyed by (task_id, strategy, N, seed) then folded by (strategy, N) with `n` trials and bootstrap CIs. No plotting, no API calls.
- `plots.py` — consumes aggregated summary dict, writes PDFs. No API calls, no aggregation logic.
- `run_m2.py` — orchestration only. Reuses `CLIDriver`, calls `build_pool` / `render_pool_block` / `parse_response` / `score_trial`. Adds `--skip-existing` (skip pools whose `parsed/<pool_id>__<probe>.json` already exists).

---

## Task 1: Extend GT embeddings to all 88 tasks

**Files:**
- Modify: `skills retrieval/src/skills_retrieval/scripts/embed_tasks_and_gts.py` (change default task_ids)
- Create output: `skills retrieval/pools/tasks_gt_embeddings_full.npz`

- [ ] **Step 1: Check current pilot-embedding script default**

Run: `head -30 "skills retrieval/src/skills_retrieval/scripts/embed_tasks_and_gts.py"`
Expected: see `default=["sb_000", "sb_003", "sb_004", "sb_006", "sb_007"]` on the `--task_ids` argparse line.

- [ ] **Step 2: Add a `--all` flag**

Modify the script's argparse block. Replace the `default=[...]` list with a tuple plus a new flag:

```python
p.add_argument("--tasks", default="data/selection_collapse/skillsbench/tasks.jsonl")
p.add_argument("--out", default="skills retrieval/pools/tasks_gt_embeddings.npz")
p.add_argument("--task_ids", nargs="+", default=None,
               help="Task IDs to embed. If omitted, embeds all tasks in --tasks file.")
```

And in `main()`, after loading tasks:
```python
tasks_all: list[dict] = []
with Path(args.tasks).open() as f:
    for line in f:
        tasks_all.append(json.loads(line))
if args.task_ids:
    tasks = [t for t in tasks_all if t["task_id"] in args.task_ids]
    tasks.sort(key=lambda t: args.task_ids.index(t["task_id"]))
else:
    tasks = tasks_all
```

(Remove the old `tasks.sort(...)` call that assumed `args.task_ids` was truthy.)

- [ ] **Step 3: Run to produce full-set embeddings**

```bash
cd "/anvil/projects/x-cis260386/william/procmem2skills/procmem2skills"
PYTHONPATH="skills retrieval/src" /anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/python3.10 \
  -m skills_retrieval.scripts.embed_tasks_and_gts \
  --out "skills retrieval/pools/tasks_gt_embeddings_full.npz"
```

Expected output: `Saved 88 tasks, <total_gts> GTs (dim=1024) → skills retrieval/pools/tasks_gt_embeddings_full.npz` (takes 2–3 min on CPU).

- [ ] **Step 4: Verify shapes**

```bash
/anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/python3.10 -c "
import numpy as np
d = np.load('skills retrieval/pools/tasks_gt_embeddings_full.npz')
print({k: d[k].shape for k in d.files})
"
```

Expected: `task_embeddings (88, 1024)`, `gt_offsets (89,)`, `gt_embeddings (~200, 1024)` (the exact GT count depends on how many GT skills each task has).

- [ ] **Step 5: Commit**

```bash
git add "skills retrieval/src/skills_retrieval/scripts/embed_tasks_and_gts.py" \
        "skills retrieval/pools/tasks_gt_embeddings_full.npz"
git commit -m "$(cat <<'EOF'
feat(skills-retrieval): embed all 88 SkillsBench tasks for Plan 2

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Bootstrap CI utility (`stats.py`)

**Files:**
- Create: `skills retrieval/src/skills_retrieval/stats.py`
- Create: `skills retrieval/tests/test_stats.py`

- [ ] **Step 1: Write failing test**

Write `skills retrieval/tests/test_stats.py`:

```python
import numpy as np

from skills_retrieval.stats import bootstrap_rate_ci


def test_ci_zero_variance():
    mean, lo, hi = bootstrap_rate_ci([1, 1, 1, 1], n_boot=500, seed=0)
    assert mean == 1.0
    assert lo == 1.0
    assert hi == 1.0


def test_ci_mean_matches_sample_mean():
    values = [1, 0, 1, 1, 0, 1, 1, 0, 1, 0]  # sample mean = 0.6
    mean, lo, hi = bootstrap_rate_ci(values, n_boot=2000, seed=0)
    assert abs(mean - 0.6) < 1e-9
    assert 0.25 < lo < 0.6
    assert 0.6 < hi < 0.95


def test_ci_brackets_mean():
    values = [1, 0] * 20
    mean, lo, hi = bootstrap_rate_ci(values, n_boot=1000, seed=42)
    assert lo <= mean <= hi


def test_ci_empty_returns_nans():
    import math
    mean, lo, hi = bootstrap_rate_ci([], n_boot=100, seed=0)
    assert math.isnan(mean)
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_ci_deterministic_with_seed():
    values = [0, 1] * 15
    a = bootstrap_rate_ci(values, n_boot=500, seed=7)
    b = bootstrap_rate_ci(values, n_boot=500, seed=7)
    assert a == b
```

- [ ] **Step 2: Run to verify fail**

```bash
cd "/anvil/projects/x-cis260386/william/procmem2skills/procmem2skills"
/anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/python3.10 -m pytest "skills retrieval/tests/test_stats.py" -v
```

Expected: `ModuleNotFoundError: No module named 'skills_retrieval.stats'`.

- [ ] **Step 3: Implement**

Write `skills retrieval/src/skills_retrieval/stats.py`:

```python
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def bootstrap_rate_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI for the mean of a 0/1 or float sequence.

    Returns (sample_mean, lo, hi) where [lo, hi] is the (1-alpha)/2..1-(1-alpha)/2
    percentile interval over n_boot resamples. NaNs on empty input.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return (math.nan, math.nan, math.nan)
    mean = float(arr.mean())
    if n == 1 or np.allclose(arr, arr[0]):
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    alpha = 1.0 - ci
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return (mean, lo, hi)
```

- [ ] **Step 4: Verify tests pass**

Run the pytest command from Step 2. Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add "skills retrieval/src/skills_retrieval/stats.py" "skills retrieval/tests/test_stats.py"
git commit -m "$(cat <<'EOF'
feat(skills-retrieval): bootstrap CI utility for rate metrics

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Aggregation module (`aggregate.py`)

**Files:**
- Create: `skills retrieval/src/skills_retrieval/aggregate.py`
- Create: `skills retrieval/tests/test_aggregate.py`

- [ ] **Step 1: Write failing test**

Write `skills retrieval/tests/test_aggregate.py`:

```python
import json
from pathlib import Path

from skills_retrieval.aggregate import aggregate_by_condition, load_per_trial


def test_aggregate_by_condition_structure(tmp_path: Path):
    rows = [
        # sb_000 random N=5 seed=0: aware hit, sel hit
        {"pool_id": "sb_000__random__n5__s0__card",
         "awareness.awareness_recall5": 1, "awareness.awareness_top1": 1, "awareness.awareness_mrr": 1.0,
         "selection.selection_top1": 1,
         "awareness.parse_fail": 0, "selection.parse_fail": 0},
        # sb_000 random N=5 seed=1: aware hit, sel miss
        {"pool_id": "sb_000__random__n5__s1__card",
         "awareness.awareness_recall5": 1, "awareness.awareness_top1": 0, "awareness.awareness_mrr": 0.5,
         "selection.selection_top1": 0,
         "awareness.parse_fail": 0, "selection.parse_fail": 0},
        # sb_000 random N=50 seed=0: aware miss, sel miss
        {"pool_id": "sb_000__random__n50__s0__card",
         "awareness.awareness_recall5": 0, "awareness.awareness_top1": 0, "awareness.awareness_mrr": 0.0,
         "selection.selection_top1": 0,
         "awareness.parse_fail": 0, "selection.parse_fail": 0},
    ]
    path = tmp_path / "per_trial.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))

    loaded = load_per_trial(path)
    assert len(loaded) == 3

    agg = aggregate_by_condition(loaded, n_boot=500, seed=0)

    # agg keyed by (strategy, n) → dict of metric → {"mean", "ci_lo", "ci_hi", "n_trials"}
    assert ("random", 5) in agg
    assert ("random", 50) in agg
    r5 = agg[("random", 5)]
    assert r5["awareness_recall5"]["mean"] == 1.0
    assert r5["awareness_recall5"]["n_trials"] == 2
    assert r5["selection_given_aware"]["mean"] == 0.5  # 1 sel hit out of 2 aware hits

    r50 = agg[("random", 50)]
    assert r50["awareness_recall5"]["mean"] == 0.0


def test_aggregate_handles_missing_probe(tmp_path: Path):
    """If a probe was skipped (context overflow), trial row has NaN-ish missing keys."""
    rows = [
        {"pool_id": "sb_000__random__n1000__s0__card",
         "awareness.awareness_recall5": 1, "awareness.awareness_top1": 1, "awareness.awareness_mrr": 1.0,
         "awareness.parse_fail": 0},
        # no selection.* keys — skipped due to context limit
    ]
    path = tmp_path / "per_trial.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    loaded = load_per_trial(path)
    agg = aggregate_by_condition(loaded, n_boot=200, seed=0)
    key = ("random", 1000)
    assert agg[key]["awareness_recall5"]["mean"] == 1.0
    # selection_top1 has 0 trials (no selection probe ran)
    assert agg[key]["selection_top1"]["n_trials"] == 0


def test_pool_id_parse():
    from skills_retrieval.aggregate import parse_pool_id
    got = parse_pool_id("sb_034__hard_neg_semantic__n200__s2__card")
    assert got == {"task_id": "sb_034", "strategy": "hard_neg_semantic", "n": 200, "seed": 2, "representation": "card"}
```

- [ ] **Step 2: Verify fail**

```bash
/anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/python3.10 -m pytest "skills retrieval/tests/test_aggregate.py" -v
```

Expected: `ModuleNotFoundError: No module named 'skills_retrieval.aggregate'`.

- [ ] **Step 3: Implement**

Write `skills retrieval/src/skills_retrieval/aggregate.py`:

```python
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

from .stats import bootstrap_rate_ci


_POOL_ID_RE = re.compile(
    r"^(?P<task_id>[^_]+(?:_[^_]+)*?)__(?P<strategy>[a-z_]+?)__n(?P<n>\d+)__s(?P<seed>\d+)__(?P<representation>\w+)$"
)


def parse_pool_id(pool_id: str) -> dict:
    """Split a pool_id back into its structured components.

    pool_id format: <task_id>__<strategy>__n<N>__s<seed>__<representation>
    e.g. "sb_034__hard_neg_semantic__n200__s2__card"
    """
    m = _POOL_ID_RE.match(pool_id)
    if not m:
        raise ValueError(f"Unparseable pool_id: {pool_id}")
    gd = m.groupdict()
    return {
        "task_id": gd["task_id"],
        "strategy": gd["strategy"],
        "n": int(gd["n"]),
        "seed": int(gd["seed"]),
        "representation": gd["representation"],
    }


def load_per_trial(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


METRICS_FROM_AWARENESS = {
    "awareness_recall5": "awareness.awareness_recall5",
    "awareness_top1": "awareness.awareness_top1",
    "awareness_mrr": "awareness.awareness_mrr",
}
METRICS_FROM_SELECTION = {
    "selection_top1": "selection.selection_top1",
}


def aggregate_by_condition(rows: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """Group rows by (strategy, n), compute per-metric bootstrap CIs.

    Returns dict keyed by (strategy:str, n:int), each value is a dict
    metric_name -> {"mean": float, "ci_lo": float, "ci_hi": float, "n_trials": int}.

    Special metric: selection_given_aware (conditional rate, denominator = awareness_recall5 hits).
    """
    by_cond: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        try:
            p = parse_pool_id(r["pool_id"])
        except ValueError:
            continue
        by_cond[(p["strategy"], p["n"])].append(r)

    agg: dict = {}
    for cond, group in sorted(by_cond.items()):
        entry: dict = {}
        for out_key, col in {**METRICS_FROM_AWARENESS, **METRICS_FROM_SELECTION}.items():
            vals = [r[col] for r in group if col in r]
            mean, lo, hi = bootstrap_rate_ci(vals, n_boot=n_boot, seed=seed)
            entry[out_key] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n_trials": len(vals)}

        # selection_given_aware: only trials that have BOTH probes
        paired = [
            (r["awareness.awareness_recall5"], r["selection.selection_top1"])
            for r in group
            if "awareness.awareness_recall5" in r and "selection.selection_top1" in r
        ]
        aware_hits_sel = [sel for aware, sel in paired if aware == 1]
        mean, lo, hi = bootstrap_rate_ci(aware_hits_sel, n_boot=n_boot, seed=seed)
        entry["selection_given_aware"] = {
            "mean": mean, "ci_lo": lo, "ci_hi": hi, "n_trials": len(aware_hits_sel),
        }

        # parse_fail: union across probes (1 if either probe failed)
        pf_vals = []
        for r in group:
            aware_pf = r.get("awareness.parse_fail", 0)
            sel_pf = r.get("selection.parse_fail", 0)
            pf_vals.append(max(aware_pf, sel_pf))
        mean, lo, hi = bootstrap_rate_ci(pf_vals, n_boot=n_boot, seed=seed)
        entry["parse_fail"] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n_trials": len(pf_vals)}

        agg[cond] = entry

    return agg
```

- [ ] **Step 4: Verify tests pass**

Run the pytest command from Step 2. Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add "skills retrieval/src/skills_retrieval/aggregate.py" "skills retrieval/tests/test_aggregate.py"
git commit -m "$(cat <<'EOF'
feat(skills-retrieval): per-condition aggregation with bootstrap CIs

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Plotting module (`plots.py`)

**Files:**
- Create: `skills retrieval/src/skills_retrieval/plots.py`
- Create: `skills retrieval/tests/test_plots.py`

- [ ] **Step 1: Write failing test**

Write `skills retrieval/tests/test_plots.py`:

```python
from pathlib import Path

from skills_retrieval.plots import plot_collapse_curve, plot_selection_aware_divergence


def _tiny_agg():
    return {
        ("random", 1): {
            "awareness_recall5": {"mean": 1.0, "ci_lo": 1.0, "ci_hi": 1.0, "n_trials": 10},
            "selection_top1": {"mean": 1.0, "ci_lo": 1.0, "ci_hi": 1.0, "n_trials": 10},
            "selection_given_aware": {"mean": 1.0, "ci_lo": 1.0, "ci_hi": 1.0, "n_trials": 10},
        },
        ("random", 5): {
            "awareness_recall5": {"mean": 1.0, "ci_lo": 0.9, "ci_hi": 1.0, "n_trials": 10},
            "selection_top1": {"mean": 1.0, "ci_lo": 0.9, "ci_hi": 1.0, "n_trials": 10},
            "selection_given_aware": {"mean": 1.0, "ci_lo": 0.9, "ci_hi": 1.0, "n_trials": 10},
        },
        ("random", 50): {
            "awareness_recall5": {"mean": 0.95, "ci_lo": 0.85, "ci_hi": 1.0, "n_trials": 10},
            "selection_top1": {"mean": 0.95, "ci_lo": 0.85, "ci_hi": 1.0, "n_trials": 10},
            "selection_given_aware": {"mean": 0.95, "ci_lo": 0.85, "ci_hi": 1.0, "n_trials": 10},
        },
        ("hard_neg_semantic", 1): {
            "awareness_recall5": {"mean": 1.0, "ci_lo": 1.0, "ci_hi": 1.0, "n_trials": 10},
            "selection_top1": {"mean": 0.8, "ci_lo": 0.6, "ci_hi": 1.0, "n_trials": 10},
            "selection_given_aware": {"mean": 0.8, "ci_lo": 0.6, "ci_hi": 1.0, "n_trials": 10},
        },
        ("hard_neg_semantic", 5): {
            "awareness_recall5": {"mean": 0.9, "ci_lo": 0.7, "ci_hi": 1.0, "n_trials": 10},
            "selection_top1": {"mean": 0.7, "ci_lo": 0.5, "ci_hi": 0.9, "n_trials": 10},
            "selection_given_aware": {"mean": 0.78, "ci_lo": 0.55, "ci_hi": 0.95, "n_trials": 9},
        },
        ("hard_neg_semantic", 50): {
            "awareness_recall5": {"mean": 0.7, "ci_lo": 0.55, "ci_hi": 0.85, "n_trials": 10},
            "selection_top1": {"mean": 0.5, "ci_lo": 0.3, "ci_hi": 0.7, "n_trials": 10},
            "selection_given_aware": {"mean": 0.71, "ci_lo": 0.5, "ci_hi": 0.9, "n_trials": 7},
        },
    }


def test_collapse_curve_writes_file(tmp_path: Path):
    out = tmp_path / "collapse.pdf"
    plot_collapse_curve(_tiny_agg(), out_path=out, metric="awareness_recall5")
    assert out.exists()
    assert out.stat().st_size > 500  # non-trivial PDF


def test_divergence_plot_writes_file(tmp_path: Path):
    out = tmp_path / "divergence.pdf"
    plot_selection_aware_divergence(_tiny_agg(), out_path=out)
    assert out.exists()
    assert out.stat().st_size > 500
```

- [ ] **Step 2: Verify fail**

```bash
/anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/python3.10 -m pytest "skills retrieval/tests/test_plots.py" -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Write `skills retrieval/src/skills_retrieval/plots.py`:

```python
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STRATEGY_LABEL = {
    "random": "Random distractors",
    "hard_neg_semantic": "Hard neg (semantic)",
    "easy_neg": "Easy neg (cluster-far)",
    "hard_neg_functional": "Hard neg (functional)",
    "adversarial": "Adversarial",
}
STRATEGY_COLOR = {
    "random": "#4C72B0",
    "hard_neg_semantic": "#C44E52",
    "easy_neg": "#55A868",
    "hard_neg_functional": "#8172B2",
    "adversarial": "#937860",
}


def _strategies_present(agg: dict) -> list[str]:
    seen: dict[str, None] = {}
    for (strategy, _n) in agg.keys():
        seen.setdefault(strategy, None)
    return list(seen.keys())


def _ns_for_strategy(agg: dict, strategy: str) -> list[int]:
    return sorted(n for (s, n) in agg.keys() if s == strategy)


def plot_collapse_curve(agg: dict, out_path: Path, metric: str = "awareness_recall5") -> None:
    """Plot metric vs pool size N on log x-axis, one line per strategy, CI band."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for strategy in _strategies_present(agg):
        ns = _ns_for_strategy(agg, strategy)
        means = [agg[(strategy, n)][metric]["mean"] for n in ns]
        los = [agg[(strategy, n)][metric]["ci_lo"] for n in ns]
        his = [agg[(strategy, n)][metric]["ci_hi"] for n in ns]
        color = STRATEGY_COLOR.get(strategy, None)
        label = STRATEGY_LABEL.get(strategy, strategy)
        ax.plot(ns, means, marker="o", label=label, color=color)
        ax.fill_between(ns, los, his, alpha=0.2, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Pool size N")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(f"Selection collapse: {metric}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_selection_aware_divergence(agg: dict, out_path: Path) -> None:
    """Two subplots: (left) awareness_recall5 and selection_top1 overlaid per strategy;
    (right) sel|aware − awareness_recall5 per strategy (the divergence).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for strategy in _strategies_present(agg):
        ns = _ns_for_strategy(agg, strategy)
        color = STRATEGY_COLOR.get(strategy, None)
        label = STRATEGY_LABEL.get(strategy, strategy)

        aware_means = [agg[(strategy, n)]["awareness_recall5"]["mean"] for n in ns]
        sel_means = [agg[(strategy, n)]["selection_top1"]["mean"] for n in ns]
        sa_means = [agg[(strategy, n)]["selection_given_aware"]["mean"] for n in ns]

        ax1.plot(ns, aware_means, marker="o", linestyle="-", color=color, label=f"{label} (aware)")
        ax1.plot(ns, sel_means, marker="x", linestyle="--", color=color, alpha=0.6, label=f"{label} (sel)")

        diverge = [sa - aw for sa, aw in zip(sa_means, aware_means)]
        ax2.plot(ns, diverge, marker="s", color=color, label=label)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("Pool size N")
        ax.grid(True, which="both", alpha=0.3)
    ax1.set_ylabel("rate")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(frameon=False, fontsize=8, loc="lower left")
    ax1.set_title("Awareness vs Selection")
    ax2.set_ylabel("sel|aware − recall@5")
    ax2.axhline(0, color="black", alpha=0.3, linewidth=0.8)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    ax2.set_title("Selection | Aware divergence")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
```

- [ ] **Step 4: Install matplotlib if missing**

```bash
/anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/python3.10 -c "import matplotlib; print(matplotlib.__version__)" \
  || /anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/pip install "matplotlib>=3.7"
```

- [ ] **Step 5: Verify tests pass**

Run:
```bash
/anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/python3.10 -m pytest "skills retrieval/tests/test_plots.py" -v
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add "skills retrieval/src/skills_retrieval/plots.py" "skills retrieval/tests/test_plots.py"
git commit -m "$(cat <<'EOF'
feat(skills-retrieval): collapse-curve and divergence plotting

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Full-sweep runner (`run_m2.py`) with resume-on-existing

**Files:**
- Create: `skills retrieval/src/skills_retrieval/run_m2.py`

This is the orchestration analogue of `run_m1.py` with these differences:
1. Defaults: all 88 task IDs, N ∈ {1, 5, 50, 200, 1000}, strategies {random, hard_neg_semantic}.
2. Uses the 88-task embedding file (`tasks_gt_embeddings_full.npz`) by default.
3. Adds `--skip-existing`: for each (pool_id, probe) pair, if `parsed/<pool_id>__<probe>.json` already exists in the target output dir, skip the API call. Used to resume a partial run.
4. After the API sweep, loads all `parsed/*.json` back, recomputes per-trial rows into `metrics/per_trial.jsonl`, then calls `aggregate_by_condition` → writes `metrics/summary_by_condition.json`, and finally calls both `plot_collapse_curve` and `plot_selection_aware_divergence` → writes PDFs under `figures/`.

- [ ] **Step 1: Load task IDs from the tasks file (all 88)**

In `run_m2.py`, replace the hard-coded pilot list with:

```python
def _all_task_ids_in(tasks_path: Path) -> list[str]:
    ids: list[str] = []
    with Path(tasks_path).open() as f:
        for line in f:
            ids.append(__import__("json").loads(line)["task_id"])
    return ids
```

Used if `--task-ids` is not passed.

- [ ] **Step 2: Write the module**

Write `skills retrieval/src/skills_retrieval/run_m2.py`:

```python
"""Plan 2: full-scale collapse-curve sweep across all SkillsBench tasks.

88 tasks × {random, hard_neg_semantic} × N ∈ {1, 5, 50, 200, 1000} × 3 seeds × 2 probes
= 5,280 API calls. Use --skip-existing to resume a partial run.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
from pathlib import Path

import numpy as np

from .aggregate import aggregate_by_condition, load_per_trial
from .cli_driver import CLIDriver
from .config import PoolSpec, RunConfig, TrialRecord
from .data import Corpus, load_tasks
from .metrics import score_trial
from .plots import plot_collapse_curve, plot_selection_aware_divergence
from .pool_builder import build_pool
from .prompt import render_awareness_prompt, render_pool_block, render_selection_prompt
from .preflight import will_fit

MODEL_CONTEXT_LIMIT = 200_000


def _all_task_ids_in(tasks_path: Path) -> list[str]:
    ids: list[str] = []
    with Path(tasks_path).open() as f:
        for line in f:
            ids.append(json.loads(line)["task_id"])
    return ids


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-meta", default="data/embeddings/skill_metadata.jsonl")
    p.add_argument("--corpus-emb", default="data/embeddings/skill_embeddings.npy")
    p.add_argument("--corpus-desc", default="data/processed/skill_corpus.jsonl")
    p.add_argument("--tasks", default="data/selection_collapse/skillsbench/tasks.jsonl")
    p.add_argument("--task-embeds", default="skills retrieval/pools/tasks_gt_embeddings_full.npz")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--task-ids", nargs="*", default=None,
                   help="If omitted, runs all tasks in --tasks file.")
    p.add_argument("--pool-sizes", nargs="+", type=int, default=[1, 5, 50, 200, 1000])
    p.add_argument("--strategies", nargs="+", default=["random", "hard_neg_semantic"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--label", default="plan2-m2")
    p.add_argument("--out-dir", default=None, help="Override output dir (for resume).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip (pool_id, probe) pairs whose parsed/ file already exists.")
    args = p.parse_args()

    task_ids = args.task_ids or _all_task_ids_in(Path(args.tasks))

    corpus = Corpus.from_paths(Path(args.corpus_meta), Path(args.corpus_emb),
                                descriptions_path=Path(args.corpus_desc))
    tasks_by_id = {t.task_id: t for t in load_tasks(Path(args.tasks))}
    tasks = [tasks_by_id[tid] for tid in task_ids]

    embeds = np.load(args.task_embeds, allow_pickle=False)
    task_emb_by_id = dict(zip(embeds["task_ids"].tolist(), embeds["task_embeddings"]))
    gt_offsets = embeds["gt_offsets"]
    gt_ids = embeds["gt_ids"].tolist()
    gt_emb = embeds["gt_embeddings"]
    gt_index_of_task_id = {tid: i for i, tid in enumerate(embeds["task_ids"].tolist())}

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = dt.datetime.now().strftime("%Y-%m-%d-%H%M")
        out_dir = Path("skills retrieval/runs") / f"{ts}-{args.label}"
    for sub in ("raw", "parsed", "pools", "metrics", "figures"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    (out_dir / "skipped.jsonl").touch()

    run_cfg = RunConfig(
        label=args.label, model=args.model,
        task_ids=task_ids, strategies=args.strategies,
        pool_sizes=args.pool_sizes, seeds=args.seeds,
        max_concurrency=args.concurrency,
    )
    (out_dir / "config.json").write_text(run_cfg.model_dump_json(indent=2))

    driver = CLIDriver(model=args.model, max_concurrency=args.concurrency)

    async def run_pool(task, spec: PoolSpec):
        parsed_dir = out_dir / "parsed"
        probe_records: list[TrialRecord] = []
        need_any_probe = False
        for probe in ["awareness", "selection"]:
            target = parsed_dir / f"{spec.pool_id}__{probe}.json"
            if args.skip_existing and target.exists():
                probe_records.append(TrialRecord.model_validate_json(target.read_text()))
                continue
            need_any_probe = True

        if not need_any_probe:
            # All probes already done. Still need the pool for scoring.
            pool_json = (out_dir / "pools" / f"{spec.pool_id}.json")
            if pool_json.exists():
                pd = json.loads(pool_json.read_text())
                return spec.pool_id, probe_records, {
                    "id_map": pd["id_map"],
                    "gt_display_ids": pd["gt_display_ids"],
                }

        # Build pool (cheap; safe to redo)
        t_idx = gt_index_of_task_id[task.task_id]
        gt_start, gt_end = int(gt_offsets[t_idx]), int(gt_offsets[t_idx + 1])
        task_gt_ids = gt_ids[gt_start:gt_end]
        task_gt_embs = gt_emb[gt_start:gt_end]
        # Pair each GT id with its body from the task (order matches tasks.jsonl GT order).
        gt_bodies = task.gt_skill_bodies[: len(task_gt_ids)]
        gt_entries = [
            (gid, gid.rsplit("_", 1)[-1], body, emb)
            for gid, body, emb in zip(task_gt_ids, gt_bodies, task_gt_embs)
        ]
        pool = build_pool(spec, task, corpus,
                          task_embedding=task_emb_by_id[task.task_id],
                          gt_entries=gt_entries)
        pool_block = render_pool_block(pool, representation="card")
        (out_dir / "pools" / f"{spec.pool_id}.json").write_text(json.dumps({
            "spec": spec.model_dump(),
            "display_ids": pool.display_ids,
            "id_map": pool.id_map,
            "gt_display_ids": pool.gt_display_ids,
        }, indent=2))

        for probe in ["awareness", "selection"]:
            target = parsed_dir / f"{spec.pool_id}__{probe}.json"
            if args.skip_existing and target.exists():
                continue
            full_prompt = (render_awareness_prompt(task.instruction, pool)
                           if probe == "awareness"
                           else render_selection_prompt(task.instruction, pool))
            if not will_fit(full_prompt, MODEL_CONTEXT_LIMIT, run_cfg.context_safety_margin):
                with (out_dir / "skipped.jsonl").open("a") as f:
                    f.write(json.dumps({"pool_id": spec.pool_id, "probe": probe,
                                        "reason": "context_overflow"}) + "\n")
                continue
            response_instruction = (
                "  <skills>ID_1,ID_2,ID_3,ID_4,ID_5</skills>  # EXACTLY 5 skills, ordered from MOST to LEAST relevant"
                if probe == "awareness"
                else "  <skill>SKILL_ID</skill>              # single best skill for solving this task"
            )
            user_prompt = (
                f"Task:\n{task.instruction}\n\n"
                f"Respond with EXACTLY ONE of:\n{response_instruction}\n\nNo other text."
            )
            rec = await driver.run_one(
                pool_id=spec.pool_id, probe=probe,
                system_prompt="You are a retrieval subject in a controlled study.",
                pool_block=pool_block,
                user_prompt=user_prompt,
            )
            probe_records.append(rec)
            (out_dir / "raw" / f"{spec.pool_id}__{probe}.txt").write_text(rec.raw_response)
            target.write_text(rec.model_dump_json(indent=2))
        return spec.pool_id, probe_records, {"id_map": pool.id_map, "gt_display_ids": pool.gt_display_ids}

    coros = []
    for task in tasks:
        for strategy in args.strategies:
            for n in args.pool_sizes:
                for seed in args.seeds:
                    spec = PoolSpec(task_id=task.task_id, strategy=strategy, n=n, seed=seed)
                    coros.append(run_pool(task, spec))

    print(f"Dispatching {len(coros)} pool-level tasks ({len(coros)*2} API calls max) "
          f"with concurrency={args.concurrency}...")
    results = await asyncio.gather(*coros)

    # Build per-trial rows (score using the pool's id_map + gt_display_ids)
    per_trial: list[dict] = []
    for pool_id, recs, pool_map in results:
        trial_row: dict = {"pool_id": pool_id}
        for rec in recs:
            parsed = {"extracted_ids": rec.extracted_ids,
                      "format_status": rec.format_status, "flags": rec.flags}
            scored = score_trial(parsed, pool_map, probe=rec.probe)
            for k, v in scored.items():
                if isinstance(v, (int, float)):
                    trial_row[f"{rec.probe}.{k}"] = v
            trial_row[f"{rec.probe}.format_status"] = rec.format_status
        per_trial.append(trial_row)

    (out_dir / "metrics" / "per_trial.jsonl").write_text(
        "\n".join(json.dumps(r) for r in per_trial)
    )

    # Aggregate + save summary
    rows = load_per_trial(out_dir / "metrics" / "per_trial.jsonl")
    agg = aggregate_by_condition(rows, n_boot=1000, seed=0)
    summary_json = {f"{s}__n{n}": v for (s, n), v in agg.items()}
    (out_dir / "metrics" / "summary_by_condition.json").write_text(
        json.dumps(summary_json, indent=2)
    )

    # Plots
    plot_collapse_curve(agg, out_dir / "figures" / "collapse_curve_recall5.pdf",
                        metric="awareness_recall5")
    plot_collapse_curve(agg, out_dir / "figures" / "collapse_curve_mrr.pdf",
                        metric="awareness_mrr")
    plot_collapse_curve(agg, out_dir / "figures" / "collapse_curve_selection.pdf",
                        metric="selection_top1")
    plot_selection_aware_divergence(agg, out_dir / "figures" / "selection_aware_divergence.pdf")

    print(f"Done. Output under {out_dir}")
    print(json.dumps({k: v["awareness_recall5"]["mean"] for k, v in summary_json.items()}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Import-check**

```bash
cd "/anvil/projects/x-cis260386/william/procmem2skills/procmem2skills"
PYTHONPATH="skills retrieval/src" /anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/python3.10 \
  -c "from skills_retrieval import run_m2; print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Smoke-run (1 task × 1 strategy × N=5 × 1 seed × 2 probes = 2 calls)**

```bash
cd "/anvil/projects/x-cis260386/william/procmem2skills/procmem2skills"
PYTHONPATH="skills retrieval/src" /anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/python3.10 \
  -m skills_retrieval.run_m2 \
  --task-ids sb_000 --pool-sizes 5 --strategies random --seeds 0 --label plan2-smoke
```

Expected: Completes in ~10s. Output: `skills retrieval/runs/<ts>-plan2-smoke/metrics/summary_by_condition.json` with `random__n5` showing `awareness_recall5.mean == 1.0`. Figures generated under `figures/`.

Inspect:
```bash
cat "skills retrieval/runs/"*plan2-smoke*"/metrics/summary_by_condition.json" | head -30
ls "skills retrieval/runs/"*plan2-smoke*"/figures/"
```

- [ ] **Step 5: Commit**

```bash
git add "skills retrieval/src/skills_retrieval/run_m2.py"
git commit -m "$(cat <<'EOF'
feat(skills-retrieval): Plan 2 full-sweep runner with resume and figures

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Pilot-scale Plan 2 validation (9 tasks, safety check)

Before committing to a 4–5 hour run, validate the pipeline on a small-but-nontrivial slice.

- [ ] **Step 1: Run pilot**

```bash
cd "/anvil/projects/x-cis260386/william/procmem2skills/procmem2skills"
PYTHONPATH="skills retrieval/src" /anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/python3.10 \
  -m skills_retrieval.run_m2 \
  --task-ids sb_000 sb_002 sb_003 sb_004 sb_005 sb_006 sb_007 sb_008 sb_009 \
  --pool-sizes 5 50 200 \
  --strategies random hard_neg_semantic \
  --seeds 0 1 2 \
  --concurrency 4 \
  --label plan2-pilot
```

Expected: 9 tasks × 2 strategies × 3 N × 3 seeds × 2 probes = **324 calls**, ~15–20 min wall clock.

- [ ] **Step 2: Inspect**

```bash
cat "skills retrieval/runs/"*plan2-pilot*"/metrics/summary_by_condition.json"
ls -la "skills retrieval/runs/"*plan2-pilot*"/figures/"
```

Sanity checks:
- `random__n5.awareness_recall5.mean ≈ 1.0`
- `hard_neg_semantic__n200.awareness_recall5.mean < random__n200.awareness_recall5.mean`
- `parse_fail.mean ≈ 0.0` across all conditions
- 4 PDF files present under `figures/`

If any check fails, STOP and debug before running the full sweep.

- [ ] **Step 3: Open the PDFs to eyeball them**

```bash
# If the host doesn't have a PDF viewer, at least confirm file integrity
/anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/python3.10 -c "
from pathlib import Path
for p in Path('skills retrieval/runs').glob('*plan2-pilot*/figures/*.pdf'):
    print(p, p.stat().st_size)
"
```

Expected: each PDF >3 KB.

- [ ] **Step 4: Commit pilot run artifacts**

```bash
cd "/anvil/projects/x-cis260386/william/procmem2skills/procmem2skills"
git add "skills retrieval/runs/"*plan2-pilot*"/config.json" \
        "skills retrieval/runs/"*plan2-pilot*"/metrics/summary_by_condition.json" \
        "skills retrieval/runs/"*plan2-pilot*"/figures/"
git commit -m "$(cat <<'EOF'
data(skills-retrieval): Plan 2 pilot run (9 tasks × 2 strat × 3 N × 3 seeds)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Full Plan 2 sweep (88 tasks)

- [ ] **Step 1: Launch the full sweep**

Run in foreground or `tmux`/`nohup`. The process is resumable via `--skip-existing`.

```bash
cd "/anvil/projects/x-cis260386/william/procmem2skills/procmem2skills"
PYTHONPATH="skills retrieval/src" /anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/python3.10 \
  -m skills_retrieval.run_m2 \
  --concurrency 4 \
  --label plan2-m2 \
  2>&1 | tee "skills retrieval/runs/plan2-m2.log"
```

Expected: 88 tasks × 2 strategies × 5 N × 3 seeds = **2,640 pool-level tasks → 5,280 API calls**. At ~3s/call with concurrency=4, wall time is roughly 66 minutes; realistically 4–5 hours when CLI startup overhead and rate-limit backoffs are counted.

If interrupted: find the run dir (`ls -dt "skills retrieval/runs/"*plan2-m2* | head -1`) and resume:

```bash
PYTHONPATH="skills retrieval/src" /anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/python3.10 \
  -m skills_retrieval.run_m2 \
  --concurrency 4 \
  --label plan2-m2 \
  --out-dir "<the-run-dir-from-above>" \
  --skip-existing \
  2>&1 | tee -a "skills retrieval/runs/plan2-m2.log"
```

- [ ] **Step 2: Spot-check intermediate progress**

Every ~30 min:
```bash
ls "skills retrieval/runs/"*plan2-m2*"/parsed/" | wc -l   # number of completed (pool_id, probe) pairs; target 5280
wc -l "skills retrieval/runs/"*plan2-m2*"/skipped.jsonl"  # should remain 0 for N ≤ 1000
```

Context-overflow skips at N=1000 are possible but unlikely (the card representation is ~60 chars/skill, so 1000 skills ≈ 60k chars ≈ 15k tokens — well under 200k).

- [ ] **Step 3: Inspect final summary**

```bash
cat "skills retrieval/runs/"*plan2-m2*"/metrics/summary_by_condition.json" | /anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/python3.10 -c "
import json, sys
d = json.load(sys.stdin)
print(f'{"condition":<40} {"R@5":>6} {"Sel":>6} {"Sel|Aw":>8} {"n":>5}')
for k, v in sorted(d.items()):
    print(f'{k:<40} {v[\"awareness_recall5\"][\"mean\"]:>6.3f} {v[\"selection_top1\"][\"mean\"]:>6.3f} {v[\"selection_given_aware\"][\"mean\"]:>8.3f} {v[\"awareness_recall5\"][\"n_trials\"]:>5d}')
"
```

- [ ] **Step 4: Write ANALYSIS.md**

Write `skills retrieval/runs/<ts>-plan2-m2/ANALYSIS.md` (replace `<ts>` with actual run dir) covering:

1. **Headline numbers:** random and hard_neg_semantic at each N, with CIs. Note where the two curves diverge.
2. **Comparison to Plan 1 M1 (fixed):** did the 5→88 task expansion preserve the qualitative picture (random flat, hard_neg degrading)?
3. **Selection vs Awareness:** is Selection|Aware already near 1.0 for random at all N (retrieval is trivial once pool is known)? Does hard_neg_semantic show a Sel|Aware drop indicating genuine confusability, or is its failure mostly in awareness (ranking)?
4. **Parse failure rate:** should be ≤1% across the full run. If higher, flag the pool sizes where it concentrates.
5. **Open questions:** what does the N=1000 point look like — does the hard_neg curve keep falling or plateau? This motivates Plan 4 (additional distractor types).

Include the table from Step 3 and reference the four figures from `figures/`.

- [ ] **Step 5: Commit results**

```bash
cd "/anvil/projects/x-cis260386/william/procmem2skills/procmem2skills"
git add "skills retrieval/runs/"*plan2-m2*"/config.json" \
        "skills retrieval/runs/"*plan2-m2*"/metrics/summary_by_condition.json" \
        "skills retrieval/runs/"*plan2-m2*"/metrics/per_trial.jsonl" \
        "skills retrieval/runs/"*plan2-m2*"/figures/" \
        "skills retrieval/runs/"*plan2-m2*"/ANALYSIS.md" \
        "skills retrieval/runs/plan2-m2.log"
git commit -m "$(cat <<'EOF'
data(skills-retrieval): Plan 2 full sweep (88 tasks × 2 strat × 5 N × 3 seeds)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

(Raw/parsed directories remain gitignored via `.gitignore` from Plan 1.)

---

## Self-review

**Spec coverage (against `skills retrieval/design-v2.md`):**
- §1 RQ1 (collapse curve): Task 4 (plot) + Task 7 (data).
- §1 RQ4 (Sel|Aware divergence): `plot_selection_aware_divergence` in Task 4 + data in Task 7.
- §5 bootstrap CI: Task 2.
- §5 per-condition aggregation: Task 3.
- §6.3 run output layout: Task 5 (`run_m2.py`).
- §8 M2 milestone (full 47+ task sweep): Task 7. (88 tasks, superset of the 47 cited in design-v2.)

**Not in this plan (by design; see Goal/Out-of-scope):**
- Representation ablation — Plan 3.
- easy_neg / hard_neg_functional / adversarial distractors — Plan 4.
- Format perturbation, confound control, FEM, per-task difficulty regression — Plan 5.
- Phase B — Plan 6.

**Placeholder scan:** No TBDs; every step shows the exact code or command.

**Type consistency:**
- `aggregate_by_condition(rows, n_boot, seed)` returns dict keyed by `(strategy:str, n:int)` → both used in `plots.py` and `run_m2.py`.
- `parse_pool_id` returns `{"task_id", "strategy", "n", "seed", "representation"}` — consumed by `aggregate.py`.
- `CLIDriver.run_one` signature matches `Driver.run_one` from Plan 1 — no changes required.
- `Pool.id_map` and `Pool.gt_display_ids` used by `run_m2.py`'s scoring block, same as `run_m1.py`.

---

## Execution handoff

Plan complete and saved to `skills retrieval/plans/2026-04-18-plan-2-collapse-curve.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Fresh subagent per task with review between tasks.

**2. Inline Execution** — Batch tasks in this session with checkpoints.

Which approach?
