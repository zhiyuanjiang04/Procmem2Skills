from __future__ import annotations

import json
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

        pf_vals = []
        for r in group:
            aware_pf = r.get("awareness.parse_fail", 0)
            sel_pf = r.get("selection.parse_fail", 0)
            pf_vals.append(max(aware_pf, sel_pf))
        mean, lo, hi = bootstrap_rate_ci(pf_vals, n_boot=n_boot, seed=seed)
        entry["parse_fail"] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n_trials": len(pf_vals)}

        agg[cond] = entry

    return agg
