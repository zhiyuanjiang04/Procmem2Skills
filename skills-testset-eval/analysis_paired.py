#!/usr/bin/env python3
"""Task-level paired statistical analysis for skill-prefill eval.

Addresses the pseudoreplication critique: the N prefill configs per task
are NOT independent samples. Instead, we compute one summary rate per task
and use paired non-parametric tests across tasks.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Rate-limit detection (narrow pattern to avoid false positives)
# ---------------------------------------------------------------------------

def is_rate_limited(r: dict) -> bool:
    stdout = (r.get("agent_stdout_tail", "") or "").lower()
    return "hit your limit" in stdout or bool(re.search(r"resets \d+:\d+", stdout))


def is_skipped(r: dict) -> bool:
    """Detect entries that were skipped (agent never ran)."""
    return r.get("agent_wall_s", 0) == 0 and not r.get("pass", False)


# ---------------------------------------------------------------------------
# Load & deduplicate
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dedup_prefill(rows: list[dict]) -> dict[tuple, dict]:
    """Deduplicate prefill rows: key = (task_id, pool_size, noise_mode).
    Keep the LAST entry for each key (latest run)."""
    out = {}
    for r in rows:
        key = (r["task_id"], r["pool_size"], r["noise_mode"])
        out[key] = r  # overwrites earlier entries
    return out


def dedup_baselines(rows: list[dict]) -> dict[tuple, dict]:
    """Deduplicate baseline rows: key = (task_id, noise_mode, seed).
    Keep the LAST entry for each key."""
    out = {}
    for r in rows:
        key = (r["task_id"], r["noise_mode"], r["seed"])
        out[key] = r
    return out


# ---------------------------------------------------------------------------
# Compute per-task rates
# ---------------------------------------------------------------------------

def compute_task_rates(prefill_map, baselines_map):
    """Return {task_id: {prefill_rate, noskill_rate, gtonly_rate, prefill_n, ...}}."""
    tasks = sorted(set(k[0] for k in prefill_map) | set(k[0] for k in baselines_map))

    results = {}
    for tid in tasks:
        # Prefill: all configs for this task (up to 12: 4 pool_sizes x 3 noise_modes)
        pf_rows = {k: v for k, v in prefill_map.items() if k[0] == tid}
        pf_valid = {k: v for k, v in pf_rows.items()
                    if not is_rate_limited(v) and not is_skipped(v)}
        pf_pass = sum(1 for v in pf_valid.values() if v.get("pass", False))
        pf_total = len(pf_valid)

        # Noskill baseline: seeds 0-4
        ns_rows = {k: v for k, v in baselines_map.items()
                   if k[0] == tid and k[1] == "noskill"}
        ns_valid = {k: v for k, v in ns_rows.items()
                    if not is_rate_limited(v) and not is_skipped(v)}
        ns_pass = sum(1 for v in ns_valid.values() if v.get("pass", False))
        ns_total = len(ns_valid)

        # Gtonly baseline: seeds 0-4
        gt_rows = {k: v for k, v in baselines_map.items()
                   if k[0] == tid and k[1] == "gtonly"}
        gt_valid = {k: v for k, v in gt_rows.items()
                    if not is_rate_limited(v) and not is_skipped(v)}
        gt_pass = sum(1 for v in gt_valid.values() if v.get("pass", False))
        gt_total = len(gt_valid)

        results[tid] = {
            "prefill_pass": pf_pass, "prefill_n": pf_total,
            "prefill_rate": pf_pass / pf_total if pf_total else float("nan"),
            "noskill_pass": ns_pass, "noskill_n": ns_total,
            "noskill_rate": ns_pass / ns_total if ns_total else float("nan"),
            "gtonly_pass": gt_pass, "gtonly_n": gt_total,
            "gtonly_rate": gt_pass / gt_total if gt_total else float("nan"),
        }
    return results


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank(x, y):
    """Wilcoxon signed-rank test on paired differences x - y.

    Returns (statistic, p_value, n_nonzero_pairs).
    Uses scipy if available, otherwise a manual implementation.
    """
    diffs = np.array(x) - np.array(y)
    nonzero = diffs[diffs != 0]
    n = len(nonzero)

    if n < 5:
        return None, None, n

    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon
        stat, p = scipy_wilcoxon(x, y, alternative="two-sided")
        return stat, p, n
    except ImportError:
        # Manual implementation
        abs_diffs = np.abs(nonzero)
        ranks = np.argsort(np.argsort(abs_diffs)) + 1.0  # simple ranking, no ties handling
        W_plus = np.sum(ranks[nonzero > 0])
        W_minus = np.sum(ranks[nonzero < 0])
        T = min(W_plus, W_minus)
        # Normal approximation for n >= 10
        if n >= 10:
            mean_T = n * (n + 1) / 4
            std_T = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            z = (T - mean_T) / std_T
            # two-tailed p from normal approximation
            from math import erfc
            p = erfc(abs(z) / np.sqrt(2))
        else:
            p = None  # can't compute without tables
        return T, p, n


def bootstrap_median_ci(x, y, n_boot=10000, alpha=0.05, seed=42):
    """Bootstrap 95% CI for the median of (x - y)."""
    rng = np.random.default_rng(seed)
    diffs = np.array(x) - np.array(y)
    n = len(diffs)
    medians = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        medians[i] = np.median(diffs[idx])
    lo = np.percentile(medians, 100 * alpha / 2)
    hi = np.percentile(medians, 100 * (1 - alpha / 2))
    observed_median = np.median(diffs)
    return observed_median, lo, hi


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_separator(char="-", width=120):
    print(char * width)


def main():
    base = Path(__file__).parent / "results"
    prefill_path = base / "sb_exec_20t.jsonl"
    baselines_path = base / "sb_baselines_n5.jsonl"

    for p in [prefill_path, baselines_path]:
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            sys.exit(1)

    # Load & dedup
    prefill_raw = load_jsonl(str(prefill_path))
    baselines_raw = load_jsonl(str(baselines_path))

    prefill_map = dedup_prefill(prefill_raw)
    baselines_map = dedup_baselines(baselines_raw)

    # Count filtered
    pf_rl = sum(1 for v in prefill_map.values() if is_rate_limited(v))
    pf_sk = sum(1 for v in prefill_map.values() if is_skipped(v))
    bl_rl = sum(1 for v in baselines_map.values() if is_rate_limited(v))
    bl_sk = sum(1 for v in baselines_map.values() if is_skipped(v))

    print("=" * 120)
    print("TASK-LEVEL PAIRED STATISTICAL ANALYSIS")
    print("=" * 120)
    print()
    print("Data summary (after deduplication, before filtering):")
    print(f"  Prefill configs:  {len(prefill_map)} unique (task x pool_size x noise_mode)")
    print(f"  Baseline entries: {len(baselines_map)} unique (task x noise_mode x seed)")
    print()
    print("Filtered out:")
    print(f"  Prefill  — rate-limited: {pf_rl}, skipped: {pf_sk}")
    print(f"  Baseline — rate-limited: {bl_rl}, skipped: {bl_sk}")

    # Compute rates
    task_rates = compute_task_rates(prefill_map, baselines_map)

    # Print per-task table
    print()
    print_separator("=")
    print("PER-TASK PASS RATES")
    print_separator("=")
    header = (
        f"{'Task':<45} {'Prefill':>12} {'NoSkill':>12} {'GtOnly':>12} "
        f"{'PF-NS':>8} {'PF-GT':>8} {'GT-NS':>8}"
    )
    print(header)
    print_separator("-")

    tasks_sorted = sorted(task_rates.keys())
    for tid in tasks_sorted:
        r = task_rates[tid]
        pf = r["prefill_rate"]
        ns = r["noskill_rate"]
        gt = r["gtonly_rate"]

        def fmt_rate(rate, n_pass, n_total):
            if np.isnan(rate):
                return "N/A"
            return f"{rate:.2f} ({n_pass}/{n_total})"

        pf_str = fmt_rate(pf, r["prefill_pass"], r["prefill_n"])
        ns_str = fmt_rate(ns, r["noskill_pass"], r["noskill_n"])
        gt_str = fmt_rate(gt, r["gtonly_pass"], r["gtonly_n"])

        def fmt_delta(a, b):
            if np.isnan(a) or np.isnan(b):
                return "N/A"
            d = a - b
            return f"{d:+.2f}"

        print(
            f"{tid:<45} {pf_str:>12} {ns_str:>12} {gt_str:>12} "
            f"{fmt_delta(pf, ns):>8} {fmt_delta(pf, gt):>8} {fmt_delta(gt, ns):>8}"
        )

    # Collect arrays for paired tests (only tasks present in all 3 conditions)
    pf_arr, ns_arr, gt_arr = [], [], []
    paired_tasks = []
    for tid in tasks_sorted:
        r = task_rates[tid]
        if np.isnan(r["prefill_rate"]) or np.isnan(r["noskill_rate"]) or np.isnan(r["gtonly_rate"]):
            continue
        pf_arr.append(r["prefill_rate"])
        ns_arr.append(r["noskill_rate"])
        gt_arr.append(r["gtonly_rate"])
        paired_tasks.append(tid)

    pf_arr = np.array(pf_arr)
    ns_arr = np.array(ns_arr)
    gt_arr = np.array(gt_arr)
    n_tasks = len(paired_tasks)

    print_separator("-")
    print(f"Tasks with all 3 conditions: {n_tasks}")
    print(f"Mean rates — Prefill: {np.mean(pf_arr):.3f}, NoSkill: {np.mean(ns_arr):.3f}, GtOnly: {np.mean(gt_arr):.3f}")
    print(f"Median rates — Prefill: {np.median(pf_arr):.3f}, NoSkill: {np.median(ns_arr):.3f}, GtOnly: {np.median(gt_arr):.3f}")

    # Paired tests
    print()
    print_separator("=")
    print("WILCOXON SIGNED-RANK TESTS (paired by task)")
    print_separator("=")
    print()

    comparisons = [
        ("Prefill vs NoSkill", pf_arr, ns_arr),
        ("Prefill vs GtOnly", pf_arr, gt_arr),
        ("GtOnly vs NoSkill", gt_arr, ns_arr),
    ]

    for label, x, y in comparisons:
        diffs = x - y
        stat, p, n_nonzero = wilcoxon_signed_rank(x, y)
        median_d, ci_lo, ci_hi = bootstrap_median_ci(x, y)
        mean_d = np.mean(diffs)

        print(f"  {label}:")
        print(f"    Mean delta:   {mean_d:+.4f}")
        print(f"    Median delta: {median_d:+.4f}")
        if stat is not None:
            p_str = f"{p:.4f}" if p is not None else "N/A (n too small for approx)"
            print(f"    Wilcoxon W:   {stat:.1f}  (n_nonzero_pairs={n_nonzero}, p={p_str})")
        else:
            print(f"    Wilcoxon: SKIPPED — only {n_nonzero} non-zero differences (need >= 5)")
        print(f"    Bootstrap 95% CI for median delta: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        # Direction summary
        n_pos = np.sum(diffs > 0)
        n_neg = np.sum(diffs < 0)
        n_tie = np.sum(diffs == 0)
        print(f"    Direction: {n_pos} tasks positive, {n_neg} negative, {n_tie} tied")
        print()

    # Caveats
    print_separator("=")
    print("CAVEATS & INTERPRETATION NOTES")
    print_separator("=")
    print("""
1. PSEUDOREPLICATION: The prefill condition has 12 configs per task (4 pool sizes
   x 3 noise modes), but these are NOT independent — they share the same task
   and ground-truth skill. We collapse to one rate per task before testing.

2. UNEQUAL DENOMINATORS: Prefill rate is computed over ~12 configs (varied pool
   sizes and noise injection), while baseline rates are over 5 seeds. These
   measure slightly different things — prefill rate captures robustness across
   skill-pool conditions, while baseline rate captures seed-to-seed variance.

3. SMALL N: With only {n_tasks} tasks, the Wilcoxon signed-rank test has limited
   power. A non-significant p-value does NOT imply no effect — it may simply
   reflect insufficient sample size.

4. BOOTSTRAP CI: The 95% CI for the median delta is computed via 10,000 bootstrap
   resamples of the {n_tasks} paired task-level differences. This is more
   appropriate than normal-theory CIs given the small, likely non-normal sample.

5. MULTIPLE COMPARISONS: Three paired tests are reported without correction.
   Apply Bonferroni (alpha = 0.05/3 = 0.017) if strict control is desired.

6. RATE-LIMIT FILTERING: Uses narrow detection ("hit your limit" or "resets H:M"
   pattern) to avoid the false-positive problem from earlier broad detection.
""".format(n_tasks=n_tasks))


if __name__ == "__main__":
    main()
