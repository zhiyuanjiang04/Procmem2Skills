#!/usr/bin/env python3
"""Classify FAIL results into error categories."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "results"
FILES = {
    "prefill": DATA_DIR / "sb_exec_20t.jsonl",
    "baseline": DATA_DIR / "sb_baselines_n5.jsonl",
}

# ── Rate-limit detection ──────────────────────────────────────────────
def is_rate_limited(r):
    stdout = (r.get("agent_stdout_tail", "") or "").lower()
    return "hit your limit" in stdout or bool(re.search(r"resets \d+:\d+", stdout))


# ── Condition label ───────────────────────────────────────────────────
def condition(r):
    nm = r.get("noise_mode", "")
    if nm in ("noskill", "gtonly"):
        return nm
    return "prefill"  # easy/hard/random all come from the prefill file


# ── Classification ────────────────────────────────────────────────────
CATEGORIES = [
    "TIMEOUT",
    "IMPORT_ERROR",
    "PATH_ERROR",
    "SETUP_FAIL",
    "TEST_FAIL",
    "TOOL_ERROR",
    "EMPTY",
    "UNKNOWN",
]


def classify(r):
    stdout = (r.get("agent_stdout_tail", "") or "")
    stderr = (r.get("agent_stderr_tail", "") or "")
    test_log = (r.get("test_log_tail", "") or "")
    combined = stdout + "\n" + stderr + "\n" + test_log
    wall = r.get("agent_wall_s", 0) or 0

    # TIMEOUT
    if wall > 600:
        return "TIMEOUT"
    low = combined.lower()
    if "timeout" in low or "killed" in low or "timed out" in low:
        return "TIMEOUT"

    # IMPORT_ERROR
    if "ImportError" in combined or "ModuleNotFoundError" in combined:
        return "IMPORT_ERROR"

    # PATH_ERROR
    if "FileNotFoundError" in combined or "No such file" in combined:
        return "PATH_ERROR"

    # SETUP_FAIL — pip failures, missing system deps
    if re.search(r"pip install.*fail|Could not install|missing.*dependenc|apt-get|No module named", combined, re.I):
        return "SETUP_FAIL"

    # TEST_FAIL — test assertions failed (code ran, wrong results)
    if re.search(r"FAILED|AssertionError|assert .* ==|test.*FAILED", combined):
        return "TEST_FAIL"

    # TOOL_ERROR
    if "command not found" in low or "permission denied" in low:
        return "TOOL_ERROR"

    # EMPTY
    if len(stdout.strip()) < 100:
        return "EMPTY"

    return "UNKNOWN"


# ── Load data ─────────────────────────────────────────────────────────
records = []
for label, path in FILES.items():
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            r["_source"] = label
            records.append(r)

# Filter to real failures
fails = [
    r for r in records
    if not r.get("pass")
    and not r.get("skipped")
    and not is_rate_limited(r)
]

print(f"Total records: {len(records)}  |  FAIL (non-skipped, non-rate-limited): {len(fails)}")
print()

# Classify
for r in fails:
    r["_category"] = classify(r)

# ── 1. Overall distribution ──────────────────────────────────────────
print("=" * 60)
print("1. OVERALL ERROR DISTRIBUTION")
print("=" * 60)
cat_counts = Counter(r["_category"] for r in fails)
for cat in CATEGORIES:
    c = cat_counts.get(cat, 0)
    pct = 100 * c / len(fails) if fails else 0
    bar = "#" * int(pct / 2)
    print(f"  {cat:<14s} {c:4d}  ({pct:5.1f}%)  {bar}")
print(f"  {'TOTAL':<14s} {len(fails):4d}")
print()

# ── 2. Per-task breakdown ────────────────────────────────────────────
print("=" * 60)
print("2. PER-TASK ERROR TYPES")
print("=" * 60)
task_cats = defaultdict(lambda: Counter())
for r in fails:
    task_cats[r["task_id"]][r["_category"]] += 1

for task in sorted(task_cats):
    cats = task_cats[task]
    total = sum(cats.values())
    parts = ", ".join(f"{k}={v}" for k, v in sorted(cats.items(), key=lambda x: -x[1]))
    print(f"  {task:<50s}  (n={total:2d})  {parts}")
print()

# ── 3. Per-condition ─────────────────────────────────────────────────
print("=" * 60)
print("3. PER-CONDITION DISTRIBUTION")
print("=" * 60)
cond_cats = defaultdict(lambda: Counter())
cond_totals = Counter()
for r in fails:
    c = condition(r)
    cond_cats[c][r["_category"]] += 1
    cond_totals[c] += 1

for cond in ["noskill", "gtonly", "prefill"]:
    cats = cond_cats[cond]
    total = cond_totals[cond]
    print(f"\n  [{cond}] — {total} failures")
    for cat in CATEGORIES:
        c = cats.get(cat, 0)
        pct = 100 * c / total if total else 0
        print(f"    {cat:<14s} {c:4d}  ({pct:5.1f}%)")
print()

# ── 4. Deep-dive: three specific tasks ──────────────────────────────
FOCUS_TASKS = {
    "azure-bgp-oscillation-route-leak": "azure-bgp",
    "civ6-adjacency-optimizer": "civ6",
    "enterprise-information-search": "enterprise-info-search",
}

print("=" * 60)
print("4. ROOT CAUSE ANALYSIS: FOCUS TASKS")
print("=" * 60)

for task_id, short in FOCUS_TASKS.items():
    task_fails = [r for r in fails if r["task_id"] == task_id]
    if not task_fails:
        print(f"\n  [{short}] — no non-rate-limited failures")
        continue

    cats = Counter(r["_category"] for r in task_fails)
    conds = Counter(condition(r) for r in task_fails)
    print(f"\n  [{short}]  {len(task_fails)} failures")
    print(f"    Categories: {dict(cats)}")
    print(f"    Conditions: {dict(conds)}")

    # Heuristic root-cause classification
    # Data Defect: missing files, wrong data format, broken fixtures
    # Capability Limit: model can't solve the task even with right setup
    # Knowledge Gap: model lacks domain knowledge
    dominant = cats.most_common(1)[0][0]

    # Sample some stdout snippets for diagnosis
    snippets = []
    for r in task_fails[:3]:
        stdout = (r.get("agent_stdout_tail", "") or "")[-300:]
        test_log = (r.get("test_log_tail", "") or "")[-300:]
        snippets.append(f"      [stdout tail] {stdout[:200]}")
        snippets.append(f"      [test_log tail] {test_log[:200]}")

    if dominant in ("PATH_ERROR", "SETUP_FAIL", "IMPORT_ERROR"):
        root = "Data Defect"
        reason = "Failures stem from missing files or broken environment setup"
    elif dominant == "TEST_FAIL":
        # Check if noskill also fails — if so, it's a capability limit
        noskill_count = sum(1 for r in task_fails if condition(r) == "noskill")
        prefill_count = sum(1 for r in task_fails if condition(r) == "prefill")
        if noskill_count > 0 and prefill_count > 0:
            root = "Capability Limit"
            reason = "Tests fail across conditions — model produces wrong results"
        else:
            root = "Knowledge Gap"
            reason = "Domain-specific assertions fail — model lacks specialized knowledge"
    elif dominant == "TIMEOUT":
        root = "Capability Limit"
        reason = "Model cannot complete the task within time budget"
    elif dominant == "EMPTY":
        root = "Capability Limit"
        reason = "Model produces no meaningful output"
    else:
        root = "Knowledge Gap"
        reason = "Unclassifiable failures suggest missing domain expertise"

    print(f"    Root cause: {root}")
    print(f"    Rationale:  {reason}")
    print(f"    Sample output snippets:")
    for s in snippets[:4]:
        print(s)
print()
