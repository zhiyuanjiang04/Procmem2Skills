"""Stage 4: aggregate paired labels into the final report tables.

Inputs:
  outputs/pair_labels_v1.jsonl   (528 paired records from stage 3)
  outputs/manifest.jsonl         (8135 trial records, for token/cost/duration)

Outputs:
  outputs/report_v1.md           (human-readable summary with tables + key numbers)
  outputs/report_v1_tables.json  (raw tables for downstream plotting)

No LLM calls. Pure pandas-style aggregation.
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parent
PAIR_LABELS = ROOT / "outputs" / "pair_labels_v1.jsonl"
MANIFEST = ROOT / "outputs" / "manifest.jsonl"
OUT_MD = ROOT / "outputs" / "report_v1.md"
OUT_JSON = ROOT / "outputs" / "report_v1_tables.json"

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 0


def load_pair_labels() -> list[dict]:
    return [json.loads(l) for l in open(PAIR_LABELS)]


def load_manifest_index() -> dict[str, dict]:
    """Index manifest by trial_id for cost/token/duration lookups."""
    idx = {}
    for line in open(MANIFEST):
        r = json.loads(line)
        if r.get("trial_id"):
            idx[r["trial_id"]] = r
    return idx


def mode_freq_by_arm(records: list[dict]) -> dict:
    """Count: how often each mode appears in each arm."""
    counts = defaultdict(lambda: defaultdict(int))   # mode -> arm -> count
    for r in records:
        per_arm = (r.get("labels") or {}).get("per_arm") or {}
        for arm in ("raw", "workflow", "skill"):
            a = per_arm.get(arm) or {}
            mode = a.get("mode")
            if mode:
                counts[mode][arm] += 1
    # Convert to sorted list of dicts
    rows = []
    for mode, arm_counts in counts.items():
        rows.append({
            "mode": mode,
            "raw": arm_counts.get("raw", 0),
            "workflow": arm_counts.get("workflow", 0),
            "skill": arm_counts.get("skill", 0),
        })
    rows.sort(key=lambda x: -(x["raw"] + x["workflow"] + x["skill"]))
    return rows


def paired_mode_delta(records: list[dict], comparison: str) -> dict:
    """For each mode, count how many times it was fixed_mode vs introduced_mode."""
    fixed_counts = defaultdict(int)
    introduced_counts = defaultdict(int)
    n_total = 0
    for r in records:
        delta = ((r.get("labels") or {}).get("deltas") or {}).get(comparison) or {}
        if not delta:
            continue
        n_total += 1
        for m in delta.get("fixed_mode", []) or []:
            fixed_counts[m] += 1
        for m in delta.get("introduced_mode", []) or []:
            introduced_counts[m] += 1
    all_modes = set(fixed_counts.keys()) | set(introduced_counts.keys())
    rows = []
    for m in all_modes:
        fxd = fixed_counts[m]
        intro = introduced_counts[m]
        rows.append({
            "mode": m,
            "fixed_by_treatment": fxd,
            "introduced_by_treatment": intro,
            "net_delta": -(intro - fxd),    # negative = treatment hurt; positive = treatment helped
        })
    rows.sort(key=lambda x: -x["net_delta"])
    return {"comparison": comparison, "n_pairs": n_total, "rows": rows}


def net_effect_dist(records: list[dict]) -> dict:
    out = {}
    for cmp in ("workflow_vs_raw", "skill_vs_raw", "skill_vs_workflow"):
        c = Counter()
        for r in records:
            delta = ((r.get("labels") or {}).get("deltas") or {}).get(cmp) or {}
            ne = delta.get("net_effect")
            if ne:
                c[ne] += 1
        out[cmp] = dict(c)
    return out


def mechanism_dist(records: list[dict]) -> dict:
    out = {}
    for key in ("skill_mechanism", "workflow_mechanism"):
        c = Counter()
        for r in records:
            lbl = r.get("labels") or {}
            v = lbl.get(key)
            if v:
                c[v] += 1
        out[key] = dict(c)
    return out


def by_setting(records: list[dict]) -> dict:
    """Net_effect distribution per setting (5s0f, 4s1f, ..., 0s5f)."""
    by = defaultdict(lambda: defaultdict(Counter))   # setting -> cmp -> Counter
    for r in records:
        s = r.get("setting")
        if not s:
            continue
        for cmp in ("workflow_vs_raw", "skill_vs_raw", "skill_vs_workflow"):
            delta = ((r.get("labels") or {}).get("deltas") or {}).get(cmp) or {}
            ne = delta.get("net_effect")
            if ne:
                by[s][cmp][ne] += 1
    return {s: {c: dict(v) for c, v in d.items()} for s, d in by.items()}


def by_benchmark(records: list[dict]) -> dict:
    by = defaultdict(lambda: defaultdict(Counter))
    for r in records:
        b = r.get("benchmark")
        for cmp in ("workflow_vs_raw", "skill_vs_raw", "skill_vs_workflow"):
            delta = ((r.get("labels") or {}).get("deltas") or {}).get(cmp) or {}
            ne = delta.get("net_effect")
            if ne:
                by[b][cmp][ne] += 1
    return {b: {c: dict(v) for c, v in d.items()} for b, d in by.items()}


def success_rate_by_arm(records: list[dict]) -> dict:
    """Per-arm success rate, overall + per setting."""
    overall = {"raw": [0, 0], "workflow": [0, 0], "skill": [0, 0]}   # [success, total]
    per_setting = defaultdict(lambda: {"raw": [0, 0], "workflow": [0, 0], "skill": [0, 0]})
    for r in records:
        s = r.get("setting")
        per_arm = (r.get("labels") or {}).get("per_arm") or {}
        for arm in ("raw", "workflow", "skill"):
            a = per_arm.get(arm) or {}
            st = a.get("status")
            if st in ("success", "failure"):
                overall[arm][1] += 1
                per_setting[s][arm][1] += 1
                if st == "success":
                    overall[arm][0] += 1
                    per_setting[s][arm][0] += 1
    def rate(pair):
        return round(pair[0] / pair[1], 4) if pair[1] else None
    return {
        "overall": {a: {"success": v[0], "total": v[1], "rate": rate(v)} for a, v in overall.items()},
        "per_setting": {s: {a: {"success": v[0], "total": v[1], "rate": rate(v)} for a, v in d.items()} for s, d in per_setting.items()},
    }


def cost_token_duration_compare(records: list[dict], manifest_idx: dict) -> dict:
    """For each (task, setting) triple, compute average cost/token/duration per arm."""
    per_arm_metrics = {"raw": defaultdict(list), "workflow": defaultdict(list), "skill": defaultdict(list)}
    paired_deltas = {"workflow_vs_raw": defaultdict(list), "skill_vs_raw": defaultdict(list), "skill_vs_workflow": defaultdict(list)}

    for r in records:
        trial_ids = r.get("trial_ids") or {}
        per_trial = {}
        for arm in ("raw", "workflow", "skill"):
            tid = trial_ids.get(arm)
            m = manifest_idx.get(tid) if tid else None
            if m is None:
                continue
            per_trial[arm] = {
                "input_tokens": m.get("agent_input_tokens"),
                "cache_tokens": m.get("agent_cache_tokens"),
                "output_tokens": m.get("agent_output_tokens"),
                "duration_sec": m.get("agent_execution_sec"),
            }
        for arm, metrics in per_trial.items():
            for k, v in metrics.items():
                if v is not None:
                    per_arm_metrics[arm][k].append(v)
        # paired
        for cmp, (a, b) in [("workflow_vs_raw", ("workflow", "raw")),
                             ("skill_vs_raw", ("skill", "raw")),
                             ("skill_vs_workflow", ("skill", "workflow"))]:
            if a in per_trial and b in per_trial:
                for k in ("input_tokens", "output_tokens", "duration_sec"):
                    va, vb = per_trial[a].get(k), per_trial[b].get(k)
                    if va is not None and vb is not None:
                        paired_deltas[cmp][k].append(va - vb)

    def summarize(lst):
        if not lst:
            return None
        return {
            "n": len(lst),
            "mean": round(mean(lst), 2),
            "median": round(median(lst), 2),
            "stdev": round(stdev(lst), 2) if len(lst) > 1 else None,
        }

    return {
        "per_arm": {a: {k: summarize(v) for k, v in d.items()} for a, d in per_arm_metrics.items()},
        "paired_deltas": {c: {k: summarize(v) for k, v in d.items()} for c, d in paired_deltas.items()},
    }


def bootstrap_ci(samples: list[float], n_iter: int = BOOTSTRAP_N, ci: float = 0.95, seed: int = BOOTSTRAP_SEED) -> dict | None:
    if not samples:
        return None
    rng = random.Random(seed)
    n = len(samples)
    means = []
    for _ in range(n_iter):
        s = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo_idx = int(((1 - ci) / 2) * n_iter)
    hi_idx = int((1 - (1 - ci) / 2) * n_iter) - 1
    return {
        "mean": round(mean(samples), 4),
        "ci_lo": round(means[lo_idx], 4),
        "ci_hi": round(means[hi_idx], 4),
        "n": n,
    }


def paired_success_rate_bootstrap(records: list[dict]) -> dict:
    """For each comparison arm, compute per-task delta in success rate with bootstrap CI."""
    deltas = {"workflow_vs_raw": [], "skill_vs_raw": [], "skill_vs_workflow": []}
    for r in records:
        per_arm = (r.get("labels") or {}).get("per_arm") or {}
        st = {arm: ((per_arm.get(arm) or {}).get("status") == "success") for arm in ("raw", "workflow", "skill")
              if per_arm.get(arm) and (per_arm.get(arm) or {}).get("status") in ("success", "failure")}
        if "raw" in st:
            if "workflow" in st:
                deltas["workflow_vs_raw"].append(int(st["workflow"]) - int(st["raw"]))
            if "skill" in st:
                deltas["skill_vs_raw"].append(int(st["skill"]) - int(st["raw"]))
        if "workflow" in st and "skill" in st:
            deltas["skill_vs_workflow"].append(int(st["skill"]) - int(st["workflow"]))
    return {cmp: bootstrap_ci(v) for cmp, v in deltas.items()}


def mode_row_table_md(rows: list[dict]) -> str:
    if not rows:
        return "(empty)\n"
    headers = list(rows[0].keys())
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(out) + "\n"


def main():
    print(f"loading paired labels from {PAIR_LABELS}...")
    records = load_pair_labels()
    print(f"  {len(records)} records")

    print(f"loading manifest from {MANIFEST}...")
    manifest_idx = load_manifest_index()
    print(f"  {len(manifest_idx)} trials indexed by trial_id")

    print("computing aggregates...")
    tables = {
        "n_records": len(records),
        "mode_freq_by_arm": mode_freq_by_arm(records),
        "paired_mode_delta": {
            "workflow_vs_raw": paired_mode_delta(records, "workflow_vs_raw"),
            "skill_vs_raw": paired_mode_delta(records, "skill_vs_raw"),
            "skill_vs_workflow": paired_mode_delta(records, "skill_vs_workflow"),
        },
        "net_effect_dist": net_effect_dist(records),
        "mechanism_dist": mechanism_dist(records),
        "by_setting": by_setting(records),
        "by_benchmark": by_benchmark(records),
        "success_rate": success_rate_by_arm(records),
        "cost_token_duration": cost_token_duration_compare(records, manifest_idx),
        "success_rate_bootstrap": paired_success_rate_bootstrap(records),
    }

    OUT_JSON.write_text(json.dumps(tables, ensure_ascii=False, indent=2))

    # Write markdown
    md = []
    md.append("# Paired Failure-Mode Report (v1)\n")
    md.append(f"Analyzed {tables['n_records']} (task, setting) triples across "
              f"raw / workflow / skill arms.\n")

    # Success rate
    md.append("## 1. Overall success rate per arm\n")
    sr = tables["success_rate"]["overall"]
    md.append("| arm | success | total | rate |")
    md.append("|---|---:|---:|---:|")
    for arm in ("raw", "workflow", "skill"):
        v = sr[arm]
        md.append(f"| {arm} | {v['success']} | {v['total']} | {(v['rate'] or 0)*100:.1f}% |")
    md.append("")

    # Paired success bootstrap
    md.append("## 2. Paired success-rate delta (bootstrap 95% CI, n_iter=1000)\n")
    md.append("| comparison | mean delta | 95% CI | n_pairs |")
    md.append("|---|---:|---|---:|")
    for cmp, v in tables["success_rate_bootstrap"].items():
        if v is None:
            md.append(f"| {cmp} | - | - | 0 |")
        else:
            md.append(f"| {cmp} | {v['mean']:+.4f} | [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}] | {v['n']} |")
    md.append("")

    # Net effect distribution
    md.append("## 3. Net-effect distribution\n")
    md.append("How does the treatment compare to baseline overall?\n")
    md.append("| comparison | fixed | regressed | unchanged | mixed | not_comparable |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for cmp, c in tables["net_effect_dist"].items():
        md.append(f"| {cmp} | {c.get('fixed', 0)} | {c.get('regressed', 0)} | {c.get('unchanged', 0)} | {c.get('mixed', 0)} | {c.get('not_comparable', 0)} |")
    md.append("")

    # Mode frequency
    md.append("## 4. Mode frequency per arm (out of 528 triples)\n")
    md.append(mode_row_table_md(tables["mode_freq_by_arm"]))

    # Paired mode delta — skill_vs_raw (the headline)
    md.append("## 5. Paired mode delta — skill vs raw (sorted by net help)\n")
    md.append("`fixed_by_treatment` = mode appeared in raw, gone in skill.")
    md.append("`introduced_by_treatment` = mode appeared in skill, not in raw.\n")
    md.append(mode_row_table_md(tables["paired_mode_delta"]["skill_vs_raw"]["rows"]))

    # Paired mode delta — workflow_vs_raw
    md.append("## 6. Paired mode delta — workflow vs raw\n")
    md.append(mode_row_table_md(tables["paired_mode_delta"]["workflow_vs_raw"]["rows"]))

    # Paired mode delta — skill_vs_workflow
    md.append("## 7. Paired mode delta — skill vs workflow\n")
    md.append(mode_row_table_md(tables["paired_mode_delta"]["skill_vs_workflow"]["rows"]))

    # Mechanism distribution
    md.append("## 8. Mechanism distribution\n")
    md.append("How is the treatment helping (or hurting)?\n")
    for key, c in tables["mechanism_dist"].items():
        md.append(f"### `{key}`")
        for m, n in sorted(c.items(), key=lambda x: -x[1]):
            md.append(f"- {m}: {n}")
        md.append("")

    # By setting (5s0f -> 0s5f)
    md.append("## 9. Trends by setting (workflow-success mix)\n")
    md.append("Setting `5s0f` = 5 success workflows, 0 failure. `0s5f` = 5 failure, 0 success.\n")
    settings_order = ["5s0f", "4s1f", "3s2f", "2s3f", "1s4f", "0s5f"]
    for cmp in ("skill_vs_raw", "workflow_vs_raw"):
        md.append(f"### `{cmp}` net_effect by setting")
        md.append("| setting | fixed | regressed | unchanged | mixed |")
        md.append("|---|---:|---:|---:|---:|")
        for s in settings_order:
            d = tables["by_setting"].get(s, {}).get(cmp, {})
            md.append(f"| {s} | {d.get('fixed', 0)} | {d.get('regressed', 0)} | {d.get('unchanged', 0)} | {d.get('mixed', 0)} |")
        md.append("")

    # By benchmark
    md.append("## 10. By benchmark\n")
    for cmp in ("skill_vs_raw",):
        md.append(f"### `{cmp}` net_effect by benchmark")
        md.append("| benchmark | fixed | regressed | unchanged | mixed |")
        md.append("|---|---:|---:|---:|---:|")
        for b in sorted(tables["by_benchmark"].keys()):
            d = tables["by_benchmark"][b].get(cmp, {})
            md.append(f"| {b} | {d.get('fixed', 0)} | {d.get('regressed', 0)} | {d.get('unchanged', 0)} | {d.get('mixed', 0)} |")
        md.append("")

    # Cost / token / duration
    md.append("## 11. Agent token / cost / duration per arm (from manifest)\n")
    pa = tables["cost_token_duration"]["per_arm"]
    md.append("### Per-arm means (median in parens)\n")
    md.append("| arm | input_tokens | output_tokens | duration_sec |")
    md.append("|---|---:|---:|---:|")
    for arm in ("raw", "workflow", "skill"):
        d = pa.get(arm, {})
        def fmt(k):
            v = d.get(k)
            if not v:
                return "-"
            return f"{v['mean']:,.0f} ({v['median']:,.0f})"
        md.append(f"| {arm} | {fmt('input_tokens')} | {fmt('output_tokens')} | {fmt('duration_sec')} |")
    md.append("")
    md.append("### Paired per-task deltas (treatment minus baseline)\n")
    md.append("| comparison | Δ input_tokens (mean) | Δ output_tokens (mean) | Δ duration_sec (mean) |")
    md.append("|---|---:|---:|---:|")
    for cmp in ("workflow_vs_raw", "skill_vs_raw", "skill_vs_workflow"):
        d = tables["cost_token_duration"]["paired_deltas"].get(cmp, {})
        def fmt(k):
            v = d.get(k)
            if not v:
                return "-"
            return f"{v['mean']:+,.0f}"
        md.append(f"| {cmp} | {fmt('input_tokens')} | {fmt('output_tokens')} | {fmt('duration_sec')} |")
    md.append("")

    OUT_MD.write_text("\n".join(md))
    print(f"\nsaved:\n  {OUT_MD}\n  {OUT_JSON}")


if __name__ == "__main__":
    main()
