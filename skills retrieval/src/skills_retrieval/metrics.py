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
    aware_hit_with_sel = [(r.get("awareness_recall5", 0), r.get("selection_top1", 0)) for r in records if "awareness_recall5" in r and "selection_top1" in r]
    if aware_hit_with_sel:
        num = sum(s for a, s in aware_hit_with_sel if a == 1)
        den = sum(1 for a, _ in aware_hit_with_sel if a == 1)
        agg["selection_given_aware"] = num / den if den else float("nan")
    return agg
