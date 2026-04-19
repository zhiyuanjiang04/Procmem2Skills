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
    assert s["awareness_mrr"] == 0.5
    assert s["awareness_recall5"] == 1


def test_aggregate_selection_given_aware():
    records = [
        {"awareness_recall5": 1, "selection_top1": 1},
        {"awareness_recall5": 1, "selection_top1": 0},
        {"awareness_recall5": 0, "selection_top1": 1},
    ]
    agg = aggregate_metrics(records)
    assert agg["selection_given_aware"] == 0.5
