import json
from pathlib import Path

from skills_retrieval.aggregate import aggregate_by_condition, load_per_trial


def test_aggregate_by_condition_structure(tmp_path: Path):
    rows = [
        {"pool_id": "sb_000__random__n5__s0__card",
         "awareness.awareness_recall5": 1, "awareness.awareness_top1": 1, "awareness.awareness_mrr": 1.0,
         "selection.selection_top1": 1,
         "awareness.parse_fail": 0, "selection.parse_fail": 0},
        {"pool_id": "sb_000__random__n5__s1__card",
         "awareness.awareness_recall5": 1, "awareness.awareness_top1": 0, "awareness.awareness_mrr": 0.5,
         "selection.selection_top1": 0,
         "awareness.parse_fail": 0, "selection.parse_fail": 0},
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

    assert ("random", 5) in agg
    assert ("random", 50) in agg
    r5 = agg[("random", 5)]
    assert r5["awareness_recall5"]["mean"] == 1.0
    assert r5["awareness_recall5"]["n_trials"] == 2
    assert r5["selection_given_aware"]["mean"] == 0.5

    r50 = agg[("random", 50)]
    assert r50["awareness_recall5"]["mean"] == 0.0


def test_aggregate_handles_missing_probe(tmp_path: Path):
    rows = [
        {"pool_id": "sb_000__random__n1000__s0__card",
         "awareness.awareness_recall5": 1, "awareness.awareness_top1": 1, "awareness.awareness_mrr": 1.0,
         "awareness.parse_fail": 0},
    ]
    path = tmp_path / "per_trial.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    loaded = load_per_trial(path)
    agg = aggregate_by_condition(loaded, n_boot=200, seed=0)
    key = ("random", 1000)
    assert agg[key]["awareness_recall5"]["mean"] == 1.0
    assert agg[key]["selection_top1"]["n_trials"] == 0


def test_pool_id_parse():
    from skills_retrieval.aggregate import parse_pool_id
    got = parse_pool_id("sb_034__hard_neg_semantic__n200__s2__card")
    assert got == {"task_id": "sb_034", "strategy": "hard_neg_semantic", "n": 200, "seed": 2, "representation": "card"}
