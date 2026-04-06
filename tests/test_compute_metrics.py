import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.selection_collapse.compute_metrics import (
    compute_awareness_recall,
    compute_selection_accuracy,
    compute_selection_given_aware,
    bootstrap_se,
)


def test_awareness_recall_basic():
    results = [
        {"gt_in_top5": True},
        {"gt_in_top5": True},
        {"gt_in_top5": False},
        {"gt_in_top5": True},
    ]
    assert compute_awareness_recall(results) == 0.75


def test_selection_accuracy_basic():
    results = [
        {"any_gt_selected": True},
        {"any_gt_selected": False},
        {"any_gt_selected": True},
        {"any_gt_selected": False},
    ]
    assert compute_selection_accuracy(results) == 0.5


def test_selection_given_aware():
    """Selection|Aware = P(select GT | GT in awareness top-5)."""
    awareness = [
        {"task_id": "t1", "trial": 0, "gt_in_top5": True},
        {"task_id": "t2", "trial": 0, "gt_in_top5": True},
        {"task_id": "t3", "trial": 0, "gt_in_top5": False},
        {"task_id": "t4", "trial": 0, "gt_in_top5": True},
    ]
    selection = [
        {"task_id": "t1", "trial": 0, "any_gt_selected": True},
        {"task_id": "t2", "trial": 0, "any_gt_selected": False},
        {"task_id": "t3", "trial": 0, "any_gt_selected": True},
        {"task_id": "t4", "trial": 0, "any_gt_selected": True},
    ]
    # Aware tasks: t1(selected), t2(not selected), t4(selected) = 2/3
    result = compute_selection_given_aware(awareness, selection)
    assert abs(result - 2/3) < 1e-6


def test_bootstrap_se():
    values = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    se = bootstrap_se(values, n_bootstrap=1000, seed=42)
    assert 0.0 < se < 0.5
