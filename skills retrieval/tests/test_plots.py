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
    assert out.stat().st_size > 500


def test_divergence_plot_writes_file(tmp_path: Path):
    out = tmp_path / "divergence.pdf"
    plot_selection_aware_divergence(_tiny_agg(), out_path=out)
    assert out.exists()
    assert out.stat().st_size > 500
