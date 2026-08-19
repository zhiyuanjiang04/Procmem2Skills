from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_clean_layout():
    expected = [
        ROOT / "scripts" / "run_raw.sh",
        ROOT / "scripts" / "collect_traces.sh",
        ROOT / "scripts" / "export_workflows.py",
        ROOT / "scripts" / "generate_skills.py",
        ROOT / "scripts" / "run_eval.sh",
        ROOT / "scripts" / "run_context_comparison.py",
        ROOT / "scripts" / "retrieval" / "run_embedding_retrieval.py",
        ROOT / "scripts" / "retrieval" / "run_agent_pick.py",
        ROOT / "scripts" / "retrieval" / "run_real_execution.sh",
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    assert not missing, missing
