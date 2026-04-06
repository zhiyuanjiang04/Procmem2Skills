import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.selection_collapse.build_pools import build_pool


def _make_corpus(n=100):
    return [
        {"id": f"skill_{i:06d}", "slug": f"author/skill-{i}", "name": f"Skill {i}",
         "description": f"Description for skill {i}", "body": f"Body for skill {i}",
         "category": "general"}
        for i in range(n)
    ]


def _make_task():
    return {
        "task_id": "sb_001",
        "domain": "test",
        "instruction": "Do something",
        "gt_skills": [
            {"skill_id": "gt_000", "name": "correct-skill",
             "content": "# correct-skill\n\n## Steps\n1. Do the thing"}
        ],
    }


def test_gt_always_in_pool():
    corpus = _make_corpus(100)
    task = _make_task()
    for n in [5, 10, 50]:
        pool = build_pool(task, corpus, pool_size=n, seed=42)
        skill_ids = [s["skill_id"] for s in pool]
        assert "gt_000" in skill_ids, f"GT missing at pool_size={n}"
        assert len(pool) == n, f"Pool size wrong: {len(pool)} != {n}"


def test_gt_position_varies_with_seed():
    corpus = _make_corpus(100)
    task = _make_task()
    positions = set()
    for seed in range(10):
        pool = build_pool(task, corpus, pool_size=20, seed=seed)
        ids = [s["skill_id"] for s in pool]
        positions.add(ids.index("gt_000"))
    assert len(positions) > 1, "GT position never changed across seeds"


def test_no_duplicate_skills():
    corpus = _make_corpus(100)
    task = _make_task()
    pool = build_pool(task, corpus, pool_size=50, seed=42)
    ids = [s["skill_id"] for s in pool]
    assert len(ids) == len(set(ids)), "Duplicate skills in pool"


def test_display_ids_sequential():
    corpus = _make_corpus(100)
    task = _make_task()
    pool = build_pool(task, corpus, pool_size=10, seed=42)
    display_ids = [s["display_id"] for s in pool]
    expected = [f"SKILL_{i+1:03d}" for i in range(10)]
    assert display_ids == expected, f"Display IDs not sequential: {display_ids}"


def test_gt_marked_correctly():
    corpus = _make_corpus(100)
    task = _make_task()
    pool = build_pool(task, corpus, pool_size=10, seed=42)
    gt_entries = [s for s in pool if s["is_gt"]]
    assert len(gt_entries) == 1
    assert gt_entries[0]["skill_id"] == "gt_000"
    non_gt = [s for s in pool if not s["is_gt"]]
    assert all(s["source"] == "clawhub_distractor" for s in non_gt)


def test_distractor_ids_format():
    corpus = _make_corpus(100)
    task = _make_task()
    pool = build_pool(task, corpus, pool_size=10, seed=42)
    distractors = [s for s in pool if not s["is_gt"]]
    for s in distractors:
        assert s["skill_id"].startswith("dist_"), f"Distractor ID format wrong: {s['skill_id']}"


def test_multiple_gt_skills():
    corpus = _make_corpus(100)
    task = {
        "task_id": "sb_002",
        "domain": "test",
        "instruction": "Do something",
        "gt_skills": [
            {"skill_id": "gt_a", "name": "skill-a", "content": "content a"},
            {"skill_id": "gt_b", "name": "skill-b", "content": "content b"},
        ],
    }
    pool = build_pool(task, corpus, pool_size=10, seed=42)
    ids = [s["skill_id"] for s in pool]
    assert "gt_a" in ids
    assert "gt_b" in ids
    assert len(pool) == 10
