import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.selection_collapse.format_catalog import format_skill_entry, format_catalog, count_tokens_approx


def test_format_skill_entry_truncates():
    skill = {
        "display_id": "SKILL_001",
        "name": "test-skill",
        "content": "x " * 500,
    }
    entry = format_skill_entry(skill, max_tokens=200)
    word_count = len(entry.split())
    assert word_count <= 250, f"Entry too long: {word_count} words"


def test_format_catalog_includes_all_skills():
    pool = [
        {"display_id": f"SKILL_{i:03d}", "name": f"skill-{i}",
         "content": f"Does thing {i}", "description": f"Desc {i}"}
        for i in range(5)
    ]
    catalog = format_catalog(pool)
    for i in range(5):
        assert f"SKILL_{i:03d}" in catalog


def test_count_tokens_approx():
    text = "hello world this is a test"
    count = count_tokens_approx(text)
    assert 4 <= count <= 10
