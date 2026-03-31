from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from procmem2skills.runtime.retrieval import SkillIndex


class SkillRetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="procmem2skills-retrieval-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_skill_index_parses_frontmatter_cards_and_metadata_search(self) -> None:
        self._write_skill(
            "terminal-python-inline",
            """---
name: terminal-python-inline
description: Run short inline Python diagnostics in terminal.
---

# Python Inline
Use this when you need `python -c` checks.
""",
        )
        self._write_skill(
            "terminal-grep",
            """---
name: terminal-grep
description: Search code with grep to locate symbols quickly.
---

# Grep
Use grep for pattern search.
""",
        )

        index = SkillIndex.from_repository(self.temp_dir)
        cards = index.cards()
        self.assertEqual({card.skill_id for card in cards}, {"terminal-python-inline", "terminal-grep"})
        self.assertTrue(any("inline Python diagnostics" in card.description for card in cards))

        hits = index.search("need quick python diagnostics", top_k=2, scope="metadata")
        self.assertEqual(hits[0].skill_id, "terminal-python-inline")

    def test_skill_index_supports_nested_skill_layouts(self) -> None:
        self._write_skill(
            "created_skills/4s1f/task-a/run-pytest/success",
            """---
name: run-pytest
description: Run pytest before code edits.
---

# Run Pytest
Use this for test-first debugging.
""",
        )

        index = SkillIndex.from_repository(self.temp_dir)
        nested_id = "created_skills/4s1f/task-a/run-pytest/success"
        self.assertIn(nested_id, index.records)

        cards = index.cards([nested_id])
        self.assertEqual(cards[0].name, "run-pytest")
        self.assertEqual(cards[0].skill_id, nested_id)

    def _write_skill(self, skill_id: str, body: str) -> None:
        skill_dir = self.temp_dir / skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
