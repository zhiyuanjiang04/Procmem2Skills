from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from procmem2skills.models import AtomicSkill, WorkflowStep
from procmem2skills.packager.skill_writer import SkillWriter


class SkillWriterDualChannelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="procmem2skills-skill-writer-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_writer_materializes_success_and_failure_variants(self) -> None:
        skill = AtomicSkill(
            skill_id="terminal-pytest",
            title="Run pytest first",
            description="Inspect test failures before editing files.",
            canonical_key="terminal-pytest",
            trigger="When tests fail in terminal tasks.",
            actions=[
                WorkflowStep(
                    order=1,
                    intent="Run tests",
                    tool="terminal",
                    operation="pytest(command=pytest -q)",
                )
            ],
            verification=["pytest -q should pass or expose deterministic failures."],
            failure_recovery=["If pytest hangs, add -k to scope failing tests."],
            support=2,
        )
        output_dir = self.temp_dir / "skills"
        written = SkillWriter().write_repository([skill], output_dir)
        names = sorted(path.name for path in written)

        self.assertEqual(names, ["terminal-pytest--failure", "terminal-pytest--success"])
        self.assertTrue((output_dir / "terminal-pytest--success" / "scripts" / "apply.sh").is_file())
        self.assertTrue((output_dir / "terminal-pytest--failure" / "scripts" / "recover.sh").is_file())
        self.assertTrue((output_dir / "terminal-pytest--success" / "scripts" / "verify.sh").is_file())
        self.assertTrue((output_dir / "terminal-pytest--failure" / "scripts" / "verify.sh").is_file())

    def test_writer_supports_created_skills_hierarchy_and_reference_files(self) -> None:
        skill = AtomicSkill(
            skill_id="controlled--task-a--4s1f",
            title="Run pytest first",
            description="Inspect test failures before editing files.",
            canonical_key="controlled:task-a:4s1f",
            trigger="When tests fail in terminal tasks.",
            actions=[
                WorkflowStep(
                    order=1,
                    intent="Run tests",
                    tool="terminal",
                    operation="terminal(command=pytest -q)",
                )
            ],
            verification=["pytest -q should pass or expose deterministic failures."],
            failure_recovery=["If pytest hangs, add -k to scope failing tests."],
            support=2,
            metadata={
                "output_layout": {
                    "root": "created_skills",
                    "condition": "4s1f",
                    "task": "task-a",
                    "skill_name": "run-pytest",
                }
            },
        )
        output_dir = self.temp_dir / "skills"
        written = SkillWriter().write_repository([skill], output_dir)

        expected_success = output_dir / "created_skills" / "4s1f" / "task-a" / "run-pytest" / "success"
        expected_failure = output_dir / "created_skills" / "4s1f" / "task-a" / "run-pytest" / "failure"

        self.assertIn(expected_success, written)
        self.assertIn(expected_failure, written)
        self.assertTrue((expected_success / "references" / "source-evidence.md").is_file())
        self.assertTrue((expected_success / "references" / "related-skills.md").is_file())

        success_skill_md = (expected_success / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("references/source-evidence.md", success_skill_md)
        self.assertIn("references/related-skills.md", success_skill_md)


if __name__ == "__main__":
    unittest.main()
