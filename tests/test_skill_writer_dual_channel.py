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


if __name__ == "__main__":
    unittest.main()
