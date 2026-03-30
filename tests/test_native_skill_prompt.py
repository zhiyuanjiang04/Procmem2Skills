from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from procmem2skills.integrations.harbor_transfer_study import _resolve_native_skill_injection
from procmem2skills.integrations.native_skill_prompt import build_native_skill_prompt_template


def _write_skill(repo: Path, skill_id: str, *, name: str, description: str, body: str) -> None:
    skill_dir = repo / skill_id
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                "---",
                "",
                body,
            ]
        ),
        encoding="utf-8",
    )


class NativeSkillPromptTemplateTest(unittest.TestCase):
    def test_build_native_skill_prompt_template_includes_instruction_and_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "skills"
            _write_skill(
                repo,
                "build-cython",
                name="Build Cython Extension",
                description="Repair cython extension builds and rerun tests.",
                body="Use `python -m pip install -e .` before `pytest -q`.",
            )
            _write_skill(
                repo,
                "nginx-logging",
                name="Nginx Request Logging",
                description="Configure nginx access logging fields.",
                body="Update `nginx.conf` and validate with `nginx -t`.",
            )
            template_path = root / "native-skill-prompt.j2"
            meta = build_native_skill_prompt_template(
                skill_repository=repo,
                output_path=template_path,
                task_names=["build-cython-ext"],
                max_skills=1,
            )
            rendered = template_path.read_text(encoding="utf-8")
            self.assertIn("{{ instruction }}", rendered)
            self.assertIn("{% raw %}", rendered)
            self.assertIn("build-cython", rendered)
            self.assertIn("Failure Guardrails", rendered)
            self.assertEqual(meta["selected_skill_count"], 1)
            self.assertEqual(meta["selected_skill_ids"], ["build-cython"])
            self.assertFalse(meta["has_failure_guardrails"])

    def test_build_native_skill_prompt_template_renders_failure_guardrails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "skills"
            _write_skill(
                repo,
                "dna-insert-reflect",
                name="DNA Insert Recovery",
                description="Recover insertion logic after verifier mismatch.",
                body="Use deterministic script and rerun verifier.",
            )
            template_path = root / "native-skill-prompt.j2"
            meta = build_native_skill_prompt_template(
                skill_repository=repo,
                output_path=template_path,
                task_names=["dna-insert"],
                max_skills=1,
                failure_guardrails={
                    "by_task": {
                        "dna-insert": {
                            "failure_signatures": [{"signature": "AssertionError: expected marker missing", "count": 2}],
                            "failed_command_sequences": ["pytest -q -> python solve.py -> pytest -q"],
                        }
                    }
                },
            )
            rendered = template_path.read_text(encoding="utf-8")
            self.assertIn("AssertionError: expected marker missing", rendered)
            self.assertIn("Avoid repeating failed command chain", rendered)
            self.assertTrue(meta["has_failure_guardrails"])


class NativeSkillInjectionModeTest(unittest.TestCase):
    def test_resolve_native_skill_injection_auto_modes(self) -> None:
        self.assertEqual(_resolve_native_skill_injection("auto", "codex"), "prompt-template")
        self.assertEqual(_resolve_native_skill_injection("auto", "opencode"), "prompt-template")
        self.assertEqual(_resolve_native_skill_injection("auto", "terminus-2"), "none")


if __name__ == "__main__":
    unittest.main()
