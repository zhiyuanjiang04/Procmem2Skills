from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from procmem2skills.models import AtomicSkill
from procmem2skills.packager.materialize import (
    SkillGenerationConfig,
    _select_failure_context,
    materialize_skill_repository,
    materialize_skill_repository_standard_llm,
)


class MaterializeFailureContextTest(unittest.TestCase):
    def _build_skill(self) -> AtomicSkill:
        return AtomicSkill(
            skill_id="terminal-pytest",
            title="Run pytest first",
            description="Use pytest for failure triage.",
            canonical_key="terminal-pytest",
            trigger="When tests fail.",
            task_origins=["build-cython-ext"],
        )

    def test_select_failure_context_prefers_matching_task(self) -> None:
        skill = self._build_skill()
        context = _select_failure_context(
            skill=skill,
            failure_analysis={"failed_trajectories": 3},
            failure_analysis_by_task={
                "build-cython-ext": {"failures": 2, "failure_signals": [{"signature": "timeout", "count": 2}]},
                "query-optimize": {"failures": 1, "failure_signals": [{"signature": "syntax", "count": 1}]},
            },
        )

        self.assertIsNotNone(context)
        assert context is not None
        self.assertIn("global_failure_analysis", context)
        self.assertEqual(
            list((context.get("task_failure_analysis") or {}).keys()),
            ["build-cython-ext"],
        )

    def test_select_failure_context_falls_back_to_top_failures(self) -> None:
        skill = AtomicSkill(
            skill_id="terminal-grep",
            title="Search code",
            description="Use grep to locate code.",
            canonical_key="terminal-grep",
            trigger="When searching code.",
            task_origins=["unseen-task"],
        )
        context = _select_failure_context(
            skill=skill,
            failure_analysis=None,
            failure_analysis_by_task={
                "task-a": {"failures": 5},
                "task-b": {"failures": 2},
            },
        )
        assert context is not None
        self.assertEqual(
            list((context.get("task_failure_analysis") or {}).keys()),
            ["task-a", "task-b"],
        )

    def test_materialize_strict_llm_requires_api_key(self) -> None:
        with tempfile.TemporaryDirectory(prefix="materialize-strict-llm-") as temp_dir:
            with patch("procmem2skills.packager.materialize._has_any_api_key", return_value=False):
                with self.assertRaises(RuntimeError):
                    materialize_skill_repository(
                        skills=[self._build_skill()],
                        output_dir=Path(temp_dir) / "skills",
                        generation=SkillGenerationConfig(
                            mode="llm-agent",
                            model="openai/gpt-5.3-codex",
                            strict_llm=True,
                        ),
                    )

    def test_materialize_passes_agent_style_and_system_prompt_to_llm_creator(self) -> None:
        with tempfile.TemporaryDirectory(prefix="materialize-llm-style-") as temp_dir:
            with patch("procmem2skills.packager.materialize._has_any_api_key", return_value=True):
                with patch("procmem2skills.packager.materialize.LLMSkillCreator") as creator_cls:
                    creator = creator_cls.return_value
                    creator.compose_skill_variants.return_value = []
                    materialize_skill_repository(
                        skills=[self._build_skill()],
                        output_dir=Path(temp_dir) / "skills",
                        generation=SkillGenerationConfig(
                            mode="llm-agent",
                            model="openai/gpt-5.3-codex",
                            skill_creator_agent_style="cc",
                            skill_creator_system_prompt="Only emit atomic and reusable skills.",
                        ),
                    )

                    kwargs = creator_cls.call_args.kwargs
                    self.assertEqual(kwargs.get("agent_style"), "cc")
                    self.assertEqual(
                        kwargs.get("custom_system_prompt"),
                        "Only emit atomic and reusable skills.",
                    )

    def test_standardized_llm_materialize_is_strict_by_default(self) -> None:
        with tempfile.TemporaryDirectory(prefix="materialize-standardized-strict-") as temp_dir:
            with patch("procmem2skills.packager.materialize._has_any_api_key", return_value=False):
                with self.assertRaises(RuntimeError):
                    materialize_skill_repository_standard_llm(
                        skills=[self._build_skill()],
                        output_dir=Path(temp_dir) / "skills",
                        model="openai/gpt-5.3-codex",
                    )


if __name__ == "__main__":
    unittest.main()
