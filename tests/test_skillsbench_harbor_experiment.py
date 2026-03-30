from __future__ import annotations

import unittest
from pathlib import Path

from procmem2skills.integrations.harbor_skillsbench_experiment import (
    _build_harbor_command,
    _resolve_memory_setting,
)
from procmem2skills.integrations.harbor_terminal_experiment import render_command


class SkillsBenchHarborExperimentTest(unittest.TestCase):
    def test_resolve_memory_setting_supports_alias_and_legacy_agent_mode(self) -> None:
        self.assertEqual(
            _resolve_memory_setting(requested="workflow-context", legacy_agent_mode="native"),
            "workflows",
        )
        self.assertEqual(
            _resolve_memory_setting(requested=None, legacy_agent_mode="skill-aware"),
            "skills",
        )
        self.assertEqual(
            _resolve_memory_setting(requested=None, legacy_agent_mode="native"),
            "none",
        )

    def test_build_harbor_command_for_workflow_setting_uses_custom_agent_and_workflow_kwargs(self) -> None:
        command = _build_harbor_command(
            harbor_bin=Path("/tmp/.venv/bin/harbor"),
            jobs_dir=Path("/tmp/jobs"),
            job_name="sb-workflow-memory",
            source_mode="dataset",
            dataset="skillsbench",
            path=None,
            model="openai/gpt-5.3-codex",
            memory_setting="workflows",
            native_agent="codex",
            agent_import_path="procmem2skills.integrations.harbor_terminal_agent:SkillAwareTerminalAgent",
            skill_repository=None,
            top_k_skills=3,
            skill_selection_mode="agent-first",
            skill_candidate_pool=12,
            workflow_memory_path=Path("/tmp/workflows"),
            workflow_max_attempts=5,
            workflow_max_workflows_per_attempt=20,
            workflow_max_steps_per_workflow=0,
            n_concurrent=5,
            n_attempts=5,
            task_names=["foo-task"],
            exclude_task_names=[],
            n_tasks=None,
            environment_type="docker",
            max_steps=20,
            command_timeout_sec=180,
            base_url="https://openrouter.ai/api/v1",
            working_dir="/workspace",
            passthrough_args=["--max-tokens", "4096"],
        )
        rendered = render_command(command)
        self.assertIn("--agent-import-path", rendered)
        self.assertIn("--ak memory_setting=workflows", rendered)
        self.assertIn("--ak workflow_memory_path=/tmp/workflows", rendered)
        self.assertIn("--ak workflow_max_attempts=5", rendered)
        self.assertIn("--ak workflow_max_workflows_per_attempt=20", rendered)
        self.assertNotIn("skill_repository=", rendered)


if __name__ == "__main__":
    unittest.main()

