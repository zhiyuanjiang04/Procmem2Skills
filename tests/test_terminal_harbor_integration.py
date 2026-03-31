from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from procmem2skills.integrations.harbor_terminal_experiment import (
    build_harbor_job_name,
    build_harbor_run_command,
    collect_harbor_progress_snapshot,
    dataset_storage_slug,
    ensure_job_dir_alias,
    format_harbor_progress_line,
    normalize_experiment_name,
    resolve_import_benchmark,
    render_command,
)
from procmem2skills.runtime.retrieval import SkillBundle
from procmem2skills.runtime.terminal_agent import (
    TerminalExecutionSnapshot,
    build_terminal_agent_messages,
    build_terminal_query,
    parse_terminal_agent_response,
    render_skill_context,
)


class TerminalAgentHelpersTest(unittest.TestCase):
    def test_parse_terminal_agent_response_from_fenced_json(self) -> None:
        response = """```json
{"thought": "Run tests first", "command": "pytest -q", "done": false}
```"""
        decision = parse_terminal_agent_response(response)
        self.assertEqual(decision.command, "pytest -q")
        self.assertEqual(decision.thought, "Run tests first")
        self.assertFalse(decision.done)

    def test_build_terminal_agent_messages_include_skill_context_and_shell_rule(self) -> None:
        snapshot = TerminalExecutionSnapshot(
            instruction="Fix the failing test.",
            cwd="/workspace/repo",
            stdout="1 failed, 3 passed",
            recent_history=["pytest -q\n1 failed, 3 passed"],
        )
        bundles = [
            SkillBundle(
                skill_id="terminal-pytest",
                body="Run pytest to inspect the exact failure before editing files.",
                references={"notes.md": "Use pytest first."},
                scripts=["triage.sh"],
            )
        ]
        messages = build_terminal_agent_messages(snapshot=snapshot, retrieved_skills=bundles)
        joined = json.dumps(messages, ensure_ascii=False)
        self.assertIn("terminal-pytest", joined)
        self.assertIn("triage.sh", joined)
        self.assertIn("shell does not preserve state", joined)
        self.assertIn("Do not emit XML tags", joined)

    def test_parse_terminal_agent_response_from_xmlish_tool_call(self) -> None:
        response = """I'll inspect the workspace first.

<function_calls>
<invoke name="do_command">
<parameter name="command">cd /app && ls -la</parameter>
</invoke>
</function_calls>
<parameter name="thought">Start with repository inspection.</parameter>
"""
        decision = parse_terminal_agent_response(response)
        self.assertEqual(decision.command, "cd /app && ls -la")
        self.assertEqual(decision.thought, "Start with repository inspection.")
        self.assertFalse(decision.done)

    def test_build_terminal_agent_messages_include_workflow_memories(self) -> None:
        snapshot = TerminalExecutionSnapshot(
            instruction="Fix the failing build.",
            cwd="/workspace/repo",
            stdout="build failed",
            recent_history=["make test\nbuild failed"],
        )
        messages = build_terminal_agent_messages(
            snapshot=snapshot,
            retrieved_skills=[],
            workflow_memories_context="Attempt 1: status=success\n- Workflow 1:\n  Objective: compile project",
        )
        joined = json.dumps(messages, ensure_ascii=False)
        self.assertIn("Workflow Memories:", joined)
        self.assertIn("Attempt 1: status=success", joined)
        self.assertIn("Objective: compile project", joined)

    def test_build_terminal_query_uses_recent_context(self) -> None:
        query = build_terminal_query(
            TerminalExecutionSnapshot(
                instruction="Build the extension and run tests.",
                cwd="/app",
                stdout="gcc failed",
                stderr="missing Python.h",
                recent_history=["python setup.py build_ext --inplace"],
            )
        )
        self.assertIn("Build the extension", query)
        self.assertIn("missing Python.h", query)
        self.assertIn("python setup.py build_ext --inplace", query)

    def test_render_skill_context_lists_bundle_assets(self) -> None:
        rendered = render_skill_context(
            [
                SkillBundle(
                    skill_id="terminal-build",
                    body="Compile the project before rerunning tests.",
                    references={"ref.md": "Build reference"},
                    scripts=["build.sh"],
                )
            ]
        )
        self.assertIn("terminal-build", rendered)
        self.assertIn("build.sh", rendered)
        self.assertIn("ref.md", rendered)


class HarborExperimentHelpersTest(unittest.TestCase):
    def test_normalize_experiment_name_slugifies_user_input(self) -> None:
        self.assertEqual(normalize_experiment_name("TB Full/Parallel Shard0"), "tb-full-parallel-shard0")
        self.assertEqual(normalize_experiment_name("__A__B__"), "a-b")

    def test_build_harbor_job_name_is_structured_and_stable(self) -> None:
        name = build_harbor_job_name(
            experiment_id="tb-full-parallel-shard0-to-openai-gpt-5-1",
            dataset="terminal-bench@2.0",
            model="openai/gpt-5.3-codex",
            phase="weak-with-skills",
        )
        self.assertTrue(name.startswith("tb-weak-with-skills-"))
        self.assertIn("gpt-5-3-codex", name)
        self.assertLessEqual(len(name), 96)

    def test_dataset_storage_slug_includes_version(self) -> None:
        self.assertEqual(dataset_storage_slug("terminal-bench@2.0"), "terminal-bench-2-0")
        self.assertEqual(dataset_storage_slug("terminal-bench-sample@2.0"), "terminal-bench-sample-2-0")

    def test_resolve_import_benchmark_auto_maps_terminal_bench_variants(self) -> None:
        self.assertEqual(
            resolve_import_benchmark(dataset="terminal-bench@2.0", import_benchmark="auto"),
            "terminal-bench",
        )
        self.assertEqual(
            resolve_import_benchmark(dataset="terminal-bench-sample@2.0", import_benchmark="auto"),
            "terminal-bench",
        )

    def test_ensure_job_dir_alias_creates_symlink_alias(self) -> None:
        with tempfile.TemporaryDirectory(prefix="procmem2skills-job-alias-") as temp:
            jobs_dir = Path(temp) / "jobs"
            actual = jobs_dir / "harbor-result-20260324-abc"
            jobs_dir.mkdir(parents=True, exist_ok=True)
            actual.mkdir(parents=True, exist_ok=True)
            alias = ensure_job_dir_alias(
                jobs_dir=jobs_dir,
                alias_name="tb-weak-with-skills-demo",
                actual_job_dir=actual,
            )
            self.assertIsNotNone(alias)
            assert alias is not None
            self.assertTrue(alias.is_symlink())
            self.assertEqual(alias.resolve(), actual.resolve())

    def test_build_harbor_run_command_uses_custom_agent(self) -> None:
        command = build_harbor_run_command(
            harbor_bin=Path("/tmp/.venv/bin/harbor"),
            jobs_dir=Path("/tmp/jobs"),
            job_name="tb-skill-live",
            dataset="terminal-bench@2.0",
            model="anthropic/claude-sonnet-4",
            agent_mode="skill-aware",
            native_agent=None,
            agent_import_path="procmem2skills.integrations.harbor_terminal_agent:SkillAwareTerminalAgent",
            skill_repository=Path("/tmp/skills"),
            top_k_skills=4,
            skill_selection_mode="agent-first",
            skill_candidate_pool=10,
            agent_kwargs=None,
            max_steps=12,
            command_timeout_sec=200,
            environment_type="docker",
            n_concurrent=2,
            task_names=["build-cython-ext", "query-optimize"],
            n_tasks=1,
            n_attempts=5,
            base_url="https://openrouter.ai/api/v1",
            working_dir="/workspace/task",
            extra_args=["--max-tokens", "4096"],
        )
        rendered = render_command(command)
        self.assertIn("--agent-import-path", rendered)
        self.assertIn("procmem2skills.integrations.harbor_terminal_agent:SkillAwareTerminalAgent", rendered)
        self.assertIn("skill_repository=/tmp/skills", rendered)
        self.assertIn("--dataset terminal-bench@2.0", rendered)
        self.assertIn("--task-name build-cython-ext", rendered)
        self.assertIn("--task-name query-optimize", rendered)
        self.assertIn("--n-attempts 5", rendered)
        self.assertIn("--ak skill_selection_mode=agent-first", rendered)
        self.assertIn("--ak skill_candidate_pool=10", rendered)
        self.assertIn("--ak base_url=https://openrouter.ai/api/v1", rendered)
        self.assertIn("--max-tokens 4096", rendered)
        self.assertNotIn("OPENROUTER_API_KEY", rendered)

    def test_build_harbor_run_command_supports_native_agent_mode(self) -> None:
        command = build_harbor_run_command(
            harbor_bin=Path("/tmp/.venv/bin/harbor"),
            jobs_dir=Path("/tmp/jobs"),
            job_name="tb-native",
            dataset="terminal-bench@2.0",
            model="openai/gpt-5.3-codex",
            agent_mode="native",
            native_agent="terminus-2",
            agent_import_path="procmem2skills.integrations.harbor_terminal_agent:SkillAwareTerminalAgent",
            skill_repository=Path("/tmp/skills"),
            top_k_skills=4,
            skill_selection_mode="agent-first",
            skill_candidate_pool=10,
            agent_kwargs={"prompt_template_path": "/tmp/native-skill-prompt.j2"},
            max_steps=12,
            command_timeout_sec=200,
            environment_type="docker",
            n_concurrent=1,
            task_names=["build-cython-ext"],
            n_tasks=1,
            n_attempts=1,
            base_url="https://openrouter.ai/api/v1",
            working_dir=None,
        )
        rendered = render_command(command)
        self.assertIn("--agent terminus-2", rendered)
        self.assertNotIn("--agent-import-path", rendered)
        self.assertNotIn("skill_repository=", rendered)
        self.assertIn("--ak prompt_template_path=/tmp/native-skill-prompt.j2", rendered)
        self.assertIn("--ae OPENAI_BASE_URL=https://openrouter.ai/api/v1", rendered)

    def test_collect_harbor_progress_snapshot_counts_results(self) -> None:
        with tempfile.TemporaryDirectory(prefix="procmem2skills-progress-") as temp:
            jobs_dir = Path(temp) / "jobs"
            job_dir = jobs_dir / "tb-live"
            trial_success = job_dir / "task-a__1"
            trial_failure = job_dir / "task-a__2"
            trial_running = job_dir / "task-b__1"
            (trial_success / "agent").mkdir(parents=True, exist_ok=True)
            (trial_failure / "agent").mkdir(parents=True, exist_ok=True)
            (trial_running / "agent").mkdir(parents=True, exist_ok=True)
            (trial_success / "agent" / "trajectory.json").write_text("{}", encoding="utf-8")
            (trial_failure / "agent" / "trajectory.json").write_text("{}", encoding="utf-8")
            (trial_running / "agent" / "trajectory.json").write_text("{}", encoding="utf-8")
            (trial_success / "result.json").write_text(
                json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}}),
                encoding="utf-8",
            )
            (trial_failure / "result.json").write_text(
                json.dumps({"verifier_result": {"rewards": {"reward": 0.0}}}),
                encoding="utf-8",
            )

            snapshot = collect_harbor_progress_snapshot(jobs_dir=jobs_dir, job_name="tb-live")

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["completed"], 2)
        self.assertEqual(snapshot["success"], 1)
        self.assertEqual(snapshot["failure"], 1)
        self.assertEqual(snapshot["trajectory_count"], 3)

    def test_format_harbor_progress_line_contains_eta_when_total_known(self) -> None:
        line = format_harbor_progress_line(
            snapshot={
                "job_name": "tb-live",
                "completed": 2,
                "success": 1,
                "failure": 1,
                "trajectory_count": 3,
            },
            elapsed_sec=20.0,
            expected_trials=4,
        )
        self.assertIn("tb-live", line)
        self.assertIn("2/4", line)
        self.assertIn("50.0%", line)
        self.assertIn("eta", line)


if __name__ == "__main__":
    unittest.main()
