from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from procmem2skills.integrations import harbor_transfer_study as transfer_module
from procmem2skills.integrations.harbor_transfer_study import (
    _resolve_task_names,
    build_failure_guardrails,
    load_failed_trajectories,
    load_success_trajectories,
    load_trial_records,
    main as transfer_main,
    select_transfer_tasks,
    summarize_task_failures,
    summarize_task_success,
)


class HarborTransferStudyHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="procmem2skills-transfer-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_summary_uses_trial_reward_not_completed_flag(self) -> None:
        job_dir = self.temp_dir / "harbor-job"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "result.json").write_text(json.dumps({"id": "job-summary"}), encoding="utf-8")
        self._write_trial(job_dir, trial_name="task-a__ok1", task_name="task-a", reward=1.0, with_exception=False)
        self._write_trial(job_dir, trial_name="task-a__fail1", task_name="task-a", reward=0.0, with_exception=True)

        records = load_trial_records(job_dir)
        self.assertEqual(len(records), 2)
        summary = summarize_task_success(job_dir)
        self.assertEqual(summary["task-a"]["attempts"], 2)
        self.assertEqual(summary["task-a"]["successes"], 1)
        self.assertEqual(summary["task-a"]["errors"], 1)

    def test_load_success_trajectories_filters_reward_and_task(self) -> None:
        job_dir = self.temp_dir / "harbor-job"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "result.json").write_text(json.dumps({"id": "job-summary"}), encoding="utf-8")
        self._write_trial(job_dir, trial_name="task-a__ok1", task_name="task-a", reward=1.0, with_exception=False)
        self._write_trial(job_dir, trial_name="task-b__ok2", task_name="task-b", reward=1.0, with_exception=False)
        self._write_trial(job_dir, trial_name="task-a__fail1", task_name="task-a", reward=0.0, with_exception=False)

        trajectories = load_success_trajectories(job_dir, allowed_tasks={"task-a"})
        self.assertEqual(len(trajectories), 1)
        self.assertEqual(trajectories[0].task_id, "task-a")
        self.assertEqual(trajectories[0].score, 1.0)

    def test_summarize_task_failures_extracts_signatures(self) -> None:
        job_dir = self.temp_dir / "harbor-job"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "result.json").write_text(json.dumps({"id": "job-summary"}), encoding="utf-8")
        self._write_trial(
            job_dir,
            trial_name="build-cython-ext__fail1",
            task_name="build-cython-ext",
            reward=0.0,
            with_exception=False,
            verifier_stdout=(
                "FAILED ../tests/test_outputs.py::test_ccomplexity\n"
                "E   AttributeError: module 'numpy' has no attribute 'int'\n"
            ),
        )

        records = load_trial_records(job_dir)
        failure_report = summarize_task_failures(records, allowed_tasks={"build-cython-ext"})
        task_report = failure_report["by_task"]["build-cython-ext"]

        self.assertEqual(task_report["attempts"], 1)
        self.assertEqual(task_report["failures"], 1)
        signatures = [item["signature"] for item in task_report["failure_signals"]]
        self.assertIn("failed-test:../tests/test_outputs.py::test_ccomplexity", signatures)
        self.assertIn("AttributeError: module 'numpy' has no attribute 'int'", signatures)

    def test_transfer_study_defaults_to_codex_native_modes(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = transfer_main(
                [
                    "--experiment-id",
                    "dryrun-defaults-test",
                    "--strong-model",
                    "openrouter/anthropic/claude-opus-4.5",
                    "--weak-model",
                    "openrouter/anthropic/claude-sonnet-4.5",
                    "--task-name",
                    "build-cython-ext",
                    "--dry-run",
                ]
            )
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["strong_agent_mode"], "native")
        self.assertEqual(payload["weak_agent_mode"], "native")
        self.assertEqual(payload["skill_agent_mode"], "native")
        self.assertIn("--agent codex", payload["strong_command"])
        self.assertIn("--agent codex", payload["weak_command"])

    def test_transfer_study_dry_run_normalizes_experiment_and_phase_job_names(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = transfer_main(
                [
                    "--experiment-id",
                    "TB Full/Parallel Shard0",
                    "--strong-model",
                    "openai/gpt-5.3-codex",
                    "--weak-model",
                    "openai/gpt-5.1",
                    "--task-name",
                    "build-cython-ext",
                    "--dry-run",
                ]
            )
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["requested_experiment_id"], "TB Full/Parallel Shard0")
        self.assertEqual(payload["experiment_id"], "tb-full-parallel-shard0")
        self.assertIn("phase_job_names", payload)
        self.assertTrue(payload["phase_job_names"]["strong_baseline"].startswith("tb-strong-baseline-"))

    def test_transfer_study_dry_run_supports_harbor_passthrough_args(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = transfer_main(
                [
                    "--experiment-id",
                    "dryrun-passthrough",
                    "--strong-model",
                    "openai/gpt-5.3-codex",
                    "--weak-model",
                    "openai/gpt-5.1",
                    "--task-name",
                    "build-cython-ext",
                    "--dry-run",
                    "--max-tokens",
                    "4096",
                ]
            )
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["harbor_passthrough_args"], ["--max-tokens", "4096"])
        self.assertIn("--max-tokens 4096", payload["strong_command"])
        self.assertIn("--max-tokens 4096", payload["weak_command"])

    def test_resolve_task_names_supports_all_tasks_sharding_and_limit(self) -> None:
        with patch.object(
            transfer_module,
            "_load_dataset_task_names",
            return_value=["task-c", "task-a", "task-b", "task-d"],
        ):
            selected = _resolve_task_names(
                explicit_task_names=["task-a", "task-a"],
                all_tasks=True,
                dataset="terminal-bench@2.0",
                task_filter="task-",
                task_limit=2,
                task_shard_count=2,
                task_shard_index=1,
            )
        self.assertEqual(selected, ["task-c", "task-d"])

    def test_resolve_task_names_requires_non_empty_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "no tasks selected"):
            _resolve_task_names(
                explicit_task_names=[],
                all_tasks=False,
                dataset="terminal-bench@2.0",
                task_filter="",
                task_limit=0,
                task_shard_count=1,
                task_shard_index=0,
            )

    def test_select_transfer_tasks_supports_failure_reflection_without_strong_success(self) -> None:
        task_names = ["dna-insert", "regex-chess", "video-processing"]
        strong_summary = {
            "dna-insert": {"attempts": 1, "successes": 0},
            "regex-chess": {"attempts": 1, "successes": 1},
            "video-processing": {"attempts": 1, "successes": 0},
        }
        weak_summary = {
            "dna-insert": {"attempts": 5, "successes": 0},
            "regex-chess": {"attempts": 5, "successes": 0},
            "video-processing": {"attempts": 5, "successes": 2},
        }
        candidate_tasks, success_tasks, failure_only_tasks, plan = select_transfer_tasks(
            task_names=task_names,
            strong_summary=strong_summary,
            weak_summary=weak_summary,
        )
        self.assertEqual(candidate_tasks, ["dna-insert", "regex-chess"])
        self.assertEqual(success_tasks, {"regex-chess"})
        self.assertEqual(failure_only_tasks, {"dna-insert"})
        self.assertEqual(plan["dna-insert"]["strategy"], "failure-reflection")
        self.assertEqual(plan["regex-chess"]["strategy"], "success-transfer")
        self.assertEqual(plan["video-processing"]["strategy"], "skip-weak-has-success")

    def test_build_failure_guardrails_extracts_signatures_and_sequences(self) -> None:
        job_dir = self.temp_dir / "harbor-job"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "result.json").write_text(json.dumps({"id": "job-summary"}), encoding="utf-8")
        self._write_trial(
            job_dir,
            trial_name="dna-insert__fail1",
            task_name="dna-insert",
            reward=0.0,
            with_exception=False,
            verifier_stdout="E AssertionError: marker not found",
        )
        trajectories = load_success_trajectories(job_dir, allowed_tasks={"dna-insert"})
        self.assertEqual(len(trajectories), 0)

        failed = load_failed_trajectories(job_dir, allowed_tasks={"dna-insert"})
        guardrails = build_failure_guardrails(
            failed,
            allowed_tasks={"dna-insert"},
            failure_analysis_by_task={
                "dna-insert": {
                    "failure_signals": [{"signature": "AssertionError: marker not found", "count": 1}],
                }
            },
        )
        self.assertIn("dna-insert", guardrails["by_task"])
        task_guard = guardrails["by_task"]["dna-insert"]
        self.assertGreaterEqual(len(task_guard["failed_command_sequences"]), 1)
        signatures = [item["signature"] for item in task_guard["failure_signatures"]]
        self.assertIn("AssertionError: marker not found", signatures)

    def _write_trial(
        self,
        job_dir: Path,
        *,
        trial_name: str,
        task_name: str,
        reward: float,
        with_exception: bool,
        verifier_stdout: str | None = None,
    ) -> None:
        trial_dir = job_dir / trial_name
        (trial_dir / "agent").mkdir(parents=True, exist_ok=True)
        (trial_dir / "verifier").mkdir(parents=True, exist_ok=True)

        trajectory_payload = {
            "schema_version": "ATIF-v1.6",
            "session_id": f"session-{trial_name}",
            "agent": {"name": "skill-aware", "model_name": "anthropic/claude-sonnet-4.5"},
            "steps": [
                {
                    "source": "user",
                    "message": "Task Description:\nDo the task.\n\nCurrent terminal state:\nroot@demo:/app#",
                },
                {
                    "source": "agent",
                    "message": "Run inspection command first.",
                    "tool_calls": [
                        {
                            "function_name": "bash_command",
                            "arguments": {"keystrokes": "ls -la\n", "duration": 0.1},
                        }
                    ],
                    "observation": {
                        "results": [
                            {
                                "content": "root@demo:/app# ls -la\nREADME.md\nroot@demo:/app#",
                            }
                        ]
                    },
                },
            ],
        }
        config_payload = {
            "task": {"path": task_name},
            "trial_name": trial_name,
        }
        result_payload = {
            "task_name": task_name,
            "trial_name": trial_name,
            "source": "terminal-bench",
            "task_checksum": f"checksum-{trial_name}",
            "verifier_result": {"rewards": {"reward": reward}},
            "exception_info": {"exception_type": "ValueError"} if with_exception else None,
        }

        (trial_dir / "agent" / "trajectory.json").write_text(
            json.dumps(trajectory_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        (trial_dir / "config.json").write_text(json.dumps(config_payload, ensure_ascii=False), encoding="utf-8")
        (trial_dir / "result.json").write_text(json.dumps(result_payload, ensure_ascii=False), encoding="utf-8")
        if verifier_stdout is not None:
            (trial_dir / "verifier" / "test-stdout.txt").write_text(verifier_stdout, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
