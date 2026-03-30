from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from procmem2skills.adapters.terminal_bench import normalize_terminal_bench_step
from procmem2skills.inducer.workflow_export import (
    WorkflowAttemptStatus,
    _merge_llm_with_rule_workflow,
    classify_trajectory_status,
    export_grouped_workflows_json,
    induce_workflows_grouped_by_task,
)
from procmem2skills.models import BenchmarkKind, Trajectory, WorkflowCandidate, WorkflowStep


class WorkflowExportTest(unittest.TestCase):
    def _build_trajectory(
        self,
        *,
        episode_id: str,
        task_id: str,
        score: float | None,
        completed: bool,
        ok: bool,
        metadata: dict | None = None,
    ) -> Trajectory:
        event = normalize_terminal_bench_step(
            {
                "summary": "Run command",
                "cwd": "/app",
                "command": "pytest -q",
                "stdout": "ok" if ok else "1 failed",
                "stderr": "" if ok else "AssertionError",
                "ok": ok,
                "exit_code": 0 if ok else 1,
            },
            1,
        )
        return Trajectory(
            episode_id=episode_id,
            benchmark=BenchmarkKind.TERMINAL_BENCH,
            harness="terminal-bench/harness",
            agent="codex",
            task_id=task_id,
            instruction="Fix failing tests.",
            events=[event],
            completed=completed,
            score=score,
            metadata=metadata or {},
        )

    def test_classify_trajectory_status_success_failure_error(self) -> None:
        success = self._build_trajectory(
            episode_id="ep-success",
            task_id="task-a",
            score=1.0,
            completed=True,
            ok=True,
        )
        failure = self._build_trajectory(
            episode_id="ep-failure",
            task_id="task-a",
            score=0.0,
            completed=True,
            ok=False,
        )
        error = Trajectory(
            episode_id="ep-error",
            benchmark=BenchmarkKind.TERMINAL_BENCH,
            harness="terminal-bench/harness",
            agent="codex",
            task_id="task-a",
            instruction="Fix failing tests.",
            events=[],
            completed=False,
            score=None,
            metadata={"error": "runtime crashed"},
        )

        self.assertEqual(classify_trajectory_status(success), WorkflowAttemptStatus.SUCCESS)
        self.assertEqual(classify_trajectory_status(failure), WorkflowAttemptStatus.FAILURE)
        self.assertEqual(classify_trajectory_status(error), WorkflowAttemptStatus.ERROR)

    def test_induce_workflows_grouped_by_task_discards_errors(self) -> None:
        trajectories = [
            self._build_trajectory(
                episode_id="ep-a-2",
                task_id="task-a",
                score=0.0,
                completed=True,
                ok=False,
            ),
            self._build_trajectory(
                episode_id="ep-a-1",
                task_id="task-a",
                score=1.0,
                completed=True,
                ok=True,
            ),
            self._build_trajectory(
                episode_id="ep-b-1",
                task_id="task-b",
                score=1.0,
                completed=True,
                ok=True,
            ),
            Trajectory(
                episode_id="ep-error",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                harness="terminal-bench/harness",
                agent="codex",
                task_id="task-a",
                instruction="Fix failing tests.",
                events=[],
                completed=False,
                score=None,
                metadata={"error": "timeout"},
            ),
        ]

        grouped, summary = induce_workflows_grouped_by_task(trajectories)

        self.assertEqual(set(grouped.keys()), {"task-a", "task-b"})
        self.assertEqual(len(grouped["task-a"]), 2)
        self.assertEqual(len(grouped["task-b"]), 1)
        self.assertEqual(grouped["task-a"][0]["episode_id"], "ep-a-1")
        self.assertEqual(grouped["task-a"][0]["status"], "success")
        self.assertEqual(grouped["task-a"][0]["attempt_index"], 1)
        self.assertEqual(grouped["task-a"][1]["episode_id"], "ep-a-2")
        self.assertEqual(grouped["task-a"][1]["status"], "failure")
        self.assertEqual(grouped["task-a"][1]["attempt_index"], 2)
        self.assertTrue(grouped["task-a"][0]["workflows"])
        self.assertEqual(summary["success"], 2)
        self.assertEqual(summary["failure"], 1)
        self.assertEqual(summary["error_discarded"], 1)

    def test_export_grouped_workflows_json_writes_task_keyed_payload(self) -> None:
        trajectories = [
            self._build_trajectory(
                episode_id="ep-a-1",
                task_id="task-a",
                score=1.0,
                completed=True,
                ok=True,
            ),
            self._build_trajectory(
                episode_id="ep-a-2",
                task_id="task-a",
                score=0.0,
                completed=True,
                ok=False,
            ),
        ]

        with tempfile.TemporaryDirectory(prefix="workflow-export-test-") as temp_dir:
            output_path = Path(temp_dir) / "grouped-workflows.json"
            summary = export_grouped_workflows_json(trajectories, output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(set(payload.keys()), {"task-a"})
        self.assertEqual(len(payload["task-a"]), 2)
        self.assertEqual(payload["task-a"][0]["status"], "success")
        self.assertEqual(payload["task-a"][1]["status"], "failure")
        self.assertEqual(summary["retained_attempts"], 2)

    def test_export_grouped_workflows_json_periodic_checkpoint(self) -> None:
        trajectories = [
            self._build_trajectory(
                episode_id="ep-a-1",
                task_id="task-a",
                score=1.0,
                completed=True,
                ok=True,
            ),
            self._build_trajectory(
                episode_id="ep-a-2",
                task_id="task-a",
                score=0.0,
                completed=True,
                ok=False,
            ),
        ]

        with tempfile.TemporaryDirectory(prefix="workflow-export-checkpoint-") as temp_dir:
            output_path = Path(temp_dir) / "workflows"
            summary = export_grouped_workflows_json(
                trajectories,
                output_path,
                checkpoint_every=1,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(set(payload.keys()), {"task-a"})
        self.assertEqual(len(payload["task-a"]), 2)
        self.assertEqual(summary.get("checkpoint_every"), 1)
        self.assertEqual(summary.get("checkpoint_writes"), 2)

    def test_hybrid_without_api_key_degrades_to_rule_mode(self) -> None:
        trajectories = [
            self._build_trajectory(
                episode_id="ep-a-1",
                task_id="task-a",
                score=1.0,
                completed=True,
                ok=True,
            )
        ]

        grouped, summary = induce_workflows_grouped_by_task(
            trajectories,
            induction_mode="hybrid",
            llm_api_key="",
        )

        self.assertIn("task-a", grouped)
        self.assertEqual(summary["requested_induction_mode"], "hybrid")
        self.assertEqual(summary["induction_mode"], "rule")
        self.assertTrue(summary.get("mode_degraded"))
        self.assertIn("mode_degraded_reason", summary)

    def test_hybrid_merge_keeps_rule_and_llm_preconditions(self) -> None:
        llm_workflow = WorkflowCandidate(
            workflow_id="wf-llm",
            source_segment_id="seg-1",
            objective="Fix failing test",
            trigger="When tests fail",
            preconditions=["repo is available"],
            steps=[WorkflowStep(order=1, intent="run tests", tool="terminal", operation="pytest -q")],
            verification=["all tests pass"],
            failure_modes=[],
            fingerprint="llmfp",
            metadata={"llm": True},
        )
        rule_workflow = WorkflowCandidate(
            workflow_id="wf-rule",
            source_segment_id="seg-1",
            objective="Fix failing test",
            trigger="When tests fail",
            preconditions=["failing test is reproducible"],
            steps=[WorkflowStep(order=1, intent="run tests", tool="terminal", operation="pytest -q")],
            verification=["captured stacktrace"],
            failure_modes=["still failing"],
            fingerprint="rulefp",
            metadata={"rule": True},
        )

        merged = _merge_llm_with_rule_workflow(llm_workflow=llm_workflow, rule_workflow=rule_workflow)

        self.assertEqual(merged.preconditions, ["repo is available", "failing test is reproducible"])
        self.assertIn("all tests pass", merged.verification)
        self.assertIn("captured stacktrace", merged.verification)


if __name__ == "__main__":
    unittest.main()
