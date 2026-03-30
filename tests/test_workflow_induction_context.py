from __future__ import annotations

import unittest

from procmem2skills.adapters.terminal_bench import normalize_terminal_bench_step
from procmem2skills.inducer.workflow import induce_workflow, render_workflow_context
from procmem2skills.models import BenchmarkKind, Trajectory
from procmem2skills.segmenter.heuristics import segment_trajectory


class WorkflowInductionContextTest(unittest.TestCase):
    def _build_terminal_trajectory(self) -> Trajectory:
        return Trajectory(
            episode_id="tb-context-1",
            benchmark=BenchmarkKind.TERMINAL_BENCH,
            harness="terminal-bench/harness",
            agent="codex",
            task_id="fix-ci",
            instruction="Investigate and fix the failing test pipeline.",
            events=[
                normalize_terminal_bench_step(
                    {
                        "summary": "Inspect repository root",
                        "cwd": "/app",
                        "command": "ls -la /app",
                        "stdout": "README.md\ntests\n",
                        "stderr": "",
                        "ok": True,
                        "exit_code": 0,
                        "thought": "Inspect the project layout and test directory first.",
                        "state_delta": {"pwd": "/app", "files_touched": []},
                        "success_signal": "Repository layout captured",
                    },
                    1,
                ),
                normalize_terminal_bench_step(
                    {
                        "summary": "Run test suite and capture failure",
                        "cwd": "/app",
                        "command": "pytest -q",
                        "stdout": "1 failed, 3 passed\n",
                        "stderr": "AssertionError: expected 4",
                        "ok": False,
                        "exit_code": 1,
                        "thought": "Reproduce the failure and capture the error message.",
                        "state_delta": {"failed_tests": ["tests/test_calc.py::test_sum"]},
                    },
                    2,
                ),
            ],
        )

    def test_induce_workflow_preserves_event_trace_and_context_payload(self) -> None:
        trajectory = self._build_terminal_trajectory()
        segments = segment_trajectory(trajectory)
        workflow = induce_workflow(segments[0])

        metadata = workflow.metadata
        event_trace = metadata.get("event_trace")
        event_trace_full = metadata.get("event_trace_full")
        context_payload = metadata.get("context_payload")
        coverage = metadata.get("information_coverage")
        cluster_reservation = metadata.get("cluster_reservation")

        self.assertIsInstance(event_trace, list)
        self.assertIsInstance(event_trace_full, list)
        self.assertEqual(len(event_trace), 1)
        self.assertEqual(len(event_trace_full), 1)
        self.assertIsInstance(context_payload, dict)
        self.assertIsInstance(coverage, dict)
        self.assertIsInstance(cluster_reservation, dict)

        first_event = event_trace[0]
        first_event_full = event_trace_full[0]
        action = first_event.get("action") or {}
        result = first_event.get("result") or {}
        full_action = first_event_full.get("action") or {}
        full_result = first_event_full.get("result") or {}

        self.assertEqual(action.get("command"), "ls -la /app")
        self.assertEqual(action.get("raw"), "ls -la /app")
        self.assertEqual(result.get("exit_code"), 0)
        self.assertEqual(full_action.get("command"), "ls -la /app")
        self.assertEqual(full_result.get("output_text"), "README.md\ntests\n")

        self.assertEqual(coverage.get("total_event_count"), 1)
        self.assertEqual(coverage.get("action_event_count"), 1)
        self.assertEqual(coverage.get("non_action_event_count"), 0)

        steps = context_payload.get("steps") or []
        timeline = context_payload.get("timeline") or []
        self.assertEqual(len(steps), 1)
        self.assertEqual(len(timeline), 1)
        self.assertIn("ls -la /app", steps[0].get("command", ""))
        self.assertEqual(steps[0].get("cwd"), "/app")
        self.assertIn("ls -la /app", timeline[0].get("command", ""))
        self.assertIn("step_bigrams", cluster_reservation)

    def test_render_workflow_context_returns_structured_prompt_block(self) -> None:
        trajectory = self._build_terminal_trajectory()
        first_segment = segment_trajectory(trajectory)[0]
        workflow = induce_workflow(first_segment)

        context_text = render_workflow_context(workflow)

        self.assertIn("Objective:", context_text)
        self.assertIn("Trigger:", context_text)
        self.assertIn("Preconditions:", context_text)
        self.assertIn("Steps:", context_text)
        self.assertIn("Verification:", context_text)
        self.assertIn("Failure Signals:", context_text)
        self.assertIn("ls -la /app", context_text)


if __name__ == "__main__":
    unittest.main()
