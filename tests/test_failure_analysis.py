from __future__ import annotations

import unittest

from procmem2skills.analysis.failure import (
    build_failure_analysis_from_trajectories,
    extract_failure_signals_from_text,
)
from procmem2skills.models import (
    Action,
    BenchmarkKind,
    Event,
    ExecutionResult,
    Observation,
    Trajectory,
)


class FailureAnalysisTest(unittest.TestCase):
    def test_extract_failure_signals_from_text_captures_pytest_patterns(self) -> None:
        text = """
FAILED ../tests/test_outputs.py::test_ccomplexity - AttributeError: module 'numpy' has no attribute 'int'
E   AttributeError: module 'numpy' has no attribute 'int'
"""
        signals = extract_failure_signals_from_text(text)
        self.assertTrue(any(signal.startswith("failed-test:") for signal in signals))
        self.assertIn("AttributeError: module 'numpy' has no attribute 'int'", signals)

    def test_build_failure_analysis_from_trajectories_groups_by_task(self) -> None:
        success = Trajectory(
            episode_id="ok-1",
            benchmark=BenchmarkKind.TERMINAL_BENCH,
            harness="terminal-bench/harness",
            agent="agent",
            task_id="build-cython-ext",
            instruction="Build extension",
            completed=True,
            score=1.0,
            events=[
                Event(
                    step_id=1,
                    observation=Observation(summary="run build"),
                    action=Action(tool="terminal", name="bash", arguments={"command": "python setup.py build_ext --inplace"}),
                    result=ExecutionResult(ok=True, output_text="done"),
                )
            ],
        )
        failure = Trajectory(
            episode_id="fail-1",
            benchmark=BenchmarkKind.TERMINAL_BENCH,
            harness="terminal-bench/harness",
            agent="agent",
            task_id="build-cython-ext",
            instruction="Build extension",
            completed=True,
            score=0.0,
            events=[
                Event(
                    step_id=1,
                    observation=Observation(summary="run tests"),
                    action=Action(tool="terminal", name="bash", arguments={"command": "pytest -q"}),
                    result=ExecutionResult(
                        ok=False,
                        output_text=(
                            "FAILED ../tests/test_outputs.py::test_ccomplexity\n"
                            "E   AttributeError: module 'numpy' has no attribute 'int'\n"
                        ),
                    ),
                )
            ],
        )

        report = build_failure_analysis_from_trajectories([success, failure])
        task_report = report["by_task"]["build-cython-ext"]

        self.assertEqual(task_report["attempts"], 2)
        self.assertEqual(task_report["successes"], 1)
        self.assertEqual(task_report["failures"], 1)
        signatures = [item["signature"] for item in task_report["failure_signals"]]
        self.assertIn("AttributeError: module 'numpy' has no attribute 'int'", signatures)


if __name__ == "__main__":
    unittest.main()
