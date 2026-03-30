from __future__ import annotations

import unittest

from procmem2skills.runtime.workflow_memory import WorkflowMemoryIndex, normalize_task_key


def _sample_workflow_payload() -> dict:
    return {
        "workflow_id": "episode-1-seg-1-wf",
        "source_segment_id": "episode-1-seg-1",
        "objective": "Fix the failing unit test.",
        "trigger": "When test output shows one failing assertion.",
        "preconditions": ["Repository is available.", "pytest is installed."],
        "steps": [
            {
                "order": 1,
                "intent": "Run tests to reproduce the failure.",
                "tool": "terminal",
                "operation": "bash_command(command=pytest -q)",
                "preconditions": ["Current directory is repository root."],
                "verification": "output includes failing test name",
            }
        ],
        "verification": ["Failing test is reproduced."],
        "failure_modes": ["pytest command not found"],
        "fingerprint": "workflow-fingerprint-1",
        "metadata": {},
    }


class WorkflowMemoryRuntimeTest(unittest.TestCase):
    def test_normalize_task_key_strips_trial_suffix(self) -> None:
        self.assertEqual(normalize_task_key("foo-task__abc123"), "foo-task")
        self.assertEqual(normalize_task_key("/tmp/tasks/foo-task"), "foo-task")

    def test_render_task_memory_returns_prompt_ready_block(self) -> None:
        index = WorkflowMemoryIndex.from_grouped_attempts(
            {
                "foo-task": [
                    {
                        "attempt_index": 0,
                        "status": "success",
                        "task_id": "foo-task",
                        "episode_id": "foo-task-0",
                        "workflows": [_sample_workflow_payload()],
                    }
                ]
            }
        )

        rendered = index.render_task_memory("foo-task__retryA")
        self.assertEqual(rendered.resolved_task_key, "foo-task")
        self.assertEqual(rendered.attempt_count, 1)
        self.assertEqual(rendered.workflow_count, 1)
        self.assertIn("Workflow memory below was induced", rendered.text)
        self.assertIn("Attempt 1: status=success", rendered.text)
        self.assertIn("Objective:", rendered.text)
        self.assertIn("Steps:", rendered.text)

    def test_render_task_memory_returns_none_when_task_missing(self) -> None:
        index = WorkflowMemoryIndex.from_grouped_attempts({})
        rendered = index.render_task_memory("missing-task")
        self.assertIsNone(rendered.resolved_task_key)
        self.assertEqual(rendered.text, "<none>")


if __name__ == "__main__":
    unittest.main()

