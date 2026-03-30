from __future__ import annotations

import unittest

from procmem2skills.research.controlled_workflow_skill_study import (
    WorkflowMixCondition,
    build_atomic_skill_from_selection,
    collect_task_workflow_pools,
    parse_condition_specs,
    sample_workflows_for_condition,
    select_eligible_tasks,
)


class ControlledWorkflowSkillStudyTest(unittest.TestCase):
    def _grouped_payload(self) -> dict:
        def wf(task: str, status: str, index: int) -> dict:
            return {
                "workflow_id": f"{task}-{status}-{index}",
                "objective": f"Objective {task} {status} {index}",
                "trigger": f"Trigger {task}",
                "preconditions": [f"pre-{status}-{index}"],
                "verification": [f"verify-{status}-{index}"],
                "failure_modes": [f"fail-{status}-{index}"],
                "steps": [
                    {
                        "order": 1,
                        "intent": f"intent-{status}-{index}",
                        "tool": "terminal",
                        "operation": f"terminal(command=echo {task}-{status}-{index})",
                        "preconditions": [],
                        "verification": f"check-{status}-{index}",
                    }
                ],
            }

        task_mix_attempts = []
        for i in range(1, 6):
            task_mix_attempts.append(
                {
                    "status": "success",
                    "task_id": "task-mix",
                    "instruction": "Solve task-mix",
                    "workflows": [wf("task-mix", "s", i)],
                }
            )
        for i in range(1, 6):
            task_mix_attempts.append(
                {
                    "status": "failure",
                    "task_id": "task-mix",
                    "instruction": "Solve task-mix",
                    "workflows": [wf("task-mix", "f", i)],
                }
            )

        task_success_only = []
        for i in range(1, 6):
            task_success_only.append(
                {
                    "status": "success",
                    "task_id": "task-success-only",
                    "instruction": "Solve task-success-only",
                    "workflows": [wf("task-success-only", "s", i)],
                }
            )

        task_insufficient = []
        for i in range(1, 6):
            task_insufficient.append(
                {
                    "status": "success",
                    "task_id": "task-insufficient-fail",
                    "instruction": "Solve task-insufficient-fail",
                    "workflows": [wf("task-insufficient-fail", "s", i)],
                }
            )
        task_insufficient.append(
            {
                "status": "failure",
                "task_id": "task-insufficient-fail",
                "instruction": "Solve task-insufficient-fail",
                "workflows": [wf("task-insufficient-fail", "f", 1)],
            }
        )

        return {
            "task-mix": task_mix_attempts,
            "task-success-only": task_success_only,
            "task-insufficient-fail": task_insufficient,
        }

    def test_parse_condition_specs(self) -> None:
        conditions = parse_condition_specs(["5s0f", "4s1f", "3s2f"])
        self.assertEqual(
            [(c.label, c.success_count, c.failure_count) for c in conditions],
            [("5s0f", 5, 0), ("4s1f", 4, 1), ("3s2f", 3, 2)],
        )

    def test_parse_condition_specs_supports_m_n_pair_formats(self) -> None:
        conditions = parse_condition_specs(["5,0", "4:1", "3/2"])
        self.assertEqual(
            [(c.label, c.success_count, c.failure_count) for c in conditions],
            [("5s0f", 5, 0), ("4s1f", 4, 1), ("3s2f", 3, 2)],
        )

    def test_parse_condition_specs_rejects_zero_total(self) -> None:
        with self.assertRaises(ValueError):
            parse_condition_specs(["0,0"])

    def test_select_eligible_tasks_drops_unbalanced_and_insufficient(self) -> None:
        pools = collect_task_workflow_pools(self._grouped_payload())
        conditions = [WorkflowMixCondition(label="4s1f", success_count=4, failure_count=1)]
        eligible = select_eligible_tasks(
            pools,
            conditions,
            require_success=True,
            require_failure_for_mixed=True,
            require_counts_for_all_conditions=True,
            minimum_mixed_failure_count=2,
        )
        self.assertEqual(eligible, ["task-mix"])

    def test_select_eligible_tasks_can_enforce_fixed_pool_requirements(self) -> None:
        pools = collect_task_workflow_pools(self._grouped_payload())
        conditions = [WorkflowMixCondition(label="4s1f", success_count=4, failure_count=1)]
        eligible = select_eligible_tasks(
            pools,
            conditions,
            require_success=True,
            require_failure_for_mixed=True,
            require_counts_for_all_conditions=True,
            minimum_success_pool_size=5,
            minimum_failure_pool_size=5,
        )
        self.assertEqual(eligible, ["task-mix"])

    def test_sample_workflows_for_condition_respects_mix_counts(self) -> None:
        pools = collect_task_workflow_pools(self._grouped_payload())
        condition = WorkflowMixCondition(label="3s2f", success_count=3, failure_count=2)
        selection = sample_workflows_for_condition(
            pools["task-mix"],
            condition=condition,
            random_seed=7,
        )
        self.assertEqual(len(selection.success_workflows), 3)
        self.assertEqual(len(selection.failure_workflows), 2)
        success_ids = {item.workflow.workflow_id for item in selection.success_workflows}
        failure_ids = {item.workflow.workflow_id for item in selection.failure_workflows}
        self.assertEqual(len(success_ids), 3)
        self.assertEqual(len(failure_ids), 2)
        self.assertTrue(all("-s-" in skill_id for skill_id in success_ids))
        self.assertTrue(all("-f-" in skill_id for skill_id in failure_ids))

    def test_build_atomic_skill_from_selection_keeps_provenance(self) -> None:
        pools = collect_task_workflow_pools(self._grouped_payload())
        condition = WorkflowMixCondition(label="4s1f", success_count=4, failure_count=1)
        selection = sample_workflows_for_condition(
            pools["task-mix"],
            condition=condition,
            random_seed=11,
        )
        skill = build_atomic_skill_from_selection(
            task_id="task-mix",
            selection=selection,
            skill_namespace="controlled",
        )
        self.assertEqual(skill.task_origins, ["task-mix"])
        self.assertEqual(skill.support, 5)
        self.assertEqual(len(skill.source_workflow_ids), 5)
        self.assertEqual(
            skill.metadata.get("workflow_mix"),
            {"success": 4, "failure": 1, "label": "4s1f"},
        )


if __name__ == "__main__":
    unittest.main()
