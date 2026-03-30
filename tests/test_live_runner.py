from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from procmem2skills.adapters.mock import MockTerminalAdapter
from procmem2skills.evaluation.pipeline import SkillDistillationPipeline
from procmem2skills.evaluation.runner import LiveRunner, PolicyDecision
from procmem2skills.models import Action
from procmem2skills.packager.skill_writer import SkillWriter
from procmem2skills.recorder.jsonl import load_trajectories


class SkillAwareMockPolicy:
    def choose_action(self, *, task, observation, retrieved_skills, step_index, trajectory_so_far):
        command = "ls"
        thought = "No matching skill found."
        for bundle in retrieved_skills:
            if _base_skill_id(bundle.skill_id) == "terminal-pytest":
                command = "pytest -q"
                thought = "Use the retrieved terminal-pytest skill."
                break
        return PolicyDecision(
            action=Action(tool="terminal", name="pytest", arguments={"command": command}, raw=command),
            thought=thought,
            retrieved_skills=[bundle.skill_id for bundle in retrieved_skills],
        )


def _base_skill_id(skill_id: str) -> str:
    lowered = skill_id.strip().lower()
    if lowered.endswith("--success") or lowered.endswith("--failure"):
        return skill_id.rsplit("--", 1)[0]
    return skill_id


class LiveRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="procmem2skills-live-"))
        sample_trajectories = load_trajectories(self.repo_root / "examples/sample-trajectories.jsonl")
        result = SkillDistillationPipeline(min_support=1).distill(sample_trajectories)
        self.skill_repo = self.temp_dir / "skills"
        SkillWriter().write_repository(result.skills, self.skill_repo)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_live_runner_retrieves_skill_and_records_episode(self) -> None:
        runner = LiveRunner(skill_repository=self.skill_repo, top_k_skills=3, max_steps=3)
        result = runner.run(MockTerminalAdapter(), SkillAwareMockPolicy(), episode_id="mock-episode")

        self.assertTrue(result.trajectory.completed)
        self.assertEqual(result.trajectory.score, 1.0)
        self.assertEqual(len(result.trajectory.events), 1)
        self.assertEqual(result.trajectory.events[0].action.arguments["command"], "pytest -q")
        retrieved_skill_ids = [hit.skill_id for hit in result.retrieved_hits[0]]
        self.assertIn("terminal-pytest", [_base_skill_id(skill_id) for skill_id in retrieved_skill_ids])


if __name__ == "__main__":
    unittest.main()
