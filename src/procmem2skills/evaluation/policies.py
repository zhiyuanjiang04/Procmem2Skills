from __future__ import annotations

from procmem2skills.evaluation.runner import PolicyDecision
from procmem2skills.models import Action


class SkillFirstTerminalPolicy:
    def choose_action(self, *, task, observation, retrieved_skills, step_index, trajectory_so_far):
        command = "ls"
        thought = "Fallback to listing files because no matching terminal skill was retrieved."
        for bundle in retrieved_skills:
            if _base_skill_id(bundle.skill_id) == "terminal-pytest":
                command = "pytest -q"
                thought = "Use terminal-pytest because the task asks to inspect a failure."
                break
        return PolicyDecision(
            action=Action(tool="terminal", name=command.split(" ", 1)[0], arguments={"command": command}, raw=command),
            thought=thought,
            retrieved_skills=[bundle.skill_id for bundle in retrieved_skills],
        )


def _base_skill_id(skill_id: str) -> str:
    lowered = skill_id.strip().lower()
    if lowered.endswith("--success") or lowered.endswith("--failure"):
        return skill_id.rsplit("--", 1)[0]
    return skill_id
