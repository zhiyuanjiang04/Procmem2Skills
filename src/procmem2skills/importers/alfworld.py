from __future__ import annotations

from pathlib import Path

from procmem2skills.adapters.alfworld import normalize_alfworld_step
from procmem2skills.importers.common import ensure_list, first_present, load_records
from procmem2skills.models import BenchmarkKind, ExecutionMode, Trajectory


def import_alfworld(path: Path, agent: str = "agent", harness: str = "alfworld/textworld") -> list[Trajectory]:
    trajectories = []
    for record in load_records(path):
        steps = ensure_list(first_present(record, "steps", "trajectory", "events", default=[]))
        if not steps:
            continue
        episode_id = str(first_present(record, "episode_id", "game_id", "task_id", default=f"alfworld-{len(trajectories)+1}"))
        instruction = str(first_present(record, "instruction", "task_desc", "goal", default="ALFWorld task"))
        task_id = str(first_present(record, "task_id", "episode_id", default=episode_id))
        events = [normalize_alfworld_step(step, index) for index, step in enumerate(steps, start=1)]
        success = bool(first_present(record, "success", default=True))
        score = first_present(record, "score", "reward", default=1.0 if success else 0.0)
        trajectories.append(
            Trajectory(
                episode_id=episode_id,
                benchmark=BenchmarkKind.ALFWORLD,
                harness=harness,
                agent=str(first_present(record, "agent", default=agent)),
                task_id=task_id,
                instruction=instruction,
                mode=ExecutionMode.OFFLINE_BOOTSTRAP,
                metadata={key: value for key, value in record.items() if key in {"task_type", "split", "scene"}},
                events=events,
                completed=True,
                score=float(score) if score is not None else None,
            )
        )
    return trajectories
