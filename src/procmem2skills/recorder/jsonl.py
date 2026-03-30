from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from procmem2skills.models import Trajectory


def append_trajectory(path: Path, trajectory: Trajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(trajectory.model_dump_json())
        handle.write("\n")


def write_trajectories(path: Path, trajectories: Iterable[Trajectory]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(trajectory.model_dump_json())
            handle.write("\n")


def load_trajectories(path: Path) -> list[Trajectory]:
    if not path.exists():
        return []
    trajectories = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            trajectories.append(Trajectory.model_validate(json.loads(line)))
    return trajectories
