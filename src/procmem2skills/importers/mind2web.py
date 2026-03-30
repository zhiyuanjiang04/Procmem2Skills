from __future__ import annotations

from pathlib import Path

from procmem2skills.adapters.mind2web import normalize_mind2web_step
from procmem2skills.importers.common import ensure_list, first_present, load_records
from procmem2skills.models import BenchmarkKind, ExecutionMode, Trajectory


def import_mind2web(path: Path, agent: str = "human-demonstrator", harness: str = "mind2web/replay") -> list[Trajectory]:
    trajectories = []
    for record in load_records(path):
        actions = ensure_list(first_present(record, "actions", "action_reprs", default=[]))
        if not actions:
            continue
        episode_id = str(first_present(record, "annotation_id", "episode_id", "task_id", default=f"mind2web-{len(trajectories)+1}"))
        instruction = str(first_present(record, "confirmed_task", "instruction", "task", default="Mind2Web task"))
        task_id = str(first_present(record, "task_id", "annotation_id", default=episode_id))
        metadata = {
            "website": first_present(record, "website"),
            "domain": first_present(record, "domain"),
            "subdomain": first_present(record, "subdomain"),
        }
        events = []
        for index, action in enumerate(actions, start=1):
            if not isinstance(action, dict):
                action = {"operation": action, "action_repr": str(action)}
            if "website" not in action and metadata["website"] is not None:
                action["website"] = metadata["website"]
            events.append(normalize_mind2web_step(action, index))
        trajectories.append(
            Trajectory(
                episode_id=episode_id,
                benchmark=BenchmarkKind.MIND2WEB,
                harness=harness,
                agent=agent,
                task_id=task_id,
                instruction=instruction,
                mode=ExecutionMode.OFFLINE_BOOTSTRAP,
                metadata={key: value for key, value in metadata.items() if value is not None},
                events=events,
                completed=True,
                score=1.0,
            )
        )
    return trajectories
