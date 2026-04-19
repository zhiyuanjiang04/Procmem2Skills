from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Corpus:
    ids: list[str]
    names: list[str]
    descriptions: list[str]
    embeddings: np.ndarray                           # (N, D) float32
    name_by_id: dict[str, str] = field(default_factory=dict)
    _idx_by_id: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_paths(cls, metadata_path: Path, embeddings_path: Path, descriptions_path: Path | None = None) -> "Corpus":
        ids: list[str] = []
        names: list[str] = []
        with Path(metadata_path).open() as f:
            for line in f:
                row = json.loads(line)
                ids.append(row["id"])
                names.append(row.get("name", row["id"]))
        embeddings = np.load(embeddings_path).astype(np.float32)
        if embeddings.shape[0] != len(ids):
            raise ValueError(f"Embeddings ({embeddings.shape[0]}) != metadata ({len(ids)})")
        descriptions = [""] * len(ids)
        if descriptions_path is not None:
            with Path(descriptions_path).open() as f:
                by_id = {json.loads(line)["id"]: json.loads(line).get("description", "") for line in f}
            descriptions = [by_id.get(i, "") for i in ids]
        name_by_id = dict(zip(ids, names))
        _idx_by_id = {i: k for k, i in enumerate(ids)}
        return cls(ids=ids, names=names, descriptions=descriptions, embeddings=embeddings, name_by_id=name_by_id, _idx_by_id=_idx_by_id)

    def index_of(self, skill_id: str) -> int:
        return self._idx_by_id[skill_id]

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class Task:
    task_id: str
    instruction: str
    gt_skill_names: list[str]
    gt_skill_bodies: list[str]
    domain: str = ""

    @property
    def gt_set(self) -> set[str]:
        return set(self.gt_skill_names)


def load_tasks(path: Path) -> list[Task]:
    tasks: list[Task] = []
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)
            gts = row.get("gt_skills") or []
            tasks.append(
                Task(
                    task_id=row["task_id"],
                    instruction=row["instruction"],
                    gt_skill_names=[g["name"] for g in gts],
                    gt_skill_bodies=[g.get("content", "") for g in gts],
                    domain=row.get("domain", ""),
                )
            )
    return tasks
