from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> dict:
    """10-skill corpus with 2D embeddings for fast unit tests."""
    metadata = [
        {"id": f"skill_{i:06d}", "slug": f"author/skill-{i}", "name": f"name-{i}", "category": "general"}
        for i in range(10)
    ]
    embeddings = np.array(
        [[np.cos(i * 0.6), np.sin(i * 0.6)] for i in range(10)],
        dtype=np.float32,
    )
    meta_path = tmp_path / "metadata.jsonl"
    emb_path = tmp_path / "embeddings.npy"
    with meta_path.open("w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")
    np.save(emb_path, embeddings)
    return {"metadata_path": meta_path, "embeddings_path": emb_path, "metadata": metadata, "embeddings": embeddings}


@pytest.fixture
def tiny_task() -> dict:
    return {
        "task_id": "sb_test",
        "task_name": "test-task",
        "domain": "test",
        "instruction": "Test task instruction.",
        "gt_skills": [
            {"skill_id": "gt_test_alpha", "name": "alpha", "dir_name": "alpha", "content": "---\nname: alpha\ndescription: Test alpha skill.\n---\nBody."},
        ],
        "n_gt_skills": 1,
    }
