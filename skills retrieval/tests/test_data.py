import json
from pathlib import Path

import numpy as np
import pytest

from skills_retrieval.data import Corpus, Task, load_tasks


def test_corpus_loads_from_paths(tiny_corpus):
    c = Corpus.from_paths(tiny_corpus["metadata_path"], tiny_corpus["embeddings_path"])
    assert len(c.ids) == 10
    assert c.embeddings.shape == (10, 2)
    assert c.ids[0] == "skill_000000"
    assert c.name_by_id["skill_000003"] == "name-3"


def test_corpus_index_lookup(tiny_corpus):
    c = Corpus.from_paths(tiny_corpus["metadata_path"], tiny_corpus["embeddings_path"])
    assert c.index_of("skill_000005") == 5


def test_load_tasks(tmp_path, tiny_task):
    p = tmp_path / "tasks.jsonl"
    with p.open("w") as f:
        f.write(json.dumps(tiny_task) + "\n")
    tasks = load_tasks(p)
    assert len(tasks) == 1
    assert tasks[0].task_id == "sb_test"
    assert tasks[0].gt_skill_names == ["alpha"]


def test_corpus_loads_descriptions(tmp_path, tiny_corpus):
    """descriptions_path is merged by ID; missing IDs get empty string."""
    desc_path = tmp_path / "descs.jsonl"
    # skill_000003 has a description; skill_000000 does not appear → should be ""
    with desc_path.open("w") as f:
        f.write(json.dumps({"id": "skill_000003", "description": "fancy desc"}) + "\n")
    c = Corpus.from_paths(tiny_corpus["metadata_path"], tiny_corpus["embeddings_path"], descriptions_path=desc_path)
    assert c.descriptions[3] == "fancy desc"
    assert c.descriptions[0] == ""
