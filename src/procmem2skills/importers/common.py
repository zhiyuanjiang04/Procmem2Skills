from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_records(path: Path) -> list[dict]:
    if path.is_dir():
        records = []
        for child in sorted(path.iterdir()):
            if child.suffix in {".json", ".jsonl"}:
                records.extend(load_records(child))
        return records
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        if "episodes" in data and isinstance(data["episodes"], list):
            return data["episodes"]
        return [data]
    raise TypeError(f"Unsupported record container in {path}")


def first_present(mapping: dict, *keys: str, default=None):
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def ensure_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
