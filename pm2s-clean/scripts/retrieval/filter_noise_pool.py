#!/usr/bin/env python3
"""Filter ClawHub noise skills into a retrieval-ready JSONL manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_ROOT, contains_cjk, read_json, skill_description, skill_name_from_md, write_json, write_jsonl


def default_noise_pool(root: Path) -> Path:
    nested = root / "skill_retrieval" / "noise_pool"
    if nested.is_dir():
        return nested
    return root / "noise_pool"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a filtered noise skill manifest.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--noise-pool", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--english-only", action="store_true")
    parser.add_argument("--min-description-chars", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    noise_pool = (args.noise_pool or default_noise_pool(root)).resolve()
    output = args.output or (root / "manifests" / "noise_skills.filtered.jsonl")

    rows: list[dict] = []
    skipped = {"dotfile": 0, "missing_skill": 0, "missing_metadata": 0, "short_description": 0, "cjk": 0}
    for skill_dir in sorted(p for p in noise_pool.iterdir() if p.is_dir()):
        if skill_dir.name.startswith("._") or skill_dir.name.startswith("."):
            skipped["dotfile"] += 1
            continue
        skill_md = skill_dir / "SKILL.md"
        metadata_path = skill_dir / "metadata.json"
        if not skill_md.is_file():
            skipped["missing_skill"] += 1
            continue
        metadata = {}
        if metadata_path.is_file():
            try:
                metadata = read_json(metadata_path)
            except Exception:
                metadata = {}
        else:
            skipped["missing_metadata"] += 1
        desc = skill_description(skill_md, metadata)
        if len(desc.strip()) < args.min_description_chars:
            skipped["short_description"] += 1
            continue
        if args.english_only and contains_cjk(desc):
            skipped["cjk"] += 1
            continue
        rows.append(
            {
                "skill_name": str(metadata.get("skill_name") or skill_name_from_md(skill_md) or skill_dir.name),
                "skill_slug": skill_dir.name,
                "skill_dir": str(skill_dir),
                "skill_md": str(skill_md),
                "description": desc,
                "source": str(metadata.get("source") or "clawhub"),
                "url": str(metadata.get("url") or ""),
                "downloads": metadata.get("downloads"),
                "hash": metadata.get("hash"),
                "version": metadata.get("version"),
            }
        )
        if args.limit and len(rows) >= args.limit:
            break

    count = write_jsonl(output, rows)
    summary_path = output.with_suffix(".summary.json")
    write_json(summary_path, {
        "noise_pool": str(noise_pool),
        "output": str(output),
        "count": count,
        "english_only": bool(args.english_only),
        "min_description_chars": int(args.min_description_chars),
        "skipped": skipped,
    })
    print(f"wrote {count} noise skills to {output}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
