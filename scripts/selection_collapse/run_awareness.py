#!/usr/bin/env python3
"""Stage 1: Awareness probe — ask the model which skills are relevant.

For each pool file in data/selection_collapse/pools/, sends the awareness
prompt to Claude Opus and parses which SKILL_XXX IDs the model lists.

Output: one JSON file per probe in data/selection_collapse/results/awareness/
"""

import json
import re
import time
from pathlib import Path

import anthropic

from scripts.selection_collapse.format_catalog import (
    catalog_token_budget,
    format_awareness_prompt,
    format_catalog,
)

ROOT = Path(__file__).resolve().parents[2]
POOLS_DIR = ROOT / "data" / "selection_collapse" / "pools"
RESULTS_DIR = ROOT / "data" / "selection_collapse" / "results" / "awareness"
TASKS_FILE = ROOT / "data" / "selection_collapse" / "skillsbench" / "selected_tasks.jsonl"


def parse_top5_from_response(response_text: str) -> list[str]:
    """Extract SKILL_XXX patterns from response, deduplicate, return up to 5."""
    matches = re.findall(r"SKILL_\d{3}", response_text)
    seen: set[str] = set()
    deduped: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
        if len(deduped) == 5:
            break
    return deduped


def call_claude_opus(prompt: str, system: str = "") -> str:
    """Call Claude Opus and return the text response."""
    client = anthropic.Anthropic()
    if not system:
        system = "You are a helpful assistant that selects relevant skills."
    message = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def load_tasks(tasks_file: Path) -> dict[str, dict]:
    """Load selected_tasks.jsonl into a dict keyed by task_id."""
    tasks: dict[str, dict] = {}
    with open(tasks_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            tasks[task["task_id"]] = task
    return tasks


def run_awareness_probes() -> None:
    """Run awareness probes for all pool files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load tasks once
    tasks = load_tasks(TASKS_FILE)

    pool_files = sorted(POOLS_DIR.glob("*.json"))
    print(f"Found {len(pool_files)} pool files")

    for pool_file in pool_files:
        pool_stem = pool_file.stem
        out_file = RESULTS_DIR / f"awareness_{pool_stem}.json"

        # Skip if output already exists (resume-safe)
        if out_file.exists():
            print(f"  SKIP {pool_stem} (already exists)")
            continue

        # Load pool data
        with open(pool_file) as f:
            pool_data = json.load(f)

        task_id = pool_data["task_id"]
        pool = pool_data["pool"]
        pool_size = pool_data["pool_size"]
        trial = pool_data["trial"]
        condition = pool_data.get("condition", "")

        # Get task instruction
        task = tasks[task_id]
        instruction = task["instruction"]

        # Find GT display IDs from pool
        gt_skill_ids = set(pool_data["gt_skill_ids"])
        gt_display_ids = [
            s["display_id"] for s in pool if s["skill_id"] in gt_skill_ids
        ]

        # Format catalog and prompt
        catalog = format_catalog(pool)
        prompt = format_awareness_prompt(instruction, catalog)
        token_budget = catalog_token_budget(pool)

        # Call Claude Opus
        print(f"  CALL {pool_stem} (pool_size={pool_size}, trial={trial})")
        raw_response = call_claude_opus(prompt)

        # Parse top-5
        model_top5 = parse_top5_from_response(raw_response)

        # Check if any GT skill is in top-5
        gt_in_top5 = any(gid in model_top5 for gid in gt_display_ids)

        # Save result
        result = {
            "pool_file": pool_file.name,
            "task_id": task_id,
            "pool_size": pool_size,
            "trial": trial,
            "condition": condition,
            "gt_display_ids": gt_display_ids,
            "model_top5": model_top5,
            "gt_in_top5": gt_in_top5,
            "raw_response": raw_response,
            "token_budget": token_budget,
        }

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"  DONE {pool_stem} -> gt_in_top5={gt_in_top5}")

        # Rate limiting
        time.sleep(0.5)


if __name__ == "__main__":
    run_awareness_probes()
