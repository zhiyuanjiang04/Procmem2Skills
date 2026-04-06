#!/usr/bin/env python3
"""Stage 2+3: Agent execution with skill catalog in context.

Agent receives the full skill catalog + task instruction. We analyze
the response to determine which skill(s) the agent referenced/used.

Output: one JSON file per run in data/selection_collapse/results/selection/
"""

import json
import re
import time
from pathlib import Path

import anthropic

from scripts.selection_collapse.format_catalog import (
    catalog_token_budget,
    format_catalog,
    format_selection_prompt,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POOL_DIR = PROJECT_ROOT / "data" / "selection_collapse" / "pools"
TASKS_FILE = PROJECT_ROOT / "data" / "selection_collapse" / "skillsbench" / "selected_tasks.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "selection_collapse" / "results" / "selection"

MODEL = "claude-opus-4-20250514"
MAX_TOKENS = 4096
SYSTEM_PROMPT = (
    "You are a skilled agent that uses available skills to complete tasks "
    "efficiently. Always reference skills by their ID when using them."
)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def parse_skill_references(response_text: str) -> list[str]:
    """Extract all SKILL_\\d{3} patterns from response, deduplicated, order-preserving."""
    seen: set[str] = set()
    result: list[str] = []
    for match in re.finditer(r"SKILL_\d{3}", response_text):
        skill_id = match.group()
        if skill_id not in seen:
            seen.add(skill_id)
            result.append(skill_id)
    return result


def call_claude_opus(prompt: str, system: str = "") -> str:
    """Call Claude Opus and return the text response."""
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system or SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def load_tasks(path: Path) -> dict[str, dict]:
    """Load selected_tasks.jsonl into a dict keyed by task_id."""
    tasks: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            tasks[task["task_id"]] = task
    return tasks


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_selection() -> None:
    """Run Stage 2+3 selection for every pool file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tasks once
    tasks = load_tasks(TASKS_FILE)

    pool_files = sorted(POOL_DIR.glob("*.json"))
    print(f"Found {len(pool_files)} pool files, output -> {OUTPUT_DIR}")

    for i, pool_path in enumerate(pool_files):
        pool_stem = pool_path.stem
        out_path = OUTPUT_DIR / f"selection_{pool_stem}.json"

        # Resume-safe: skip if already done
        if out_path.exists():
            print(f"[{i+1}/{len(pool_files)}] SKIP {pool_stem} (exists)")
            continue

        # Load pool
        with open(pool_path) as f:
            pool_data = json.load(f)

        task_id = pool_data["task_id"]
        pool = pool_data["pool"]
        pool_size = pool_data["pool_size"]
        trial = pool_data["trial"]
        condition = pool_data["condition"]

        # Look up task instruction
        task = tasks.get(task_id)
        if task is None:
            print(f"[{i+1}/{len(pool_files)}] SKIP {pool_stem} (task_id {task_id} not found)")
            continue
        instruction = task["instruction"]

        # Build prompt
        catalog_text = format_catalog(pool)
        prompt = format_selection_prompt(instruction, catalog_text)

        # Token budget info
        budget = catalog_token_budget(pool)

        # Determine GT display_ids
        gt_display_ids = [s["display_id"] for s in pool if s.get("is_gt", False)]

        print(f"[{i+1}/{len(pool_files)}] RUN {pool_stem} (pool_size={pool_size}, trial={trial})")

        # Call the model
        raw_response = call_claude_opus(prompt)

        # Parse references
        referenced_skills = parse_skill_references(raw_response)
        n_skills_referenced = len(referenced_skills)
        first_selection = referenced_skills[0] if referenced_skills else None

        # Evaluate
        gt_set = set(gt_display_ids)
        first_is_gt = first_selection in gt_set if first_selection else False
        any_gt_selected = bool(gt_set & set(referenced_skills))
        independent_solve = n_skills_referenced == 0

        # Build result
        result = {
            "pool_file": pool_path.name,
            "task_id": task_id,
            "pool_size": pool_size,
            "trial": trial,
            "condition": condition,
            "gt_display_ids": gt_display_ids,
            "referenced_skills": referenced_skills,
            "first_selection": first_selection,
            "first_is_gt": first_is_gt,
            "any_gt_selected": any_gt_selected,
            "n_skills_referenced": n_skills_referenced,
            "raw_response": raw_response,
            "independent_solve": independent_solve,
            "token_budget": budget,
        }

        # Write atomically via tmp
        tmp_path = out_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(result, f, indent=2)
        tmp_path.rename(out_path)

        print(f"  -> refs={referenced_skills}, first_is_gt={first_is_gt}, any_gt={any_gt_selected}")

        # Rate limit
        time.sleep(0.5)

    print("Done.")


if __name__ == "__main__":
    run_selection()
