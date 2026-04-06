#!/usr/bin/env python3
"""
Download and extract SkillsBench task-skill pairs for the Selection Collapse experiment.

Reads the cloned SkillsBench repo and extracts:
- Task instruction text (instruction.md)
- GT skill content (environment/skills/*/SKILL.md)
- Task domain/category (task.toml -> metadata.category)
- Task ID

Outputs tasks.jsonl and selected_tasks.jsonl to data/selection_collapse/skillsbench/
"""

import json
import re
import sys
from pathlib import Path

# Paths
REPO_ROOT = Path("/anvil/projects/x-cis260386/william/procmem2skills/skillsbench_repo")
TASKS_DIR = REPO_ROOT / "tasks"
OUTPUT_DIR = Path("/anvil/projects/x-cis260386/william/procmem2skills/procmem2skills/data/selection_collapse/skillsbench")


def parse_toml_category(toml_path: Path) -> str:
    """Extract category from task.toml without a TOML library."""
    text = toml_path.read_text()
    match = re.search(r'category\s*=\s*"([^"]+)"', text)
    return match.group(1) if match else "unknown"


def parse_skill_name(skill_md_text: str, fallback_name: str) -> str:
    """Extract skill name from YAML frontmatter in SKILL.md."""
    if skill_md_text.startswith("---"):
        end = skill_md_text.find("---", 3)
        if end != -1:
            frontmatter = skill_md_text[3:end]
            match = re.search(r'name:\s*(.+)', frontmatter)
            if match:
                return match.group(1).strip().strip('"').strip("'")
    return fallback_name


def extract_tasks():
    """Walk the tasks directory and extract all task-skill pairs."""
    if not TASKS_DIR.exists():
        print(f"ERROR: Tasks directory not found: {TASKS_DIR}", file=sys.stderr)
        sys.exit(1)

    tasks = []
    task_dirs = sorted(TASKS_DIR.iterdir())

    for idx, task_dir in enumerate(task_dirs):
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name

        # Read instruction
        instruction_path = task_dir / "instruction.md"
        if not instruction_path.exists():
            print(f"  SKIP {task_name}: no instruction.md")
            continue
        instruction = instruction_path.read_text().strip()

        # Read category from task.toml
        toml_path = task_dir / "task.toml"
        category = parse_toml_category(toml_path) if toml_path.exists() else "unknown"

        # Find all SKILL.md files
        skills_dir = task_dir / "environment" / "skills"
        gt_skills = []
        if skills_dir.exists():
            for skill_dir in sorted(skills_dir.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if skill_md.exists():
                    content = skill_md.read_text().strip()
                    name = parse_skill_name(content, skill_dir.name)
                    gt_skills.append({
                        "skill_id": f"gt_{task_name}_{skill_dir.name}",
                        "name": name,
                        "dir_name": skill_dir.name,
                        "content": content,
                    })

        if not gt_skills:
            print(f"  SKIP {task_name}: no SKILL.md files found")
            continue

        task_id = f"sb_{idx:03d}"
        tasks.append({
            "task_id": task_id,
            "task_name": task_name,
            "domain": category,
            "instruction": instruction,
            "gt_skills": gt_skills,
            "n_gt_skills": len(gt_skills),
        })

    return tasks


def select_tasks(tasks, min_tasks=30, max_tasks=50, max_skills=2, min_domains=6):
    """
    Filter to tasks with 1-max_skills GT skills, covering min_domains+ domains.
    Prefer diversity across domains.
    """
    # First filter: only tasks with <= max_skills
    candidates = [t for t in tasks if t["n_gt_skills"] <= max_skills]

    # Ensure domain diversity: pick tasks round-robin across domains
    from collections import defaultdict
    by_domain = defaultdict(list)
    for t in candidates:
        by_domain[t["domain"]].append(t)

    # Sort domains by number of tasks (ascending) so rare domains get picked first
    domain_order = sorted(by_domain.keys(), key=lambda d: len(by_domain[d]))

    selected = []
    selected_ids = set()

    # Round-robin pick
    while len(selected) < max_tasks:
        added_this_round = False
        for domain in domain_order:
            if len(selected) >= max_tasks:
                break
            remaining = [t for t in by_domain[domain] if t["task_id"] not in selected_ids]
            if remaining:
                pick = remaining[0]
                selected.append(pick)
                selected_ids.add(pick["task_id"])
                added_this_round = True
        if not added_this_round:
            break

    # Check constraints
    domains_covered = set(t["domain"] for t in selected)
    print(f"  Selected {len(selected)} tasks across {len(domains_covered)} domains")
    if len(selected) < min_tasks:
        print(f"  WARNING: only {len(selected)} tasks selected (wanted >= {min_tasks})")
    if len(domains_covered) < min_domains:
        print(f"  WARNING: only {len(domains_covered)} domains (wanted >= {min_domains})")

    return selected


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Extracting tasks from SkillsBench...")
    tasks = extract_tasks()
    print(f"Extracted {len(tasks)} tasks with GT skills")

    # Write all tasks
    all_path = OUTPUT_DIR / "tasks.jsonl"
    with open(all_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Wrote {len(tasks)} tasks to {all_path}")

    # Domain summary
    from collections import Counter
    domain_counts = Counter(t["domain"] for t in tasks)
    print(f"\nDomain distribution ({len(domain_counts)} domains):")
    for domain, count in domain_counts.most_common():
        print(f"  {domain}: {count}")

    # Skill count distribution
    skill_counts = Counter(t["n_gt_skills"] for t in tasks)
    print(f"\nSkill count distribution:")
    for n, count in sorted(skill_counts.items()):
        print(f"  {n} skills: {count} tasks")

    # Select subset
    print("\nSelecting tasks...")
    selected = select_tasks(tasks)

    sel_path = OUTPUT_DIR / "selected_tasks.jsonl"
    with open(sel_path, "w") as f:
        for t in selected:
            f.write(json.dumps(t) + "\n")
    print(f"Wrote {len(selected)} selected tasks to {sel_path}")

    # Print selected task summary
    sel_domain_counts = Counter(t["domain"] for t in selected)
    print(f"\nSelected task domains ({len(sel_domain_counts)} domains):")
    for domain, count in sel_domain_counts.most_common():
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    main()
