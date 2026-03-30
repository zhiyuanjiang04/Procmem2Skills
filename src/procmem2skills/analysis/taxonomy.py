from __future__ import annotations

import re
from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from procmem2skills.models import Trajectory
from procmem2skills.normalization import operation_family


class TaxonomyLeaf(BaseModel):
    count: int = 0
    sample_tasks: list[str] = Field(default_factory=list)
    action_families: list[str] = Field(default_factory=list)


class TaxonomyReport(BaseModel):
    total_tasks: int = 0
    hierarchy: dict[str, dict[str, dict[str, dict[str, TaxonomyLeaf]]]] = Field(default_factory=dict)


def build_taxonomy_report(trajectories: list[Trajectory], sample_limit: int = 5) -> TaxonomyReport:
    hierarchy: dict[str, dict[str, dict[str, dict[str, TaxonomyLeaf]]]] = {}
    leaves: dict[tuple[str, str, str, str], dict[str, object]] = defaultdict(
        lambda: {"count": 0, "sample_tasks": [], "action_families": Counter()}
    )

    for trajectory in trajectories:
        environment = _environment_label(trajectory)
        domain = _domain_label(trajectory)
        goal = _goal_label(trajectory)
        subgoal = _subgoal_label(trajectory)
        key = (environment, domain, goal, subgoal)
        bucket = leaves[key]
        bucket["count"] = int(bucket["count"]) + 1
        samples = bucket["sample_tasks"]
        if trajectory.task_id not in samples and len(samples) < sample_limit:
            samples.append(trajectory.task_id)
        action_counter: Counter = bucket["action_families"]  # type: ignore[assignment]
        for family in _trajectory_action_families(trajectory):
            action_counter[family] += 1

    for environment, domain, goal, subgoal in sorted(leaves):
        bucket = leaves[(environment, domain, goal, subgoal)]
        hierarchy.setdefault(environment, {}).setdefault(domain, {}).setdefault(goal, {})[subgoal] = TaxonomyLeaf(
            count=int(bucket["count"]),
            sample_tasks=list(bucket["sample_tasks"]),
            action_families=[family for family, _ in bucket["action_families"].most_common(6)],
        )

    return TaxonomyReport(total_tasks=len(trajectories), hierarchy=hierarchy)


def _environment_label(trajectory: Trajectory) -> str:
    benchmark = trajectory.benchmark.value
    if benchmark in {"mind2web", "webarena"}:
        return "browser"
    if benchmark == "terminal-bench":
        return "terminal"
    if benchmark == "alfworld":
        return "text-world"
    families = _trajectory_action_families(trajectory)
    if any(family.startswith("browser-") for family in families):
        return "browser"
    if any(family.startswith("terminal-") for family in families):
        return "terminal"
    if any(family.startswith("text-world-") for family in families):
        return "text-world"
    return benchmark


def _domain_label(trajectory: Trajectory) -> str:
    metadata = trajectory.metadata or {}
    benchmark = trajectory.benchmark.value

    if benchmark == "mind2web":
        parts = [metadata.get("domain"), metadata.get("subdomain"), metadata.get("website")]
        return "/".join(str(part) for part in parts if part) or "mind2web/unknown"
    if benchmark == "webarena":
        return str(metadata.get("site") or metadata.get("site_id") or "webarena/site")
    if benchmark == "alfworld":
        return str(metadata.get("task_type") or metadata.get("scene") or "alfworld/task")
    if benchmark == "terminal-bench":
        task_id = trajectory.task_id or "terminal-task"
        prefix = task_id.split("-", 1)[0]
        return f"{prefix}/{task_id}"

    return benchmark


def _goal_label(trajectory: Trajectory) -> str:
    metadata = trajectory.metadata or {}
    task_type = str(metadata.get("task_type") or "").lower()
    if any(token in task_type for token in {"clean", "heat", "cool", "put", "place", "pick", "examine", "slice"}):
        return "manipulate-object"

    text = " ".join(
        part
        for part in [
            trajectory.task_id,
            trajectory.instruction,
            task_type,
        ]
        if part
    ).lower()

    patterns = [
        ("build-install-fix", r"\b(build|compile|install|fix|patch|compatib|dependency|extension)\b"),
        ("test-verify", r"\b(test|verify|assert|check|pass|fail|debug)\b"),
        ("search-locate", r"\b(search|find|locate|lookup|grep|query)\b"),
        ("navigate-open", r"\b(open|navigate|visit|go to|browse|access)\b"),
        ("form-fill-select", r"\b(type|enter|fill|select|choose|book|register|login|checkout)\b"),
        ("extract-report", r"\b(extract|report|summarize|collect|read|inspect|compare)\b"),
        ("manipulate-object", r"\b(clean|heat|cool|place|put|take|pick|move|toggle|wash)\b"),
    ]
    for label, pattern in patterns:
        if re.search(pattern, text):
            return label
    return "other"


def _subgoal_label(trajectory: Trajectory) -> str:
    metadata = trajectory.metadata or {}
    benchmark = trajectory.benchmark.value
    if benchmark == "mind2web":
        return str(metadata.get("website") or metadata.get("subdomain") or "task")
    if benchmark == "webarena":
        return str(metadata.get("site") or metadata.get("site_id") or "task")
    if benchmark == "alfworld":
        return str(metadata.get("scene") or metadata.get("split") or "task")
    if benchmark == "terminal-bench":
        return trajectory.task_id
    return trajectory.task_id


def _trajectory_action_families(trajectory: Trajectory) -> list[str]:
    families = []
    for event in trajectory.events:
        if event.action is None:
            continue
        operation = event.action.raw or event.action.name
        if event.action.arguments:
            command = event.action.arguments.get("command")
            element = event.action.arguments.get("element")
            value = event.action.arguments.get("value")
            if command:
                operation = f"{event.action.name}(command={command})"
            elif element is not None or value is not None:
                rendered = ", ".join(
                    f"{key}={value}"
                    for key, value in [("element", element), ("value", value)]
                    if value is not None
                )
                operation = f"{event.action.name}({rendered})" if rendered else event.action.name
        families.append(operation_family(event.action.tool, operation))
    return sorted(set(families))
