from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from procmem2skills.adapters.terminal_bench import normalize_terminal_bench_step
from procmem2skills.importers.common import ensure_list, first_present, load_records
from procmem2skills.models import BenchmarkKind, ExecutionMode, Trajectory

_PROMPT_RE = re.compile(r"(?m)^(?P<prompt>[^\n]*@[^:\n]+:(?P<cwd>[^#\n]*)# )")
_TASK_DESCRIPTION_RE = re.compile(r"Task Description:\n(?P<body>.*?)(?:\n\nCurrent terminal state:|\Z)", re.S)
_ERROR_RE = re.compile(
    r"(?m)(Traceback \(most recent call last\):|^bash: |^/bin/sh: |^sh: |No module named |ImportError: |AttributeError: )"
)
_COMMAND_ARGUMENT_KEYS = {
    "bash_command": "keystrokes",
    "exec_command": "cmd",
    "do_command": "command",
    "run_command": "command",
}


@dataclass(frozen=True)
class TerminalBenchBundle:
    trajectory_path: Path
    config_path: Path | None = None
    result_path: Path | None = None


@dataclass(frozen=True)
class TerminalBlock:
    cwd: str | None
    output_text: str | None


def _load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _normalize_dataset_name(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return text


def _normalize_dataset_version(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _compose_dataset_spec(name: str | None, version: str | None) -> str | None:
    normalized_name = _normalize_dataset_name(name)
    if not normalized_name:
        return None
    normalized_version = _normalize_dataset_version(version)
    if not normalized_version:
        return normalized_name
    return f"{normalized_name}@{normalized_version}"


def _extract_dataset_identity(
    *,
    config_payload: dict,
    result_payload: dict,
    task_payload: dict,
    harness: str,
) -> tuple[str, str | None, str]:
    dataset_candidates = [
        result_payload.get("dataset"),
        config_payload.get("dataset"),
        task_payload.get("dataset"),
        result_payload.get("dataset_spec"),
        config_payload.get("dataset_spec"),
    ]
    for candidate in dataset_candidates:
        if isinstance(candidate, dict):
            name = _normalize_dataset_name(candidate.get("name") or candidate.get("dataset_name"))
            version = _normalize_dataset_version(candidate.get("version") or candidate.get("dataset_version"))
            spec = _compose_dataset_spec(name, version)
            if spec:
                return name or "terminal-bench", version, spec
            continue
        if candidate is None:
            continue
        text = str(candidate).strip()
        if not text:
            continue
        if "@" in text:
            raw_name, raw_version = text.split("@", 1)
            name = _normalize_dataset_name(raw_name)
            version = _normalize_dataset_version(raw_version)
            spec = _compose_dataset_spec(name, version)
            if spec:
                return name or "terminal-bench", version, spec
            continue
        normalized_name = _normalize_dataset_name(text)
        if normalized_name:
            return normalized_name, None, normalized_name

    dataset_name = _normalize_dataset_name(
        first_present(
            result_payload,
            "dataset_name",
            default=first_present(config_payload, "dataset_name", default=None),
        )
    )
    dataset_version = _normalize_dataset_version(
        first_present(
            result_payload,
            "dataset_version",
            default=first_present(config_payload, "dataset_version", default=None),
        )
    )
    if dataset_name:
        return dataset_name, dataset_version, _compose_dataset_spec(dataset_name, dataset_version) or dataset_name

    harness_prefix = str(harness or "").split("/", 1)[0].strip().lower()
    if harness_prefix:
        if "@" in harness_prefix:
            raw_name, raw_version = harness_prefix.split("@", 1)
            name = _normalize_dataset_name(raw_name)
            version = _normalize_dataset_version(raw_version)
            spec = _compose_dataset_spec(name, version)
            if spec:
                return name or "terminal-bench", version, spec
        if "terminal-bench" in harness_prefix:
            return harness_prefix, None, harness_prefix

    source_name = _normalize_dataset_name(first_present(result_payload, "source", default="terminal-bench")) or "terminal-bench"
    return source_name, None, source_name


def _normalize_task_tags(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        chunks = [chunk.strip() for chunk in value.split(",")]
        return sorted({chunk for chunk in chunks if chunk})
    if isinstance(value, (list, tuple, set)):
        tags = [str(item).strip() for item in value if str(item).strip()]
        return sorted(set(tags))
    text = str(value).strip()
    return [text] if text else []


def _discover_terminal_bench_bundles(path: Path) -> list[TerminalBenchBundle]:
    if path.is_file():
        if path.name == "trajectory.json":
            parent = path.parent.parent if path.parent.name == "agent" else path.parent
            return [
                TerminalBenchBundle(
                    trajectory_path=path,
                    config_path=parent / "config.json",
                    result_path=parent / "result.json",
                )
            ]
        return []

    trajectory_paths = sorted(path.rglob("trajectory.json"))
    bundles = []
    for trajectory_path in trajectory_paths:
        parent = trajectory_path.parent.parent if trajectory_path.parent.name == "agent" else trajectory_path.parent
        bundles.append(
            TerminalBenchBundle(
                trajectory_path=trajectory_path,
                config_path=parent / "config.json",
                result_path=parent / "result.json",
            )
        )
    return bundles


def _looks_like_atif_trajectory(record: dict) -> bool:
    return isinstance(record, dict) and "schema_version" in record and isinstance(record.get("steps"), list)


def _extract_task_description(message: str) -> str:
    match = _TASK_DESCRIPTION_RE.search(message)
    if match:
        return match.group("body").strip()
    return message.strip()


def _parse_terminal_blocks(content: str) -> list[TerminalBlock]:
    matches = list(_PROMPT_RE.finditer(content))
    if not matches:
        return []
    blocks = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        block_text = content[start:end].strip("\n")
        _, _, remainder = block_text.partition("\n")
        cwd = match.group("cwd").strip() or None
        output_text = remainder.strip("\n") or None
        blocks.append(TerminalBlock(cwd=cwd, output_text=output_text))
    return blocks


def _infer_ok(output_text: str | None) -> bool:
    if not output_text:
        return True
    return _ERROR_RE.search(output_text) is None


def _command_summary(command: str) -> str:
    first_line = command.splitlines()[0].strip()
    if len(first_line) > 96:
        first_line = first_line[:93] + "..."
    return f"Execute `{first_line}`"


def _extract_command_from_tool_call(tool_call: dict) -> str | None:
    function_name = str(tool_call.get("function_name", "")).strip()
    if not function_name:
        return None
    arguments = tool_call.get("arguments") or {}
    if not isinstance(arguments, dict):
        return None
    argument_key = _COMMAND_ARGUMENT_KEYS.get(function_name)
    if argument_key is None:
        return None
    value = arguments.get(argument_key)
    if value is None:
        return None
    command = str(value).strip()
    return command or None


def _tool_call_id(tool_call: dict) -> str | None:
    for key in ("tool_call_id", "id", "call_id"):
        value = tool_call.get(key)
        if value:
            return str(value)
    return None


def _build_observation_index(results: list[dict]) -> tuple[dict[str, str], list[str], str]:
    by_call_id: dict[str, str] = {}
    sequential_texts: list[str] = []
    merged_texts: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        content = result.get("content")
        if content is None:
            continue
        text = str(content).strip()
        if not text:
            continue
        merged_texts.append(text)
        call_id = result.get("source_call_id")
        if call_id:
            by_call_id[str(call_id)] = text
        else:
            sequential_texts.append(text)
    merged = "\n\n".join(merged_texts)
    return by_call_id, sequential_texts, merged


def _load_atif_trajectory(bundle: TerminalBenchBundle, agent: str, harness: str) -> Trajectory | None:
    trajectory_payload = _load_json(bundle.trajectory_path)
    config_payload = _load_json(bundle.config_path)
    result_payload = _load_json(bundle.result_path)
    return _trajectory_from_atif_payload(
        trajectory_payload,
        config_payload=config_payload,
        result_payload=result_payload,
        agent=agent,
        harness=harness,
        source_name=bundle.trajectory_path.stem,
    )


def _trajectory_from_atif_payload(
    trajectory_payload: dict,
    config_payload: dict,
    result_payload: dict,
    agent: str,
    harness: str,
    source_name: str = "trajectory",
) -> Trajectory | None:
    if not _looks_like_atif_trajectory(trajectory_payload):
        return None
    steps = ensure_list(trajectory_payload.get("steps"))
    if not steps:
        return None

    user_messages = [str(step.get("message", "")) for step in steps if step.get("source") == "user"]
    instruction = _extract_task_description(user_messages[0]) if user_messages else "Terminal-Bench task"

    events = []
    for step in steps:
        if step.get("source") != "agent":
            continue
        tool_calls = ensure_list(step.get("tool_calls"))
        command_calls = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            command = _extract_command_from_tool_call(tool_call)
            if command:
                command_calls.append(tool_call)
        if not command_calls:
            continue

        observation = step.get("observation") or {}
        results = ensure_list(observation.get("results"))
        by_call_id, sequential_texts, merged_content = _build_observation_index(results)
        merged_blocks = _parse_terminal_blocks(merged_content) if merged_content else []
        thought = step.get("message")

        for call_index, tool_call in enumerate(command_calls):
            arguments = tool_call.get("arguments") or {}
            command = _extract_command_from_tool_call(tool_call)
            if not command:
                continue

            call_id = _tool_call_id(tool_call)
            call_content = by_call_id.get(call_id) if call_id else None
            if call_content is None and call_index < len(sequential_texts):
                call_content = sequential_texts[call_index]
            call_blocks = _parse_terminal_blocks(call_content) if call_content else []
            block = call_blocks[0] if call_blocks else (merged_blocks[call_index] if call_index < len(merged_blocks) else None)
            output_text = block.output_text if block else (call_content or merged_content or None)
            ok = _infer_ok(output_text)
            raw_step = {
                "summary": _command_summary(command),
                "thought": thought,
                "cwd": block.cwd if block else (str(arguments.get("cwd")) if arguments.get("cwd") else None),
                "command": command,
                "stdout": output_text,
                "stderr": None,
                "ok": ok,
                "exit_code": 0 if ok else 1,
                "tests": ((result_payload.get("verifier_result") or {}).get("rewards") or {}),
            }
            events.append(normalize_terminal_bench_step(raw_step, len(events) + 1))

    if not events:
        return None

    rewards = (result_payload.get("verifier_result") or {}).get("rewards") or {}
    reward = first_present(rewards, "reward", default=None)
    task_config = config_payload.get("task") or {}
    task_payload = result_payload.get("task") or {}
    if not isinstance(task_config, dict):
        task_config = {}
    if not isinstance(task_payload, dict):
        task_payload = {}
    merged_task_payload = dict(task_payload)
    merged_task_payload.update(task_config)
    dataset_name, dataset_version, dataset_spec = _extract_dataset_identity(
        config_payload=config_payload,
        result_payload=result_payload,
        task_payload=merged_task_payload,
        harness=harness,
    )
    agent_payload = trajectory_payload.get("agent") or {}
    episode_id = str(
        first_present(result_payload, "trial_name", default=None)
        or first_present(config_payload, "trial_name", default=None)
        or trajectory_payload.get("session_id")
        or f"terminal-bench-{source_name}"
    )
    task_id = str(
        first_present(result_payload, "task_name", default=None)
        or task_config.get("path")
        or trajectory_payload.get("session_id")
        or episode_id
    )
    resolved_agent = str(agent_payload.get("model_name") or agent_payload.get("name") or agent)

    metadata = {
        "source": first_present(result_payload, "source", default="terminal-bench"),
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "dataset_spec": dataset_spec,
        "task_checksum": result_payload.get("task_checksum"),
        "task_path": first_present(task_config, "path", default=first_present(task_payload, "path", default=None)),
        "task_category": first_present(task_config, "category", default=first_present(task_payload, "category", default=None)),
        "task_difficulty": first_present(
            task_config,
            "difficulty",
            default=first_present(task_payload, "difficulty", default=None),
        ),
        "task_tags": _normalize_task_tags(
            first_present(task_config, "tags", default=first_present(task_payload, "tags", default=None))
        ),
        "task_repo": first_present(task_config, "repo", default=first_present(task_payload, "repo", default=None)),
        "task_git_url": first_present(
            task_config,
            "git_url",
            default=first_present(task_payload, "git_url", default=None),
        ),
        "task_git_commit_id": first_present(
            task_config,
            "git_commit_id",
            default=first_present(task_payload, "git_commit_id", default=None),
        ),
        "trial_name": first_present(result_payload, "trial_name", default=None)
        or first_present(config_payload, "trial_name", default=None),
        "schema_version": trajectory_payload.get("schema_version"),
        "import_format": "atif",
    }
    metadata.update({key: value for key, value in result_payload.items() if key in {"started_at", "finished_at"}})
    metadata = {key: value for key, value in metadata.items() if value is not None}

    return Trajectory(
        episode_id=episode_id,
        benchmark=BenchmarkKind.TERMINAL_BENCH,
        harness=harness,
        agent=resolved_agent,
        task_id=task_id,
        instruction=instruction,
        mode=ExecutionMode.OFFLINE_BOOTSTRAP,
        metadata=metadata,
        events=events,
        completed=True,
        score=float(reward) if reward is not None else None,
    )


def import_terminal_bench(path: Path, agent: str = "agent", harness: str = "terminal-bench/harness") -> list[Trajectory]:
    bundles = _discover_terminal_bench_bundles(path)
    if bundles:
        trajectories = []
        for bundle in bundles:
            trajectory = _load_atif_trajectory(bundle, agent=agent, harness=harness)
            if trajectory is not None:
                trajectories.append(trajectory)
        if trajectories:
            return trajectories

    trajectories = []
    for record in load_records(path):
        if _looks_like_atif_trajectory(record):
            trajectory = _trajectory_from_atif_payload(
                record,
                config_payload={},
                result_payload={},
                agent=agent,
                harness=harness,
                source_name=path.stem if path.is_file() else f"record-{len(trajectories)+1}",
            )
            if trajectory is not None:
                trajectories.append(trajectory)
            continue
        steps = ensure_list(first_present(record, "steps", "trajectory", "events", default=[]))
        if not steps:
            continue
        episode_id = str(first_present(record, "episode_id", "run_id", "task_id", default=f"terminal-bench-{len(trajectories)+1}"))
        instruction = str(first_present(record, "instruction", "prompt", default="Terminal-Bench task"))
        task_id = str(first_present(record, "task_id", default=episode_id))
        events = [normalize_terminal_bench_step(step, index) for index, step in enumerate(steps, start=1)]
        success = bool(first_present(record, "success", default=True))
        score = first_present(record, "score", default=1.0 if success else 0.0)
        dataset_name, dataset_version, dataset_spec = _extract_dataset_identity(
            config_payload={},
            result_payload=record,
            task_payload=(record.get("task") or {}) if isinstance(record.get("task"), dict) else {},
            harness=harness,
        )
        task_payload = record.get("task") if isinstance(record.get("task"), dict) else {}
        metadata = {
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "dataset_spec": dataset_spec,
            "task_path": first_present(task_payload, "path", default=first_present(record, "task_path", default=None)),
            "task_category": first_present(
                task_payload,
                "category",
                default=first_present(record, "category", default=None),
            ),
            "task_difficulty": first_present(
                task_payload,
                "difficulty",
                default=first_present(record, "difficulty", default=None),
            ),
            "task_tags": _normalize_task_tags(
                first_present(task_payload, "tags", default=first_present(record, "tags", default=None))
            ),
            "task_checksum": first_present(record, "task_checksum", default=None),
            "task_repo": first_present(task_payload, "repo", default=first_present(record, "repo", default=None)),
        }
        metadata = {key: value for key, value in metadata.items() if value not in (None, "", [])}
        trajectories.append(
            Trajectory(
                episode_id=episode_id,
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                harness=harness,
                agent=str(first_present(record, "agent", default=agent)),
                task_id=task_id,
                instruction=instruction,
                mode=ExecutionMode.OFFLINE_BOOTSTRAP,
                metadata=metadata,
                events=events,
                completed=True,
                score=float(score) if score is not None else None,
            )
        )
    return trajectories
