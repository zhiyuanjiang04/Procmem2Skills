from __future__ import annotations

from pathlib import Path

from procmem2skills.adapters.webarena import normalize_webarena_step
from procmem2skills.importers.common import ensure_list, first_present, load_records
from procmem2skills.models import BenchmarkKind, ExecutionMode, Trajectory


def import_webarena(path: Path, agent: str = "agent", harness: str = "browsergym/webarena") -> list[Trajectory]:
    browsergym_dirs = _discover_browsergym_result_dirs(path)
    if browsergym_dirs:
        trajectories = []
        for result_dir in browsergym_dirs:
            trajectory = _load_browsergym_trajectory(result_dir, agent=agent, harness=harness)
            if trajectory is not None:
                trajectories.append(trajectory)
        if trajectories:
            return trajectories

    trajectories = []
    for record in load_records(path):
        steps = ensure_list(first_present(record, "steps", "trajectory", "events", default=[]))
        if not steps:
            continue
        episode_id = str(first_present(record, "episode_id", "trace_id", "task_id", default=f"webarena-{len(trajectories)+1}"))
        instruction = str(first_present(record, "intent", "instruction", default="WebArena task"))
        task_id = str(first_present(record, "task_id", default=episode_id))
        events = [normalize_webarena_step(step, index) for index, step in enumerate(steps, start=1)]
        success = bool(first_present(record, "success", default=True))
        score = first_present(record, "score", "reward", default=1.0 if success else 0.0)
        trajectories.append(
            Trajectory(
                episode_id=episode_id,
                benchmark=BenchmarkKind.WEB_ARENA,
                harness=harness,
                agent=str(first_present(record, "agent", default=agent)),
                task_id=task_id,
                instruction=instruction,
                mode=ExecutionMode.OFFLINE_BOOTSTRAP,
                metadata={key: value for key, value in record.items() if key in {"site", "site_id", "session_id"}},
                events=events,
                completed=True,
                score=float(score) if score is not None else None,
            )
        )
    return trajectories


def _discover_browsergym_result_dirs(path: Path) -> list[Path]:
    if path.is_file():
        candidate = path.parent
        return [candidate] if _is_browsergym_result_dir(candidate) else []

    candidates = []
    if _is_browsergym_result_dir(path):
        candidates.append(path)
    candidates.extend(
        sorted(summary_path.parent for summary_path in path.rglob("summary_info.json") if _is_browsergym_result_dir(summary_path.parent))
    )
    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _is_browsergym_result_dir(path: Path) -> bool:
    return path.is_dir() and (path / "summary_info.json").exists() and any(path.glob("step_*.pkl.gz"))


def _load_browsergym_trajectory(path: Path, agent: str, harness: str) -> Trajectory | None:
    exp_result = _load_browsergym_exp_result(path)
    if exp_result is None:
        return None

    tape = getattr(exp_result, "tape", {}) or {}
    segments = ensure_list(tape.get("steps"))
    grouped: dict[int, dict[str, object]] = {}
    for segment in segments:
        step = int(((segment.get("metadata") or {}).get("step")) or 0)
        bucket = grouped.setdefault(step, {"observation": None, "thought": "", "actions": []})
        kind = segment.get("kind")
        if kind == "browsergym_observation":
            bucket["observation"] = segment
        elif kind == "browsergym_thought":
            bucket["thought"] = str(segment.get("text") or "")
        elif kind == "browsergym_action":
            bucket["actions"].append(segment)

    summary = getattr(exp_result, "summary_info", {}) or {}
    env_args = getattr(getattr(exp_result, "exp_args", None), "env_args", None)
    task_name = str(getattr(env_args, "task_name", path.name))
    task_id = task_name.split("/")[-1]

    first_observation = None
    for step in sorted(grouped):
        observation = grouped[step]["observation"]
        if observation is not None:
            first_observation = observation
            break
    instruction = _browsergym_goal_text(first_observation)

    events = []
    ordered_steps = [step for step in sorted(grouped) if grouped[step]["actions"]]
    for step in ordered_steps:
        bucket = grouped[step]
        observation_segment = bucket["observation"]
        observation_payload = _browsergym_observation_payload(observation_segment)
        thought = str(bucket["thought"] or "")
        action_segments = ensure_list(bucket["actions"])
        for action_index, action_segment in enumerate(action_segments, start=1):
            raw_step = {
                "observation": observation_payload,
                "action": _browsergym_action_payload(action_segment),
                "info": {
                    "ok": not bool(summary.get("err_msg")),
                    "message": summary.get("err_msg") or "",
                },
                "success_signal": "task completed" if bool(summary.get("terminated")) and float(summary.get("cum_reward", 0.0)) > 0 else None,
            }
            event = normalize_webarena_step(raw_step, len(events) + 1)
            if action_index == 1 and thought:
                event.thought = thought
            events.append(event)

    if not events:
        return None

    return Trajectory(
        episode_id=str(path.name),
        benchmark=BenchmarkKind.WEB_ARENA,
        harness=harness,
        agent=str(agent),
        task_id=task_id,
        instruction=instruction or task_id,
        mode=ExecutionMode.OFFLINE_BOOTSTRAP,
        metadata={
            "source_format": "browsergym-exp-dir",
            "exp_dir": str(path),
        },
        events=events,
        completed=True,
        score=float(summary.get("cum_reward", 0.0)),
    )


def _load_browsergym_exp_result(path: Path):
    try:
        from browsergym.experiments.loop import get_exp_result
    except ImportError as exc:  # pragma: no cover - exercised only in browsergym-enabled envs
        raise RuntimeError(
            "Importing BrowserGym/WebArena result directories requires browsergym to be installed."
        ) from exc
    return get_exp_result(path)


def _browsergym_goal_text(observation_segment: dict | None) -> str:
    if observation_segment is None:
        return "WebArena task"
    obs = observation_segment.get("obs") or {}
    return str(obs.get("goal") or "WebArena task")


def _browsergym_observation_payload(observation_segment: dict | None) -> dict:
    obs = (observation_segment or {}).get("obs") or {}
    url = _browsergym_active_url(obs)
    summary = str(obs.get("goal") or obs.get("last_action_error") or url or "WebArena step")
    dom = str(obs.get("axtree_txt") or obs.get("dom_txt") or "")
    screenshot = (observation_segment or {}).get("screenshot") or ""
    return {
        "summary": summary,
        "url": url,
        "dom": dom,
        "screenshot": screenshot,
    }


def _browsergym_active_url(obs: dict) -> str:
    urls = ensure_list(obs.get("open_pages_urls"))
    index = obs.get("active_page_index")
    if isinstance(index, int) and 0 <= index < len(urls):
        return str(urls[index])
    if urls:
        return str(urls[0])
    return str(obs.get("url") or "")


def _browsergym_action_payload(segment: dict) -> dict:
    name = str(segment.get("name") or "web-action")
    arguments = ensure_list(segment.get("arguments"))
    normalized_name = name
    normalized_args: dict[str, object] = {}

    if name in {"fill", "type"}:
        normalized_name = "type"
        if arguments:
            normalized_args["selector"] = arguments[0]
        if len(arguments) > 1:
            normalized_args["value"] = arguments[1]
    elif name in {"select", "select_option"}:
        normalized_name = "select"
        if arguments:
            normalized_args["selector"] = arguments[0]
        if len(arguments) > 1:
            normalized_args["value"] = arguments[1]
    elif name in {"click", "hover", "press"}:
        if arguments:
            normalized_args["selector"] = arguments[0]
    elif name in {"goto", "go_back", "go_forward", "new_tab", "tab_focus", "open"}:
        normalized_name = "navigate"
        if arguments:
            normalized_args["target"] = arguments[0]
    else:
        if arguments:
            normalized_args["value"] = ", ".join(str(argument) for argument in arguments)

    return {
        "name": normalized_name,
        "arguments": normalized_args,
        "raw": f"{name}({', '.join(repr(argument) for argument in arguments)})",
    }
