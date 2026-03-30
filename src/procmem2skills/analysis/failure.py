from __future__ import annotations

import re
from collections import Counter, defaultdict

from procmem2skills.models import Trajectory

_EXCEPTION_LINE_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception):[^\n]+)")
_PYTEST_FAILED_TEST_PATTERN = re.compile(r"FAILED\s+([^\s]+)")
_PYTEST_ERROR_PREFIX_PATTERN = re.compile(r"^E\s+(.+)$", re.MULTILINE)


def extract_failure_signals_from_text(text: str, *, max_signals: int = 4) -> list[str]:
    if not text:
        return []

    signals: list[str] = []

    for match in _PYTEST_FAILED_TEST_PATTERN.finditer(text):
        candidate = f"failed-test:{_normalize_signal(match.group(1))}"
        _append_unique(signals, candidate, max_signals=max_signals)

    for match in _EXCEPTION_LINE_PATTERN.finditer(text):
        candidate = _normalize_signal(match.group(1))
        if _is_actionable_signal(candidate):
            _append_unique(signals, candidate, max_signals=max_signals)

    for match in _PYTEST_ERROR_PREFIX_PATTERN.finditer(text):
        candidate = _normalize_signal(match.group(1))
        if _is_actionable_signal(candidate):
            _append_unique(signals, candidate, max_signals=max_signals)

    lowered = text.lower()
    if "could not parse terminal agent response as json" in lowered:
        _append_unique(
            signals,
            "ValueError: could not parse terminal agent response as JSON",
            max_signals=max_signals,
        )
    if "timeout" in lowered or "timed out" in lowered:
        _append_unique(signals, "timeout", max_signals=max_signals)
    if "no such file or directory" in lowered:
        _append_unique(signals, "FileNotFoundError: no such file or directory", max_signals=max_signals)
    if "permission denied" in lowered:
        _append_unique(signals, "PermissionError: permission denied", max_signals=max_signals)

    return signals[:max_signals]


def build_failure_analysis_from_trajectories(
    trajectories: list[Trajectory],
    *,
    max_signals_per_task: int = 8,
    max_samples_per_task: int = 3,
) -> dict:
    task_buckets: dict[str, dict] = defaultdict(
        lambda: {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "signal_counter": Counter(),
            "sample_failures": [],
        }
    )
    global_signal_counter: Counter[str] = Counter()
    failed_trajectories = 0

    for trajectory in trajectories:
        task_name = trajectory.task_id or "unknown-task"
        bucket = task_buckets[task_name]
        bucket["attempts"] += 1

        failed = _trajectory_failed(trajectory)
        if not failed:
            bucket["successes"] += 1
            continue

        bucket["failures"] += 1
        failed_trajectories += 1

        signals = _collect_trajectory_failure_signals(trajectory)
        if not signals:
            signals = ["unknown-failure"]

        for signal in signals:
            bucket["signal_counter"][signal] += 1
            global_signal_counter[signal] += 1

        sample_failures = bucket["sample_failures"]
        if len(sample_failures) < max_samples_per_task:
            sample_failures.append(
                {
                    "episode_id": trajectory.episode_id,
                    "score": trajectory.score,
                    "completed": trajectory.completed,
                    "signals": signals[:3],
                }
            )

    by_task: dict[str, dict] = {}
    for task_name, bucket in sorted(task_buckets.items()):
        signal_counter: Counter[str] = bucket["signal_counter"]
        by_task[task_name] = {
            "attempts": bucket["attempts"],
            "successes": bucket["successes"],
            "failures": bucket["failures"],
            "failure_signals": [
                {"signature": signature, "count": count}
                for signature, count in signal_counter.most_common(max_signals_per_task)
            ],
            "sample_failures": bucket["sample_failures"],
        }

    return {
        "global": {
            "total_trajectories": len(trajectories),
            "failed_trajectories": failed_trajectories,
            "top_failure_signals": [
                {"signature": signature, "count": count}
                for signature, count in global_signal_counter.most_common(max_signals_per_task)
            ],
        },
        "by_task": by_task,
    }


def _trajectory_failed(trajectory: Trajectory) -> bool:
    if trajectory.score is not None:
        return trajectory.score < 1.0
    if not trajectory.completed:
        return True
    return any(event.result is not None and not event.result.ok for event in trajectory.events)


def _collect_trajectory_failure_signals(trajectory: Trajectory) -> list[str]:
    signals: list[str] = []

    if not trajectory.completed:
        _append_unique(signals, "trajectory-not-completed")
    if trajectory.score is not None and trajectory.score < 1.0:
        _append_unique(signals, "score-below-success-threshold")

    for event in trajectory.events:
        result = event.result
        if result is None or result.ok:
            continue
        for signal in extract_failure_signals_from_text(result.output_text or ""):
            _append_unique(signals, signal)

    metadata = trajectory.metadata or {}
    for key in ("exception_type", "failure_reason", "error_type"):
        value = metadata.get(key)
        if value:
            _append_unique(signals, _normalize_signal(str(value)))

    return signals


def _append_unique(target: list[str], candidate: str, *, max_signals: int = 12) -> None:
    normalized = _normalize_signal(candidate)
    if not normalized:
        return
    if normalized in target:
        return
    if len(target) >= max_signals:
        return
    target.append(normalized)


def _normalize_signal(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text).strip())
    if len(normalized) > 220:
        normalized = normalized[:217].rstrip() + "..."
    return normalized


def _is_actionable_signal(signal: str) -> bool:
    if not signal:
        return False
    if signal.endswith("..."):
        return False
    lowered = signal.lower()
    keywords = ("error", "exception", "failed", "timeout", "timed out")
    return any(keyword in lowered for keyword in keywords)
