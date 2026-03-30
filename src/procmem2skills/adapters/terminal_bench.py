from __future__ import annotations

from procmem2skills.adapters.base import AdapterMode, AdapterProfile
from procmem2skills.models import Action, ArtifactRef, BenchmarkKind, Event, ExecutionResult, Observation

PROFILE = AdapterProfile(
    benchmark=BenchmarkKind.TERMINAL_BENCH,
    harness="terminal-bench/harness",
    mode=AdapterMode.INTERACTIVE,
    observation_modalities=["terminal-output", "filesystem", "test-results"],
    action_modalities=["shell-command", "file-edit", "script-exec"],
    evaluation_style="end-to-end task success through oracle tests",
    notes="Best suited for validating that distilled skills can be consumed by real CLI agents.",
)


def _command_name(command: str) -> str:
    text = str(command or "").replace("\\n", "\n")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return line.split(" ", 1)[0]
    fallback = text.strip()
    return fallback.split(" ", 1)[0] if fallback else "noop"


def normalize_terminal_bench_step(raw_step: dict, step_id: int) -> Event:
    command = raw_step.get("command") or raw_step.get("action") or "noop"
    artifacts = []
    for diff_path in raw_step.get("diffs", []):
        artifacts.append(ArtifactRef(kind="file-diff", path=str(diff_path)))
    return Event(
        step_id=step_id,
        observation=Observation(
            summary=raw_step.get("summary", ""),
            text=raw_step.get("cwd"),
            structured={"stdout": raw_step.get("stdout"), "stderr": raw_step.get("stderr")},
        ),
        thought=raw_step.get("thought"),
        action=Action(
            tool=raw_step.get("tool", "terminal"),
            name=_command_name(command),
            arguments={"command": command},
            raw=command,
        ),
        result=ExecutionResult(
            ok=bool(raw_step.get("ok")) if raw_step.get("ok") is not None else (raw_step.get("exit_code") or 0) == 0,
            output_text=raw_step.get("stdout"),
            exit_code=raw_step.get("exit_code"),
            metadata={"stderr": raw_step.get("stderr"), "tests": raw_step.get("tests")},
        ),
        state_delta=raw_step.get("state_delta") or {},
        artifacts=artifacts,
        success_signal=raw_step.get("success_signal"),
    )
