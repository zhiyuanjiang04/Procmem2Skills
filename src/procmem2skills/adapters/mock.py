from __future__ import annotations

from dataclasses import dataclass

from procmem2skills.adapters.base import (
    AdapterCapability,
    AdapterMode,
    AdapterProfile,
    BenchmarkAdapter,
    StepResult,
    TaskDescriptor,
)
from procmem2skills.models import Action, BenchmarkKind, Observation


PROFILE = AdapterProfile(
    benchmark=BenchmarkKind.TERMINAL_BENCH,
    harness="mock/terminal",
    mode=AdapterMode.INTERACTIVE,
    observation_modalities=["terminal-output"],
    action_modalities=["shell-command"],
    evaluation_style="mock functional completion",
    notes="Local interactive adapter for validating the live runner without external harnesses.",
)


@dataclass
class ScriptedStep:
    expected_command: str
    observation: Observation
    done: bool = False
    success_signal: str | None = None
    message: str | None = None


class MockTerminalAdapter(BenchmarkAdapter):
    profile = PROFILE
    capabilities = frozenset({AdapterCapability.TERMINAL, AdapterCapability.FUNCTIONAL_EVAL})

    def __init__(self) -> None:
        self._task = TaskDescriptor(
            task_id="mock-terminal-task",
            instruction="Run pytest and inspect the failure before editing anything.",
        )
        self._steps = [
            ScriptedStep(
                expected_command="pytest -q",
                observation=Observation(
                    summary="Pytest failed with one assertion error",
                    text="/workspace/mock-repo",
                    structured={"stdout": "1 failed, 3 passed"},
                ),
                done=True,
                success_signal="captured failing test output",
                message="baseline failure reproduced",
            )
        ]
        self._index = 0
        self._done = False

    def task(self) -> TaskDescriptor:
        return self._task

    def reset(self, task_id: str | None = None) -> Observation:
        self._index = 0
        self._done = False
        return Observation(
            summary="Repository is ready. Run tests to inspect failures.",
            text="/workspace/mock-repo",
            structured={"stdout": "README.md\nsrc\ntests"},
        )

    def step(self, action: Action) -> StepResult:
        step = self._steps[self._index]
        command = action.arguments.get("command") or action.raw or ""
        ok = command == step.expected_command
        self._done = ok and step.done
        self._index = min(self._index + 1, len(self._steps) - 1)
        return StepResult(
            observation=step.observation,
            reward=1.0 if ok else 0.0,
            done=self._done,
            info={
                "message": step.message if ok else f"unexpected command: {command}",
                "exit_code": 1 if ok else 2,
                "error": None if ok else "unexpected-command",
            },
            artifacts=[],
            success_signal=step.success_signal if ok else None,
        )

    def is_done(self) -> bool:
        return self._done

    def score(self) -> dict:
        return {"score": 1.0 if self._done else 0.0}
