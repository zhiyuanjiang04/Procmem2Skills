"""Tests for CLIDriver using a mocked asyncio.create_subprocess_exec."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from skills_retrieval.cli_driver import CLIDriver


class FakeProcess:
    """Fake asyncio subprocess."""

    def __init__(self, stdout_text: str, returncode: int = 0):
        self._stdout = stdout_text.encode()
        self.returncode = returncode
        self.communicate = AsyncMock(return_value=(self._stdout, b""))


def test_cli_driver_returns_trial_record():
    async def _run():
        fake_proc = FakeProcess("<skill>SKILL_000</skill>")

        with patch(
            "skills_retrieval.cli_driver.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=fake_proc),
        ):
            driver = CLIDriver(model="claude-sonnet-4-6", max_concurrency=1)
            rec = await driver.run_one(
                pool_id="p0",
                probe="selection",
                system_prompt="You are a retrieval subject.",
                pool_block="SKILL_000: alpha — desc",
                user_prompt="Task: test\n\nRespond with <skill>...</skill>",
            )

        assert rec.raw_response == "<skill>SKILL_000</skill>"
        assert rec.extracted_ids == ["SKILL_000"]
        assert rec.model == "claude-sonnet-4-6"
        assert rec.input_tokens is None
        assert rec.output_tokens is None
        assert rec.format_status in ("clean", "warning")

    asyncio.run(_run())


def test_cli_driver_retry_on_failure():
    """After transient failures, driver should retry and eventually return fail record."""

    async def _run():
        bad_proc = FakeProcess("", returncode=1)
        bad_proc.communicate = AsyncMock(return_value=(b"", b"rate limit"))

        with patch(
            "skills_retrieval.cli_driver.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=bad_proc),
        ):
            driver = CLIDriver(model="claude-sonnet-4-6", max_concurrency=1)
            # Patch sleep so the test doesn't actually wait
            with patch("skills_retrieval.cli_driver.asyncio.sleep", new=AsyncMock()):
                rec = await driver.run_one(
                    pool_id="p0",
                    probe="selection",
                    system_prompt="sys",
                    pool_block="pool",
                    user_prompt="prompt",
                )

        assert rec.format_status == "fail"
        assert rec.flags.get("cli_error")
        assert rec.extracted_ids == []

    asyncio.run(_run())
