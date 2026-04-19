"""CLI-backend driver: shells out to `claude -p` instead of the Anthropic SDK.

Drop-in replacement for Driver.run_one. Token counts are unavailable (None).
"""
from __future__ import annotations

import asyncio
import time

from .config import Probe, TrialRecord
from .parser import parse_response

_CLAUDE_CMD = "claude"
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.0  # seconds


class CLIDriver:
    """Async driver that shells out to the `claude` CLI (Max plan OAuth)."""

    def __init__(self, model: str, max_concurrency: int = 4):
        self._model = model
        self._sem = asyncio.Semaphore(max_concurrency)

    async def run_one(
        self,
        *,
        pool_id: str,
        probe: Probe,
        system_prompt: str,
        pool_block: str,
        user_prompt: str,
    ) -> TrialRecord:
        combined_system = system_prompt + "\n\n" + pool_block

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            if attempt > 0:
                await asyncio.sleep(_RETRY_BACKOFF)
            try:
                async with self._sem:
                    start = time.time()
                    proc = await asyncio.create_subprocess_exec(
                        _CLAUDE_CMD,
                        "-p", user_prompt,
                        "--model", self._model,
                        "--system-prompt", combined_system,
                        "--output-format", "text",
                        "--tools", "",
                        "--no-session-persistence",
                        "--permission-mode", "bypassPermissions",
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await proc.communicate()
                    latency_ms = int((time.time() - start) * 1000)

                if proc.returncode != 0:
                    err_text = stderr.decode(errors="replace").strip()
                    raise RuntimeError(
                        f"claude CLI exited {proc.returncode}: {err_text[:200]}"
                    )

                raw = stdout.decode(errors="replace").strip()
                parsed = parse_response(raw, probe=probe)
                return TrialRecord(
                    pool_id=pool_id,
                    probe=probe,
                    model=self._model,
                    raw_response=raw,
                    extracted_ids=parsed["extracted_ids"],
                    format_status=parsed["format_status"],
                    flags=parsed["flags"],
                    latency_ms=latency_ms,
                    input_tokens=None,
                    output_tokens=None,
                    cache_read_tokens=None,
                    cache_creation_tokens=None,
                )
            except Exception as exc:
                last_exc = exc

        # All retries exhausted — return a failed record instead of crashing the run
        return TrialRecord(
            pool_id=pool_id,
            probe=probe,
            model=self._model,
            raw_response=f"ERROR after {_MAX_RETRIES} attempts: {last_exc}",
            extracted_ids=[],
            format_status="fail",
            flags={"cli_error": True},
            latency_ms=0,
            input_tokens=None,
            output_tokens=None,
            cache_read_tokens=None,
            cache_creation_tokens=None,
        )
