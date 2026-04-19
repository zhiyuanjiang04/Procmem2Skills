from __future__ import annotations

import asyncio
import time
from typing import Any

from .config import Probe, TrialRecord
from .parser import parse_response


class Driver:
    """Async Anthropic SDK wrapper with prompt caching on the pool block.

    Strict responsibility: dispatch, retry, parse. No pool building, no scoring.
    """

    def __init__(self, client: Any, model: str, max_concurrency: int = 8, max_tokens: int = 1024):
        self._client = client
        self._model = model
        self._sem = asyncio.Semaphore(max_concurrency)
        self._max_tokens = max_tokens

    async def run_one(
        self,
        *,
        pool_id: str,
        probe: Probe,
        system_prompt: str,
        pool_block: str,
        user_prompt: str,
    ) -> TrialRecord:
        async with self._sem:
            start = time.time()
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=0.0,
                system=[
                    {"type": "text", "text": system_prompt},
                    {"type": "text", "text": pool_block, "cache_control": {"type": "ephemeral"}},
                ],
                messages=[{"role": "user", "content": user_prompt}],
            )
            latency_ms = int((time.time() - start) * 1000)

        raw = resp.content[0].text
        parsed = parse_response(raw, probe=probe)
        usage = getattr(resp, "usage", None)

        return TrialRecord(
            pool_id=pool_id,
            probe=probe,
            model=self._model,
            raw_response=raw,
            extracted_ids=parsed["extracted_ids"],
            format_status=parsed["format_status"],
            flags=parsed["flags"],
            latency_ms=latency_ms,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) if usage else None,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", None) if usage else None,
        )
