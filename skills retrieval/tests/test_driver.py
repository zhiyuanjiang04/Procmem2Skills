import asyncio
from unittest.mock import AsyncMock, MagicMock

from skills_retrieval.driver import Driver


class FakeMessages:
    def __init__(self, response_text: str):
        self._text = response_text
        self.create = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text=response_text)],
            usage=MagicMock(input_tokens=100, output_tokens=20, cache_creation_input_tokens=0, cache_read_input_tokens=0),
        ))


class FakeClient:
    def __init__(self, response_text: str):
        self.messages = FakeMessages(response_text)


def test_driver_returns_trial_record():
    async def _run():
        client = FakeClient("<skill>SKILL_000</skill>")
        driver = Driver(client=client, model="claude-sonnet-4-6", max_concurrency=1)
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
        assert rec.input_tokens == 100

    asyncio.run(_run())
