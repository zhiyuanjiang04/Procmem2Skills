import asyncio
from unittest.mock import AsyncMock, MagicMock

from skills_retrieval.config import PoolSpec
from skills_retrieval.data import Corpus, Task
from skills_retrieval.pool_builder import build_pool
from skills_retrieval.prompt import render_awareness_prompt, render_pool_block
from skills_retrieval.driver import Driver
from skills_retrieval.metrics import score_trial


class FakeClient:
    def __init__(self, reply: str):
        self.messages = MagicMock()
        usage = MagicMock(input_tokens=100, output_tokens=10, cache_creation_input_tokens=0, cache_read_input_tokens=0)
        self.messages.create = AsyncMock(return_value=MagicMock(content=[MagicMock(text=reply)], usage=usage))


def test_pipeline_scores_correct_pick(tiny_corpus):
    corpus = Corpus.from_paths(tiny_corpus["metadata_path"], tiny_corpus["embeddings_path"])
    task = Task(task_id="t0", instruction="Pick alpha", gt_skill_names=["alpha"], gt_skill_bodies=["alpha body"], domain="test")
    gt_entries = [("gt_t0_alpha", "alpha", "alpha body", corpus.embeddings[3])]
    spec = PoolSpec(task_id="t0", strategy="random", n=5, seed=0)
    pool = build_pool(spec, task, corpus, task_embedding=corpus.embeddings[3], gt_entries=gt_entries)

    gt_display = pool.gt_display_ids[0]
    others = [d for d in pool.display_ids if d != gt_display][:4]
    reply = f"<skills>{gt_display},{others[0]},{others[1]},{others[2]},{others[3]}</skills>"

    driver = Driver(client=FakeClient(reply), model="claude-sonnet-4-6", max_concurrency=1)

    async def _go():
        rec = await driver.run_one(
            pool_id=spec.pool_id,
            probe="awareness",
            system_prompt="You are a retrieval subject.",
            pool_block=render_pool_block(pool, representation="card"),
            user_prompt=render_awareness_prompt(task.instruction, pool).split("Available skills")[0],
        )
        return rec

    rec = asyncio.run(_go())
    pool_map = {"id_map": pool.id_map, "gt_display_ids": pool.gt_display_ids}
    parsed = {"extracted_ids": rec.extracted_ids, "format_status": rec.format_status, "flags": rec.flags}
    scored = score_trial(parsed, pool_map, probe="awareness")
    assert scored["awareness_top1"] == 1
    assert scored["awareness_mrr"] == 1.0
    assert scored["awareness_recall5"] == 1
