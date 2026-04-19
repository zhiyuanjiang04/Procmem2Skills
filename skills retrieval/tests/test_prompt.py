from skills_retrieval.config import PoolSpec
from skills_retrieval.pool_builder import Pool
from skills_retrieval.prompt import render_awareness_prompt, render_selection_prompt, render_pool_block


def _pool():
    spec = PoolSpec(task_id="t0", strategy="random", n=3, seed=0)
    return Pool(
        spec=spec,
        display_ids=["SKILL_000", "SKILL_001", "SKILL_002"],
        id_map={"SKILL_000": "gt_t0_alpha", "SKILL_001": "skill_000005", "SKILL_002": "skill_000002"},
        cards={
            "SKILL_000": {"name": "alpha", "description": "Test alpha skill.", "body": "alpha body"},
            "SKILL_001": {"name": "beta", "description": "Test beta skill.", "body": "beta body"},
            "SKILL_002": {"name": "gamma", "description": "Test gamma skill.", "body": "gamma body"},
        },
        gt_display_ids=["SKILL_000"],
    )


def test_awareness_prompt_mentions_five_and_order(tiny_task):
    prompt = render_awareness_prompt(task_instruction=tiny_task["instruction"], pool=_pool())
    assert "EXACTLY 5" in prompt or "exactly 5" in prompt.lower()
    assert "ordered from" in prompt.lower()
    assert "SKILL_000" in prompt
    assert "alpha" in prompt


def test_selection_prompt_has_single_tag_instruction(tiny_task):
    prompt = render_selection_prompt(task_instruction=tiny_task["instruction"], pool=_pool())
    assert "<skill>" in prompt
    assert "SKILL_000" in prompt


def test_card_block_uses_one_line_per_skill():
    pool = _pool()
    block = render_pool_block(pool, representation="card")
    assert block.count("\n") >= 2
    assert "SKILL_000: alpha" in block or "SKILL_000:" in block
