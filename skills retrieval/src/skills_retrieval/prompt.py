from __future__ import annotations

from .config import Representation
from .pool_builder import Pool


def render_pool_block(pool: Pool, representation: Representation = "card") -> str:
    lines: list[str] = []
    for did in pool.display_ids:
        card = pool.cards[did]
        if representation == "card":
            desc = card["description"].strip().replace("\n", " ")
            lines.append(f"{did}: {card['name']} — {desc}")
        elif representation == "name_only":
            lines.append(f"{did}: {card['name']}")
        elif representation == "desc_only":
            desc = card["description"].strip().replace("\n", " ")
            lines.append(f"{did}: {desc}")
        elif representation == "full":
            lines.append(f"{did}:\n{card['body']}\n---")
        else:
            raise NotImplementedError(f"representation {representation} not in Plan 1")
    return "\n".join(lines)


def render_awareness_prompt(task_instruction: str, pool: Pool) -> str:
    return _render("awareness", task_instruction, pool)


def render_selection_prompt(task_instruction: str, pool: Pool) -> str:
    return _render("selection", task_instruction, pool)


def _render(probe: str, task_instruction: str, pool: Pool) -> str:
    pool_block = render_pool_block(pool, representation=pool.spec.representation)
    response_instruction = (
        "  <skills>ID_1,ID_2,ID_3,ID_4,ID_5</skills>  # EXACTLY 5 skills, ordered from MOST to LEAST relevant"
        if probe == "awareness"
        else "  <skill>SKILL_ID</skill>              # single best skill for solving this task"
    )
    return (
        "You are a retrieval subject in a controlled study.\n"
        "\n"
        "Task:\n"
        f"{task_instruction}\n"
        "\n"
        f"Available skills ({len(pool.display_ids)}):\n"
        f"{pool_block}\n"
        "\n"
        "Respond with EXACTLY ONE of:\n"
        f"{response_instruction}\n"
        "\n"
        "No other text."
    )
