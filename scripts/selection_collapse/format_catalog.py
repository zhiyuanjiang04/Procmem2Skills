"""Format skill pools into text catalogs for LLM context injection."""


def count_tokens_approx(text: str) -> int:
    """Approximate token count (~1 token per 4 chars)."""
    return max(1, len(text) // 4)


def format_skill_entry(skill: dict, max_tokens: int = 200) -> str:
    """Format one skill as a labeled entry, truncating body to max_tokens."""
    display_id = skill.get("display_id", "SKILL_???")
    name = skill.get("name", "unnamed")
    body = skill.get("description") or skill.get("content", "")

    # Truncate body to fit within max_tokens
    # Use both char-based (~4 chars/token) and word-based (~0.75 words/token)
    # caps to handle varied token density
    max_chars = max_tokens * 4
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + "..."
    words = body.split()
    max_words = int(max_tokens * 1.1)  # slight margin over 1:1 word:token
    if len(words) > max_words:
        body = " ".join(words[:max_words]) + "..."

    header = f"[{display_id}] Name: {name}"
    return f"{header}\n{body}"


def format_catalog(pool: list, max_tokens_per_skill: int = 200) -> str:
    """Format entire skill pool, entries separated by \\n---\\n."""
    entries = [format_skill_entry(s, max_tokens=max_tokens_per_skill) for s in pool]
    return "\n---\n".join(entries)


def format_awareness_prompt(task_instruction: str, catalog: str) -> str:
    """Format a prompt asking the LLM to identify the 5 most relevant skills."""
    return (
        "Given the following task and skill catalog, list the 5 skills that would "
        "most likely lead to successful completion of the task. For each, provide "
        "the skill_id and a one-sentence justification of why it would help.\n"
        "\n"
        f"Task: {task_instruction}\n"
        "\n"
        "Skills:\n"
        f"{catalog}\n"
        "\n"
        "Respond in this exact format:\n"
        "1. SKILL_XXX - justification\n"
        "2. SKILL_XXX - justification\n"
        "3. SKILL_XXX - justification\n"
        "4. SKILL_XXX - justification\n"
        "5. SKILL_XXX - justification"
    )


def format_selection_prompt(task_instruction: str, catalog: str) -> str:
    """Format a prompt for skill-augmented task completion."""
    return (
        "You have access to the skills listed below. Use whichever skill(s) are "
        "most relevant to complete the following task. If you use a skill, "
        "reference it by its ID (e.g., SKILL_001).\n"
        "\n"
        "Skills:\n"
        f"{catalog}\n"
        "\n"
        "---\n"
        "\n"
        f"Task: {task_instruction}"
    )


def catalog_token_budget(pool: list, max_tokens_per_skill: int = 200) -> dict:
    """Return token budget statistics for a skill catalog."""
    catalog = format_catalog(pool, max_tokens_per_skill=max_tokens_per_skill)
    total_tokens = count_tokens_approx(catalog)
    n_skills = len(pool)
    return {
        "n_skills": n_skills,
        "total_tokens_approx": total_tokens,
        "tokens_per_skill_avg": total_tokens / n_skills if n_skills > 0 else 0,
        "catalog_length_chars": len(catalog),
    }
