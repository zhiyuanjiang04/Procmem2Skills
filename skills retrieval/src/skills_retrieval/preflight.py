from __future__ import annotations


def count_tokens_approx(text: str) -> int:
    return max(1, len(text) // 4)


def will_fit(prompt: str, model_context_limit: int, safety_margin: int = 1000) -> bool:
    return count_tokens_approx(prompt) + safety_margin < model_context_limit
