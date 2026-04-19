from __future__ import annotations

import re
from typing import Literal

_SELECTION_RE = re.compile(r"<skill>([^<]+)</skill>", re.IGNORECASE)
_AWARENESS_RE = re.compile(r"<skills>([^<]+)</skills>", re.IGNORECASE)


def parse_response(raw: str, probe: Literal["awareness", "selection"]) -> dict:
    pattern = _AWARENESS_RE if probe == "awareness" else _SELECTION_RE
    matches = pattern.findall(raw)
    flags: dict = {}
    if not matches:
        return {"extracted_ids": [], "format_status": "fail", "flags": {"parse_fail": True}, "raw_text": raw}

    first_inner = matches[0]
    if len(matches) > 1:
        flags["multiple_tags"] = True

    ids = [tok.strip() for tok in first_inner.split(",") if tok.strip()]

    if probe == "awareness":
        deduped: list[str] = []
        seen: set[str] = set()
        for i in ids:
            if i in seen:
                flags["dup_violation"] = True
                continue
            seen.add(i)
            deduped.append(i)
        ids = deduped
        if len(ids) < 5:
            flags["length_violation"] = True
        elif len(ids) > 5:
            flags["length_violation"] = True
            ids = ids[:5]
    else:
        ids = ids[:1]

    is_clean = not flags and _is_exact_match(raw, probe)
    format_status = "clean" if is_clean else "warning"
    return {"extracted_ids": ids, "format_status": format_status, "flags": flags, "raw_text": raw}


def _is_exact_match(raw: str, probe: str) -> bool:
    stripped = raw.strip()
    if probe == "selection":
        return bool(re.fullmatch(r"<skill>[^<]+</skill>", stripped, re.IGNORECASE))
    return bool(re.fullmatch(r"<skills>[^<]+</skills>", stripped, re.IGNORECASE))
