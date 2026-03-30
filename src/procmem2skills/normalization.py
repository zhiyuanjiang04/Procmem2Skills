from __future__ import annotations

import html
import re
import shlex
from difflib import SequenceMatcher
from pathlib import PurePosixPath

from procmem2skills.models import WorkflowStep

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_TEXT_NODE_RE = re.compile(r"<text[^>]*>(.*?)</text>", re.I | re.S)
_HTML_ATTR_RE = re.compile(r"(?:id|name|aria[_-]label|placeholder|title|alt)=['\"]([^'\"]+)['\"]", re.I)
_HTML_ATTR_ASSIGN_RE = re.compile(r"\b[\w:-]+=([\"']).*?\1|\b[\w:-]+=[^\s>]+")
_URL_RE = re.compile(r"https?://\S+")
_PATH_RE = re.compile(r"(?:[a-zA-Z]:)?(?:/[\w.\-]+)+")
_NUMBER_RE = re.compile(r"\b\d+\b")
_BROWSER_LOW_VALUE_PHRASES = {
    "skip to main content",
    "skip to content",
    "main content",
    "open menu",
    "toggle menu open",
    "menu",
    "search",
    "tock home page",
}
_BROWSER_STOPWORDS = {
    "a",
    "an",
    "all",
    "and",
    "at",
    "for",
    "from",
    "has",
    "here",
    "in",
    "is",
    "it",
    "more",
    "new",
    "now",
    "of",
    "on",
    "open",
    "that",
    "the",
    "to",
    "use",
    "with",
    "your",
}
_BROWSER_ACTION_VERBS = {
    "click",
    "choose",
    "enter",
    "fill",
    "go",
    "goto",
    "input",
    "navigate",
    "open",
    "press",
    "search",
    "select",
    "tap",
    "type",
}
_TERMINAL_SUBCOMMAND_HEADS = {
    "apt",
    "apt-get",
    "brew",
    "cargo",
    "git",
    "npm",
    "pip",
    "pip3",
    "pnpm",
    "poetry",
    "uv",
    "yarn",
}


def compact_text(text: str | None, limit: int = 120) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = compact_text(text, limit=400).lower()
    text = _URL_RE.sub("<url>", text)
    text = _PATH_RE.sub("<path>", text)
    text = _NUMBER_RE.sub("<num>", text)
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "<str>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def operation_family(tool: str | None, operation: str | None) -> str:
    tool_key = _slug(tool or "tool")
    raw_operation = operation or ""
    operation_text = compact_text(raw_operation, limit=400)
    if not raw_operation and not operation_text:
        return f"{tool_key}-step"
    if tool_key == "browser":
        return f"{tool_key}-{_browser_family(operation_text)}"
    if tool_key == "terminal":
        return f"{tool_key}-{_terminal_family(raw_operation)}"
    if tool_key == "text-world":
        return f"{tool_key}-{_text_world_family(operation_text)}"
    return f"{tool_key}-{_generic_family(operation_text)}"


def trigger_phrase(step: WorkflowStep) -> str:
    family = operation_family(step.tool, step.operation)
    fallback = _trigger_from_family(family)
    intent = compact_text(step.intent, limit=120)
    if _is_machine_like_intent(intent, step.operation):
        return fallback
    if not intent:
        return fallback
    if intent[0].isupper():
        intent = intent[0].lower() + intent[1:]
    return f"When the agent needs to {intent.rstrip('.')}."


def generic_trigger_phrase(tool: str | None, operation: str | None) -> str:
    return _trigger_from_family(operation_family(tool, operation))


def summarize_observation(tool: str | None, summary: str | None, text: str | None) -> list[str]:
    tool_key = _slug(tool or "tool")
    conditions = []
    summary_text = compact_text(summary, limit=80)
    raw_text = compact_text(text, limit=160)

    if tool_key == "browser":
        hint = _browser_hint(summary_text, text or "")
        if hint:
            conditions.append(f"Page shows: {hint}")
        if summary_text and _is_state_like_browser_summary(summary_text) and not _is_redundant(summary_text, hint):
            conditions.append(f"Current goal state: {summary_text}")
        return _dedupe(conditions)

    if tool_key == "terminal":
        cwd_hint = _terminal_cwd_hint(text)
        if cwd_hint:
            conditions.append(f"Working directory: {cwd_hint}")
        if summary_text and "execute `" not in summary_text.lower():
            conditions.append(f"Task context: {summary_text}")
        return _dedupe(conditions)

    if tool_key == "text-world":
        hint = summary_text or _first_sentence(raw_text)
        if hint:
            conditions.append(f"Scene state: {hint}")
        return _dedupe(conditions)

    hint = summary_text or raw_text
    if hint:
        conditions.append(f"Context: {hint}")
    return _dedupe(conditions)


def sequence_similarity(left: list[str], right: list[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(a=left, b=right).ratio()


def condition_key(text: str) -> str:
    return normalize_text(text)


def _browser_family(operation: str) -> str:
    name = _operation_name(operation)
    mapping = {
        "click": "click",
        "tap": "click",
        "press": "click",
        "type": "type",
        "input": "type",
        "fill": "type",
        "select": "select",
        "choose": "select",
        "navigate": "navigate",
        "open": "navigate",
        "goto": "navigate",
    }
    return mapping.get(name, name or "action")


def _terminal_family(operation: str) -> str:
    command = _extract_named_argument(operation, "command") or operation
    command = _primary_terminal_command(command)
    tokens = _shell_tokens(command)
    if not tokens:
        return _generic_family(operation)
    tokens = _terminal_focus_tokens(tokens)
    if not tokens:
        return _generic_family(operation)
    head = _slug(tokens[0])
    if head == "pip3":
        head = "pip"
    if head in {"python", "python3"}:
        if len(tokens) >= 2 and tokens[1] == "-c":
            return "python-inline"
        if len(tokens) >= 3 and tokens[1] == "-m":
            return f"python-module-{_slug(tokens[2])}"
        return head
    subcommand = _terminal_subcommand(head, tokens[1:])
    if subcommand:
        return f"{head}-{subcommand}"
    return head


def _text_world_family(operation: str) -> str:
    name = _operation_name(operation)
    mapping = {
        "go": "move",
        "move": "move",
        "walk": "move",
        "take": "take",
        "pick": "take",
        "get": "take",
        "put": "put",
        "place": "put",
        "drop": "put",
        "look": "look",
        "examine": "look",
        "inspect": "look",
        "clean": "clean",
        "wash": "clean",
        "heat": "heat",
        "cool": "cool",
        "open": "open",
        "close": "close",
        "toggle": "toggle",
        "use": "use",
    }
    return mapping.get(name, name or "action")


def _generic_family(operation: str) -> str:
    return _operation_name(operation) or "step"


def _operation_name(operation: str) -> str:
    operation = compact_text(operation, limit=200)
    match = re.match(r"\s*([a-zA-Z0-9_-]+)", operation)
    return _slug(match.group(1) if match else operation)


def _trigger_from_family(family: str) -> str:
    if family == "browser-click":
        return "When the agent needs to click a relevant page element."
    if family == "browser-type":
        return "When the agent needs to type into a page input."
    if family == "browser-select":
        return "When the agent needs to choose an option from a page control."
    if family == "browser-navigate":
        return "When the agent needs to navigate to a target page."
    if family == "text-world-move":
        return "When the agent needs to move to a relevant location."
    if family == "text-world-take":
        return "When the agent needs to pick up a relevant object."
    if family == "text-world-put":
        return "When the agent needs to place an object in the target receptacle."
    if family == "text-world-look":
        return "When the agent needs to inspect the scene or an object."
    if family == "text-world-clean":
        return "When the agent needs to clean the target object."
    if family.startswith("terminal-"):
        return _terminal_trigger_from_family(family)
    return "When the agent needs to complete this reusable subtask."


def _is_machine_like_intent(intent: str, operation: str | None) -> bool:
    if not intent:
        return True
    lowered = intent.lower()
    if re.search(r"[a-z][A-Z]|[A-Z]{3,}", intent):
        return True
    if "(" in intent or ")" in intent:
        return True
    operation_name = _operation_name(operation or "")
    return lowered == operation_name or lowered.startswith(f"{operation_name} ")


def _browser_hint(summary: str, text: str) -> str:
    best_candidate = ""
    best_score = float("-inf")
    for candidate in _browser_text_candidates(text):
        score = _browser_candidate_score(candidate)
        if score > best_score:
            best_candidate = candidate
            best_score = score
    if best_candidate and best_score >= 2.0:
        return best_candidate
    if _is_state_like_browser_summary(summary) and _is_informative_browser_text(summary):
        return summary
    return ""


def _terminal_cwd_hint(text: str | None) -> str:
    compact = compact_text(text, limit=120)
    if not compact:
        return ""
    try:
        path = PurePosixPath(compact)
    except Exception:
        return compact
    if compact.startswith("/"):
        tail = path.parts[-2:] if len(path.parts) >= 2 else path.parts
        return "/".join(tail) or compact
    return compact


def _extract_named_argument(operation: str, name: str) -> str:
    marker = f"{name}="
    start = operation.find(marker)
    if start < 0:
        return ""
    value = operation[start + len(marker) :].strip()
    if value.endswith(")"):
        value = value[:-1]
    return value.strip()


def _shell_tokens(command: str) -> list[str]:
    command = compact_text(command, limit=300)
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _primary_terminal_command(command: str) -> str:
    command = str(command).replace("\\n", "\n").replace("\\t", "\t")
    lines = [line.strip() for line in command.splitlines() if line.strip()]
    executable = [line for line in lines if not line.startswith("#")]
    if executable:
        return executable[0]
    return command


def _terminal_focus_tokens(tokens: list[str]) -> list[str]:
    focused = list(tokens)
    while focused:
        while focused and (_looks_like_env_assignment(focused[0]) or focused[0] == "env"):
            focused = focused[1:]
        if len(focused) >= 3 and focused[0] == "cd" and focused[2] in {"&&", ";"}:
            focused = focused[3:]
            continue
        return focused
    return focused


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return compact_text(parts[0], limit=80)


def _is_redundant(left: str, right: str) -> bool:
    return bool(left and right and normalize_text(left) == normalize_text(right))


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if not item:
            continue
        key = normalize_text(item)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def _slug(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "step"


def _terminal_subcommand(head: str, tokens: list[str]) -> str:
    if head not in _TERMINAL_SUBCOMMAND_HEADS:
        return ""
    for token in tokens:
        if token in {"|", "||", "&&", ";"}:
            break
        if not token or token.startswith("-"):
            continue
        candidate = token.strip()
        if (
            " " in candidate
            or any(marker in candidate for marker in ("/", "\\", ".", "=", ":", "@", "*", "(", ")", "[", "]", "{", "}"))
            or any(char.isdigit() for char in candidate)
        ):
            return ""
        slug = _slug(candidate)
        if slug and slug != "step":
            return slug
        return ""
    return ""


def is_informative_signal(text: str | None) -> bool:
    compact = compact_text(text, limit=120)
    if not compact:
        return False
    if compact.startswith("Page shows: "):
        return _is_informative_browser_text(compact.split(": ", 1)[1])
    if compact.startswith("Current goal state: "):
        return _is_state_like_browser_summary(compact.split(": ", 1)[1])
    return True


def _browser_text_candidates(text: str) -> list[str]:
    raw = html.unescape(text or "")
    candidates = []

    for match in _HTML_TEXT_NODE_RE.finditer(raw):
        _append_browser_candidate(candidates, match.group(1))
        if len(candidates) >= 64:
            break

    for value in _HTML_ATTR_RE.findall(raw):
        _append_browser_candidate(candidates, value)
        if len(candidates) >= 96:
            break

    if not candidates:
        visible = _HTML_TAG_RE.sub("\n", raw)
        visible = re.sub(r"</?[\w:-]+", "\n", visible)
        visible = _HTML_ATTR_ASSIGN_RE.sub(" ", visible)
        visible = re.sub(r"[<>{}=|/]+", " ", visible)
        visible = re.sub(
            r"\b(?:backend_node_id|node_id|backend|selector|aria|role|tag|type|class|value|name|id)\b",
            " ",
            visible,
        )
        for part in re.split(r"(?:\n|\t| {2,}|\s\|\s)", visible):
            _append_browser_candidate(candidates, part)
            if len(candidates) >= 64:
                break

    return _dedupe(candidates)


def _append_browser_candidate(candidates: list[str], value: str) -> None:
    cleaned = compact_text(" ".join(str(value).split()), limit=80)
    cleaned = cleaned.strip(" -:;,.|/")
    if cleaned:
        candidates.append(cleaned)


def _browser_candidate_score(text: str) -> float:
    compact = compact_text(text, limit=120)
    lowered = compact.lower()
    if not compact or _looks_machine_generated_clue(compact):
        return -10.0
    if lowered in _BROWSER_LOW_VALUE_PHRASES or lowered.startswith("skip to main content"):
        return -10.0

    tokens = re.findall(r"[a-z0-9]+", lowered)
    alpha_tokens = [token for token in tokens if re.search(r"[a-z]", token)]
    numeric_tokens = [token for token in tokens if token.isdigit()]
    if not alpha_tokens:
        return -10.0
    if len(numeric_tokens) > len(alpha_tokens):
        return -4.0

    content_tokens = [token for token in alpha_tokens if token not in _BROWSER_STOPWORDS]
    if not content_tokens:
        return -4.0

    score = float(len(set(content_tokens)) * 2)
    score -= max(0, len(alpha_tokens) - 4) * 1.5
    score -= (len(alpha_tokens) - len(content_tokens)) * 0.75
    score -= len(numeric_tokens) * 2.0
    if any(char in compact for char in ".?!:"):
        score -= 0.5
    if len(compact) > 48:
        score -= 1.0
    return score


def _is_informative_browser_text(text: str) -> bool:
    return _browser_candidate_score(text) >= 2.0


def _is_state_like_browser_summary(summary: str) -> bool:
    compact = compact_text(summary, limit=120)
    if not compact or _looks_machine_generated_clue(compact):
        return False
    first_word = _slug(compact.split(" ", 1)[0])
    return first_word not in _BROWSER_ACTION_VERBS


def _looks_machine_generated_clue(text: str) -> bool:
    compact = compact_text(text, limit=120).lower()
    if not compact:
        return True
    if compact in {"...", "<", ">"}:
        return True
    if not re.search(r"[a-z]", compact):
        return True
    tokens = re.findall(r"[a-z0-9]+", compact)
    alpha_tokens = [token for token in tokens if re.search(r"[a-z]", token)]
    if not alpha_tokens:
        return True
    noisy = {"backend", "node", "ctl", "content", "main"}
    return len(alpha_tokens) <= 2 and any(token in noisy or token.startswith("ctl") for token in alpha_tokens)


def _terminal_trigger_from_family(family: str) -> str:
    action = family.split("-", 1)[1] if "-" in family else family
    semantic_templates = {
        "cd": "When the agent needs to move into a relevant working directory from the terminal.",
        "ls": "When the agent needs to inspect the filesystem from the terminal.",
        "find": "When the agent needs to locate relevant files or paths from the terminal.",
        "cat": "When the agent needs to inspect file contents from the terminal.",
        "head": "When the agent needs to inspect the beginning of a file or command output from the terminal.",
        "echo": "When the agent needs to emit or append a short text value from the terminal.",
        "grep": "When the agent needs to search files or code for a relevant pattern from the terminal.",
        "sed": "When the agent needs to apply a targeted text edit from the terminal.",
        "git-clone": "When the agent needs to clone the target repository from the terminal.",
        "pip3": "When the agent needs to inspect or manage Python packages from the terminal.",
        "pip-install": "When the agent needs to install a required Python dependency from the terminal.",
        "pip3-install": "When the agent needs to install a required Python dependency from the terminal.",
        "pip-uninstall": "When the agent needs to remove a conflicting Python dependency from the terminal.",
        "pip3-uninstall": "When the agent needs to remove a conflicting Python dependency from the terminal.",
        "python": "When the agent needs to run a Python script or module from the terminal.",
        "python3": "When the agent needs to run a Python script or module from the terminal.",
        "python-inline": "When the agent needs to run a short Python snippet from the terminal.",
        "python-module-pytest": "When the agent needs to run the test suite from the terminal.",
        "pytest": "When the agent needs to run the test suite from the terminal.",
    }
    if action in semantic_templates:
        return semantic_templates[action]
    command_family = action.replace("-", " ")
    return f"When the agent needs to use `{command_family}` from the terminal."


def _looks_like_env_assignment(token: str) -> bool:
    if not token or token.startswith("-"):
        return False
    name, sep, _ = token.partition("=")
    return bool(sep and name.replace("_", "").isalnum())
