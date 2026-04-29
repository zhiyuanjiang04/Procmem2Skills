"""Build compact task cards for LLM clustering.

Each card has: task_id, one-sentence goal, top tools/libs from solution.sh,
and category. Goal extracted heuristically (first sentence of instruction).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
TASKS_JSONL = ROOT.parent / "data" / "tasks.jsonl"
SOLUTIONS_DIR = ROOT.parent.parent / "terminal-bench" / "original-tasks"
OUT_FILE = ROOT / "outputs" / "task_cards.jsonl"

# Tools/commands worth surfacing to the LLM. Detected via word-boundary regex.
TOOL_PATTERNS = {
    "python3", "python", "pip", "pip3", "uv",
    "openpyxl", "pandas", "numpy", "torch", "transformers", "sklearn",
    "huggingface_hub", "datasets",
    "git", "gh", "curl", "wget", "tar", "zip", "unzip", "7z",
    "qemu", "qemu-system-x86_64", "docker", "ssh", "scp",
    "gcc", "g++", "clang", "make", "cmake", "cargo", "rustc", "go",
    "sqlite", "sqlite3", "psql", "mysql", "duckdb", "parquet",
    "ffmpeg", "easyocr", "PIL", "pillow",
    "openssl", "hashcat", "john",
    "node", "npm", "bun", "yarn",
    "bash", "sh", "awk", "sed", "grep", "find",
    "jq", "yq",
    "vim", "nvim",
    "matplotlib", "scipy", "biopython", "openai",
    "stan", "rstan", "R", "Rscript",
    "fastext", "fasttext", "spacy", "nltk", "mteb",
    "celery", "redis", "nginx", "postgres",
    "regex", "re",
    "torch.distributed", "deepspeed",
    "mlflow", "wandb",
    "selenium", "beautifulsoup4", "requests",
    "ollama", "llama-cpp", "vllm",
}
# Pre-compile per-pattern (some have special chars)
_token_re = re.compile(r"\b([A-Za-z][A-Za-z0-9_+\-\.]*)\b")


def first_sentence(instr: str, limit: int = 240) -> str:
    instr = instr.strip()
    if not instr:
        return ""
    # crude: split on .!?\n followed by space, take first non-empty piece
    parts = re.split(r"(?<=[.!?])\s+|\n+", instr, maxsplit=4)
    sent = next((p.strip() for p in parts if p.strip()), instr[:limit])
    return sent[:limit]


def extract_tools(solution_text: str, max_tools: int = 8) -> list[str]:
    """Pick out tool names that appear in the solution and matter to clustering."""
    found = set()
    for m in _token_re.finditer(solution_text):
        tok = m.group(1)
        if tok in TOOL_PATTERNS:
            found.add(tok)
        elif tok.lower() in TOOL_PATTERNS:
            found.add(tok.lower())
    # Also catch python imports (`import X`, `from X import ...`)
    for m in re.finditer(r"\bimport\s+([a-zA-Z_][a-zA-Z0-9_\.]*)", solution_text):
        found.add(m.group(1).split(".")[0])
    for m in re.finditer(r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_\.]*)\s+import", solution_text):
        found.add(m.group(1).split(".")[0])
    # Drop obvious noise
    noise = {"if", "then", "fi", "do", "done", "for", "in", "as", "with", "is",
             "to", "the", "a", "an", "of", "or", "and", "not", "from", "self"}
    found = {t for t in found if t not in noise and len(t) > 1}
    # Prefer the canonical tool patterns first
    preferred = [t for t in TOOL_PATTERNS if t in found]
    extras = sorted(found - set(preferred))
    out = preferred + extras
    return out[:max_tools]


def load_solution(task_id: str) -> str:
    sol = SOLUTIONS_DIR / task_id / "solution.sh"
    if sol.exists():
        try:
            return sol.read_text(errors="replace")
        except Exception:
            return ""
    # fallback: maybe task uses a different solution layout
    sol_dir = SOLUTIONS_DIR / task_id
    if sol_dir.exists():
        # take the first .sh / .py we find
        for ext in ("*.sh", "*.py"):
            for f in sol_dir.glob(ext):
                try:
                    return f.read_text(errors="replace")
                except Exception:
                    continue
    return ""


def main():
    tasks = [json.loads(l) for l in open(TASKS_JSONL)]
    print(f"loaded {len(tasks)} tasks")

    cards = []
    no_solution = 0
    for t in tasks:
        tid = t["task_id"]
        sol = load_solution(tid)
        if not sol:
            no_solution += 1
        tools = extract_tools(sol) if sol else []
        cards.append({
            "task_id": tid,
            "goal": first_sentence(t.get("instruction", "")),
            "tools": tools,
            "category": t.get("category") or "?",
        })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"wrote {len(cards)} cards → {OUT_FILE}")
    print(f"  tasks with no solution.sh: {no_solution}")
    print("\nfirst 3 cards:")
    for c in cards[:3]:
        print(json.dumps(c, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
