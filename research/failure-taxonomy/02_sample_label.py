"""Stage 2: stratified sample + LLM raw-label each trial.

Inputs:
  outputs/manifest.jsonl  (from 01_build_manifest.py)

Process:
  1. Stratified sample N trials across (benchmark, setting, arm, outcome).
  2. For each sampled trial, load codex.txt + instruction.md + SKILL.md (if skill arm)
     into a single prompt.
  3. Call Claude Sonnet 4.6 via `claude -p` headless OAuth (no API credit).
  4. Parse JSON, cache each labeled trial to disk.

Outputs:
  outputs/labels_raw.jsonl  (one record per sampled trial with LLM labels)

Env vars:
  PARALLEL=N      number of concurrent Claude calls (default 1)
  SAMPLE_N=N      total sample target (default 240)
  CELL_MIN=N      min samples per (bench, setting, arm, outcome) cell (default 3)
  SEED=N          random seed for reproducibility (default 0)
"""
import json
import os
import random
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# manifest stores paths relative to /Users/hudx/Desktop (ROOT.parent.parent)
DATA_BASE = ROOT.parent.parent
MANIFEST = ROOT / "outputs" / "manifest.jsonl"
OUT = ROOT / "outputs" / "labels_raw.jsonl"
CACHE = ROOT / "outputs" / "_label_cache"

MODEL = "claude-sonnet-4-6"
TIMEOUT = 600

# How much of each evidence file to include in the prompt.
INSTRUCTION_MAX = 3000        # chars
SKILL_MAX = 3000
CODEX_HEAD = 6000             # opening of trace
CODEX_TAIL = 12000            # last steps + error message (more useful)


SYSTEM_AND_INSTRUCTIONS = """\
You are labeling agent trajectories from a controlled study comparing three
conditions on a software-engineering benchmark:
  - raw:       no procedural memory, no skill injected
  - workflow:  past workflow memories appended to instruction
  - skill:     a curated SKILL.md injected into the agent's environment

The trial outcome is given (success/failure with a reward). Your job is to
explain WHY the trial ended that way, using the trajectory and (when relevant)
the candidate skill or injected workflow. Be concrete; cite tool calls or
output snippets.

Output STRICT JSON only (no preamble, no markdown fences) with this schema:
{
  "freeform_reasoning_short": "1-2 sentences summarizing what happened",
  "primary_mode_candidate": "short label, free text, e.g. 'missing python dependency' or 'agent looped on same failing test'",
  "secondary_factors": ["short label", "..."],
  "evidence_spans": [
    {"source": "codex.txt" | "result.json" | "instruction.md" | "SKILL.md",
     "quote": "verbatim snippet, <=160 chars"}
  ],
  "skill_effect_judgment": "helps" | "neutral" | "hurts" | "not_applicable",
  "skill_effect_reason": "one short sentence; for arm != skill use not_applicable + ''",
  "capability_vs_knowledge": "knowledge_missing" | "knowledge_present_but_misused" | "capability_limit" | "environmental"
}

Rules:
  - For arm=raw and arm=workflow trials, skill_effect_judgment MUST be "not_applicable".
  - "primary_mode_candidate" should be a noun phrase, not a sentence.
  - Always provide at least one evidence_span quoting the trajectory or result.
  - If trial succeeded, primary_mode_candidate should describe the success path
    (e.g. "clean python implementation passed all tests") and capability_vs_knowledge
    can be "knowledge_present_but_misused" only if the agent recovered from misuse.
"""


def tail(text: str, n_chars: int) -> str:
    if len(text) <= n_chars:
        return text
    return "...[truncated]...\n" + text[-n_chars:]


def head(text: str, n_chars: int) -> str:
    if len(text) <= n_chars:
        return text
    return text[:n_chars] + "\n...[truncated]..."


def head_tail(text: str, head_n: int, tail_n: int) -> str:
    if len(text) <= head_n + tail_n:
        return text
    return text[:head_n] + "\n...[truncated middle]...\n" + text[-tail_n:]


def read_file(rel_path: str | None, max_chars: int = 0) -> str:
    if not rel_path:
        return ""
    p = DATA_BASE / rel_path
    if not p.exists():
        return ""
    try:
        s = p.read_text(errors="replace")
    except Exception:
        return ""
    if max_chars and len(s) > max_chars:
        return s[:max_chars] + "\n...[truncated]..."
    return s


def build_prompt(rec: dict) -> str:
    instruction = read_file(rec.get("instruction_path"), INSTRUCTION_MAX)
    skill = read_file(rec.get("skill_path"), SKILL_MAX) if rec["arm"] == "skill" else ""
    codex = read_file(rec.get("codex_path"))
    codex_excerpt = head_tail(codex, CODEX_HEAD, CODEX_TAIL) if codex else "(no codex.txt)"

    parts = [SYSTEM_AND_INSTRUCTIONS, "", "--- TRIAL METADATA ---"]
    parts.append(json.dumps({
        "task_name": rec["task_name"],
        "benchmark": rec["benchmark"],
        "setting": rec["setting"],
        "arm": rec["arm"],
        "status": rec["status"],
        "reward": rec["reward"],
        "exception_type": rec.get("exception_type"),
        "duration_sec": rec.get("duration_sec"),
    }, ensure_ascii=False, indent=2))
    parts.append("")
    parts.append("--- TASK INSTRUCTION (instruction.md, may include injected workflow if arm=workflow) ---")
    parts.append(instruction or "(no instruction.md available)")
    if rec["arm"] == "skill":
        parts.append("")
        parts.append("--- INJECTED SKILL (SKILL.md) ---")
        parts.append(skill or "(no SKILL.md available)")
    parts.append("")
    parts.append("--- TRAJECTORY (agent/codex.txt, head + tail) ---")
    parts.append(codex_excerpt)
    parts.append("")
    parts.append("Output JSON only:")
    return "\n".join(parts)


def call_claude(prompt: str, timeout: int = TIMEOUT) -> tuple[str, dict]:
    proc = subprocess.run(
        ["claude", "-p",
         "--model", MODEL,
         "--output-format", "json",
         "--tools", "",
         "--no-session-persistence"],
        input=prompt,
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed (rc={proc.returncode}):\n{proc.stderr[:1500]}")
    payload = json.loads(proc.stdout)
    if payload.get("is_error"):
        raise RuntimeError(f"claude returned error: {payload}")
    return payload["result"], payload


def parse_response(raw: str) -> dict:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[len("json"):].lstrip("\n")
        s = s.rsplit("```", 1)[0]
    return json.loads(s)


def stratified_sample(records: list[dict], target_n: int, cell_min: int, seed: int) -> list[dict]:
    """Sample at least cell_min per cell, then fill remainder uniformly."""
    rng = random.Random(seed)
    cells = defaultdict(list)
    for r in records:
        if not r.get("codex_path"):
            continue  # need trajectory to label
        cell = (r["benchmark"], r["setting"], r["arm"], r["status"])
        cells[cell].append(r)

    # Shuffle each cell
    for k in cells:
        rng.shuffle(cells[k])

    chosen: list[dict] = []
    # Round 1: cell_min per cell
    for k, lst in cells.items():
        take = min(cell_min, len(lst))
        chosen.extend(lst[:take])
        cells[k] = lst[take:]

    print(f"  after cell_min={cell_min}: {len(chosen)} sampled across {len(cells)} cells")

    if len(chosen) >= target_n:
        return chosen[:target_n]

    # Round 2: round-robin from remaining
    remaining = sum(len(v) for v in cells.values())
    cell_keys = list(cells.keys())
    rng.shuffle(cell_keys)
    i = 0
    while len(chosen) < target_n and remaining > 0:
        k = cell_keys[i % len(cell_keys)]
        if cells[k]:
            chosen.append(cells[k].pop(0))
            remaining -= 1
        i += 1
    return chosen


def label_one(rec: dict) -> dict:
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", rec.get("trial_id") or rec.get("trial_name", "anon"))
    cache_file = CACHE / f"{safe_id}.json"
    if cache_file.exists():
        return {**json.loads(cache_file.read_text()), "_cached": True}

    prompt = build_prompt(rec)
    raw, meta = call_claude(prompt)
    parsed = parse_response(raw)
    out = {
        "trial_id": rec.get("trial_id"),
        "task_name": rec["task_name"],
        "benchmark": rec["benchmark"],
        "setting": rec["setting"],
        "arm": rec["arm"],
        "status": rec["status"],
        "reward": rec["reward"],
        "labels": parsed,
        "_meta": {
            "duration_ms": meta.get("duration_ms"),
            "cost_usd": meta.get("total_cost_usd"),
            "prompt_chars": len(prompt),
        },
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    parallel = int(os.environ.get("PARALLEL", "1"))
    target_n = int(os.environ.get("SAMPLE_N", "240"))
    cell_min = int(os.environ.get("CELL_MIN", "3"))
    seed = int(os.environ.get("SEED", "0"))

    records = [json.loads(l) for l in open(MANIFEST)]
    print(f"loaded {len(records)} manifest records")
    print(f"sampling target={target_n}, cell_min={cell_min}, seed={seed}")

    sampled = stratified_sample(records, target_n, cell_min, seed)
    print(f"sampled {len(sampled)} trials\n")

    def task_fn(idx_rec):
        idx, rec = idx_rec
        try:
            out = label_one(rec)
            return idx, rec, out, "ok"
        except Exception as e:
            return idx, rec, {"error": str(e), "trial_id": rec.get("trial_id")}, "error"

    results: list[dict | None] = [None] * len(sampled)
    n_done = 0
    n_cached = 0
    n_error = 0
    total_cost = 0.0
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(task_fn, (i, r)) for i, r in enumerate(sampled)]
        for f in as_completed(futures):
            idx, rec, out, status = f.result()
            results[idx] = out
            n_done += 1
            if status == "error":
                n_error += 1
                print(f"[{n_done}/{len(sampled)}] {rec['task_name'][:40]} arm={rec['arm']} ERROR: {out.get('error', '')[:120]}",
                      flush=True)
            else:
                if out.get("_cached"):
                    n_cached += 1
                cost = (out.get("_meta") or {}).get("cost_usd") or 0
                total_cost += cost
                tag = "[CACHED]" if out.get("_cached") else ""
                primary = (out.get("labels") or {}).get("primary_mode_candidate", "?")
                print(f"[{n_done}/{len(sampled)}] {rec['task_name'][:30]} arm={rec['arm']:<8} status={rec['status']} {tag} → {primary[:60]} cost=${cost:.3f}",
                      flush=True)

    with open(OUT, "w") as f:
        for r in results:
            if r is not None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nsaved {len(results)} labels → {OUT}")
    print(f"  cached: {n_cached}, errors: {n_error}")
    print(f"  total cost: ${total_cost:.3f}")


if __name__ == "__main__":
    main()
