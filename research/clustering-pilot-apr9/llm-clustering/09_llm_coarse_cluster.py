"""First-tier coarse clustering: process tasks in batches (~80 each), then
merge candidate clusters in a final small call.

Single-batch approach hangs/times-out on 241-task prompts via Claude Code OAuth
(prompt cache misses + slow OAuth path). Batching keeps each call to ~5k tokens.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
CARDS_FILE = ROOT / "outputs" / "task_cards.jsonl"
OUT_FILE = ROOT / "outputs" / "llm_coarse.json"
BATCH_CACHE_DIR = ROOT / "outputs" / "_coarse_batches"

MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 60          # tasks per call
TIMEOUT = 600            # seconds per call
DO_MERGE = False         # LLM merge times out at 48k char prompts; concat instead


PROMPT_HEADER = """\
You are an expert at organizing software-engineering tasks into reusable-skill groups.

I have {n_tasks} terminal-bench tasks (this is batch {batch_idx} of {n_batches}).
Each task is described by:
  ID:       short slug
  Goal:     one-sentence summary
  Tools:    main commands/libraries used in the oracle solution
  Category: high-level label (only weakly informative — many "software-engineering"
            tasks span very different actual skills)

Group these tasks so that **all tasks in one group can plausibly be solved by a single
reusable skill**. A reusable skill is a structured procedure (steps, preconditions,
common failure modes) that an agent can follow. Two tasks belong together when:
  - their solutions follow the same procedure shape, AND
  - they use overlapping tools/libraries, AND
  - the skill that solves one would actually help on the other (not just be loosely related)

Constraints:
  - Aim for 8 to 15 groups in this batch
  - Each group has 2 to 12 tasks
  - Tasks that don't clearly belong anywhere go into "unclustered" (do NOT force them into a group)
  - Each group must have:
      * a concrete `skill_concept` (one sentence describing the underlying skill)
      * a `reasoning` (why these tasks share that skill)
  - Avoid generic categories like "uses Python" or "involves files" — be specific about the procedure

Output STRICT JSON only (no preamble, no markdown fences):
{{
  "clusters": [
    {{
      "id": "B{batch_idx}_C1",
      "skill_concept": "...",
      "member_ids": ["task_a", "task_b", ...],
      "reasoning": "..."
    }}
  ],
  "unclustered": ["task_x", "task_y", ...]
}}

Every task MUST appear exactly once across clusters + unclustered.

Tasks:
"""


MERGE_PROMPT_HEADER = """\
You have several candidate cluster lists from different batches of the same dataset.
Some clusters across batches likely describe the SAME underlying skill and should be merged.
Other batch-level clusters might be too narrow and should stay distinct.

Your job: produce a final consolidated set of clusters by merging the batch clusters
that share the same skill, while keeping batch clusters that are genuinely distinct as separate.

Rules:
  - Aim for 20 to 35 final clusters total
  - Each final cluster has 2 to 15 tasks
  - Each final cluster must have:
      * `skill_concept`
      * `member_ids` (combined membership)
      * `reasoning`
      * `merged_from`: list of batch cluster ids that were merged (single id if no merge)
  - Tasks previously in any batch's "unclustered" list go into final "unclustered" unless
    you confidently see them belong to a final cluster.
  - Every task must appear exactly once across final clusters + unclustered.

Output STRICT JSON only:
{
  "clusters": [
    {
      "id": "C1",
      "skill_concept": "...",
      "member_ids": ["..."],
      "reasoning": "...",
      "merged_from": ["B1_C3", "B2_C7"]
    }
  ],
  "unclustered": ["..."]
}

Batch results follow:
"""


def format_cards(cards):
    lines = []
    for c in cards:
        tools = ", ".join(c["tools"]) if c["tools"] else "-"
        lines.append(
            f"- ID: {c['task_id']}\n"
            f"  Goal: {c['goal']}\n"
            f"  Tools: {tools}\n"
            f"  Category: {c['category']}"
        )
    return "\n".join(lines)


def call_claude(prompt: str, model: str = MODEL, timeout: int = TIMEOUT) -> tuple[str, dict]:
    proc = subprocess.run(
        ["claude", "-p",
         "--model", model,
         "--output-format", "json",
         "--tools", "",
         "--no-session-persistence"],
        input=prompt,
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed (rc={proc.returncode}):\n{proc.stderr[:2000]}")
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


def chunk(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def cluster_one_batch(cards, batch_idx, n_batches):
    BATCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = BATCH_CACHE_DIR / f"batch_{batch_idx:02d}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        n_clusters = len(cached["parsed"].get("clusters", []))
        n_uncl = len(cached["parsed"].get("unclustered", []))
        print(f"  batch {batch_idx}/{n_batches}: {len(cards)} tasks  [CACHED]  "
              f"→ {n_clusters} clusters, {n_uncl} unclustered", flush=True)
        return cached["parsed"], cached["meta"]

    prompt = PROMPT_HEADER.format(n_tasks=len(cards), batch_idx=batch_idx, n_batches=n_batches) \
        + format_cards(cards)
    print(f"  batch {batch_idx}/{n_batches}: {len(cards)} tasks, prompt {len(prompt)} chars",
          flush=True)
    raw, meta = call_claude(prompt)
    parsed = parse_response(raw)
    # save IMMEDIATELY so we don't lose work
    cache_file.write_text(json.dumps({
        "batch_idx": batch_idx,
        "n_tasks": len(cards),
        "parsed": parsed,
        "meta": {k: meta.get(k) for k in ("duration_ms", "total_cost_usd", "session_id")},
    }, ensure_ascii=False, indent=2))
    n_clusters = len(parsed.get("clusters", []))
    n_uncl = len(parsed.get("unclustered", []))
    cost = meta.get("total_cost_usd", 0)
    dur = meta.get("duration_ms", 0) / 1000
    print(f"     → {n_clusters} clusters, {n_uncl} unclustered, "
          f"{dur:.0f}s, ${cost:.3f}  (cached to {cache_file.name})", flush=True)
    return parsed, meta


def concat_batches(batch_results: list[dict]) -> dict:
    """Concatenate batch-level clusters with no merge. IDs prefixed with batch index."""
    final_clusters = []
    final_unclustered = []
    for i, br in enumerate(batch_results, 1):
        for c in br.get("clusters", []):
            cid = c.get("id", f"B{i}_anon")
            if not cid.startswith(f"B{i}"):
                cid = f"B{i}_{cid}"
            final_clusters.append({
                "id": cid,
                "skill_concept": c.get("skill_concept", ""),
                "member_ids": c.get("member_ids", []),
                "reasoning": c.get("reasoning", ""),
                "merged_from": [cid],
            })
        final_unclustered.extend(br.get("unclustered", []))
    return {
        "clusters": final_clusters,
        "unclustered": sorted(set(final_unclustered)),
    }


def merge_batches(batch_results):
    """Run a final LLM call to merge batch-level clusters into a global set."""
    body_lines = []
    for i, br in enumerate(batch_results, 1):
        body_lines.append(f"\n## Batch {i}")
        body_lines.append(json.dumps(br, ensure_ascii=False, indent=2))
    prompt = MERGE_PROMPT_HEADER + "\n".join(body_lines)
    print(f"\nmerging {len(batch_results)} batches, prompt {len(prompt)} chars",
          flush=True)
    raw, meta = call_claude(prompt)
    parsed = parse_response(raw)
    print(f"  → {len(parsed.get('clusters', []))} merged clusters, "
          f"{len(parsed.get('unclustered', []))} unclustered, "
          f"${meta.get('total_cost_usd', 0):.3f}",
          flush=True)
    return parsed, meta


def validate(parsed: dict, all_ids: set[str]) -> tuple[bool, str]:
    seen = []
    for c in parsed.get("clusters", []):
        seen.extend(c.get("member_ids", []))
    seen.extend(parsed.get("unclustered", []))
    seen_set = set(seen)
    missing = all_ids - seen_set
    extra = seen_set - all_ids
    issues = []
    if missing:
        issues.append(f"missing {len(missing)}: {sorted(missing)[:5]}")
    if extra:
        issues.append(f"unknown {len(extra)}: {sorted(extra)[:5]}")
    if len(seen) != len(seen_set):
        from collections import Counter
        dups = [t for t, n in Counter(seen).items() if n > 1]
        issues.append(f"duplicates {len(dups)}: {dups[:5]}")
    return (not issues), "; ".join(issues) if issues else "ok"


def main():
    cards = [json.loads(l) for l in open(CARDS_FILE)]
    all_ids = {c["task_id"] for c in cards}
    print(f"loaded {len(cards)} cards, batching by {BATCH_SIZE}")

    batches = list(chunk(cards, BATCH_SIZE))
    n_batches = len(batches)

    batch_results = []
    total_cost = 0.0
    for i, batch in enumerate(batches, 1):
        parsed, meta = cluster_one_batch(batch, i, n_batches)
        total_cost += meta.get("total_cost_usd", 0) or 0
        batch_results.append(parsed)

    # Either LLM-merge (if enabled and prompt size manageable) or just concatenate.
    if DO_MERGE:
        final_parsed, merge_meta = merge_batches(batch_results)
        total_cost += merge_meta.get("total_cost_usd", 0) or 0
    else:
        final_parsed = concat_batches(batch_results)

    ok, msg = validate(final_parsed, all_ids)
    if not ok:
        print(f"WARN: validation issues after merge: {msg}")

    out = {
        "model": MODEL,
        "n_input_tasks": len(cards),
        "n_batches": n_batches,
        "n_clusters": len(final_parsed.get("clusters", [])),
        "n_unclustered": len(final_parsed.get("unclustered", [])),
        "total_cost_usd": round(total_cost, 4),
        "validation": {"ok": ok, "message": msg},
        "_batch_results": batch_results,
        **final_parsed,
    }
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nsaved → {OUT_FILE}")
    print(f"  {out['n_clusters']} clusters, {out['n_unclustered']} unclustered")
    sizes = sorted([len(c["member_ids"]) for c in final_parsed.get("clusters", [])], reverse=True)
    print(f"  cluster sizes: {sizes}")
    print(f"  total cost: ${total_cost:.3f}")


if __name__ == "__main__":
    main()
