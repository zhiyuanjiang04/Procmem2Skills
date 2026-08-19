"""Stage 3: aggregate free-text primary_mode_candidate labels into a v1
canonical taxonomy.

Batched approach (single big LLM call times out on OAuth at ~38k tokens):
  Round A: split labels into 4 batches of ~60, each batch → propose ~10 modes + assignments.
  Round B: merge the 4 batch proposals into a unified v1 taxonomy + global assignments.

Both rounds use Claude Sonnet 4.6 via `claude -p` headless OAuth.
Each batch result is cached on disk so reruns skip completed batches.
"""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS_FILE = ROOT / "outputs" / "labels_raw.jsonl"
OUT_MAP = ROOT / "outputs" / "canonical_mode_map.v1.json"
OUT_DEF = ROOT / "outputs" / "mode_definition.v1.md"
BATCH_CACHE = ROOT / "outputs" / "_aggregate_batches"

MODEL = "claude-sonnet-4-6"
TIMEOUT = int(os.environ.get("TIMEOUT", "600"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "60"))


BATCH_PROMPT = """\
You are mining canonical failure/success modes from agent-trajectory labels.

Each input record (one trajectory) has:
  id, benchmark, setting, arm, status,
  primary_mode_candidate (free text), secondary_factors,
  skill_effect_judgment, capability_vs_knowledge.

Task: produce a small set of canonical modes (snake_case names) covering THIS BATCH,
and assign every input id to exactly one mode.

Modes should be specific procedural patterns, not generic catch-alls. Cover:
  - environment/dependency failures
  - API or library misuse
  - debugging-loop / no-progress
  - verification/output-format mismatches
  - timeout / long-horizon failures
  - skill-specific patterns (skill misguidance, skill ignored, skill helped)
  - workflow-specific patterns
  - successful-execution patterns (clean / recovered / skill-guided)

Output STRICT JSON only (no markdown fences):
{
  "modes": [
    {"name": "missing_python_dependency",
     "definition": "1-3 sentences",
     "n_assigned": 17}
  ],
  "assignments": [
    {"id": "<trial_id>", "mode": "missing_python_dependency", "reason": "one short sentence"}
  ]
}

Rules:
- Every input id MUST appear once in assignments.
- Mode names in assignments MUST be defined in modes.
- Aim for 8–14 modes per batch. Prefer fewer, clearer modes over many narrow ones.

Input records:
"""


MERGE_PROMPT = """\
You are consolidating mode taxonomies from multiple batches into a single
canonical v1 taxonomy.

Each batch proposed a list of modes (name + definition + n_assigned). Many
batch modes will overlap. Your job: produce a unified set of 9-14 canonical
modes, and a mapping from each batch-level mode name to its unified target.

Output STRICT JSON only (no markdown fences):
{
  "modes": [
    {"name": "unified_snake_case",
     "definition": "1-3 sentences",
     "merged_from_batch_modes": ["batch1_mode_name", "batch2_mode_name"]}
  ],
  "batch_mode_map": {
    "batch1_mode_name": "unified_snake_case",
    "batch2_mode_name": "unified_snake_case"
  }
}

Rules:
- Every batch mode name that appears in input MUST appear as a key in batch_mode_map.
- Every value in batch_mode_map MUST be a name in modes[].
- Aim for 9-14 unified modes total.
- "merged_from_batch_modes" lists the source-batch names that were folded into each unified mode.

Batch mode lists:
"""


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


def make_input_records(labels: list[dict]) -> list[dict]:
    out = []
    for r in labels:
        if "labels" not in r or r.get("labels") is None:
            continue
        lbl = r["labels"]
        if not isinstance(lbl, dict):
            continue
        out.append({
            "id": r.get("trial_id") or r.get("trial_name"),
            "benchmark": r["benchmark"],
            "setting": r["setting"],
            "arm": r["arm"],
            "status": r["status"],
            "primary_mode_candidate": lbl.get("primary_mode_candidate", ""),
            "secondary_factors": lbl.get("secondary_factors", []),
            "skill_effect_judgment": lbl.get("skill_effect_judgment", ""),
            "capability_vs_knowledge": lbl.get("capability_vs_knowledge", ""),
        })
    return out


def chunk(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def aggregate_batch(records: list[dict], batch_idx: int) -> dict:
    BATCH_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = BATCH_CACHE / f"batch_{batch_idx:02d}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        print(f"  batch {batch_idx} [CACHED] modes={len(cached.get('modes', []))}, "
              f"assignments={len(cached.get('assignments', []))}")
        return cached

    body = json.dumps(records, ensure_ascii=False, indent=2)
    prompt = BATCH_PROMPT + body
    print(f"  batch {batch_idx}: {len(records)} records, prompt {len(prompt)} chars")
    raw, meta = call_claude(prompt)
    parsed = parse_response(raw)
    cache_file.write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
    cost = meta.get("total_cost_usd", 0)
    print(f"     → modes={len(parsed.get('modes', []))}, "
          f"assignments={len(parsed.get('assignments', []))}, "
          f"cost=${cost:.3f}")
    return parsed


def merge_batches(batch_results: list[dict]) -> dict:
    """Two-step merge: send only mode definitions to LLM (cheap, fast),
    then locally map every assignment via batch_mode_map."""
    # Collect all batch modes
    batch_modes_only = []
    for i, br in enumerate(batch_results, 1):
        for m in br.get("modes", []):
            batch_modes_only.append({
                "batch": i,
                "name": m["name"],
                "definition": m.get("definition", ""),
                "n_assigned": m.get("n_assigned", 0),
            })

    body = json.dumps(batch_modes_only, ensure_ascii=False, indent=2)
    prompt = MERGE_PROMPT + body
    print(f"\nmerge: {len(batch_modes_only)} batch modes, prompt {len(prompt)} chars")
    raw, meta = call_claude(prompt, timeout=900)
    parsed = parse_response(raw)
    cost = meta.get("total_cost_usd", 0)
    n_unified = len(parsed.get("modes", []))
    print(f"  → {n_unified} unified modes, cost=${cost:.3f}")

    # Locally re-map all assignments
    batch_mode_map = parsed.get("batch_mode_map", {})
    unified_modes = {m["name"]: m for m in parsed.get("modes", [])}
    counts = {n: 0 for n in unified_modes}

    final_assignments = []
    unmapped = []
    for i, br in enumerate(batch_results, 1):
        for a in br.get("assignments", []):
            src_mode = a["mode"]
            tgt = batch_mode_map.get(src_mode)
            if tgt is None or tgt not in unified_modes:
                unmapped.append((src_mode, a["id"]))
                continue
            final_assignments.append({
                "id": a["id"],
                "mode": tgt,
                "source_batch": i,
                "source_batch_mode": src_mode,
                "reason": a.get("reason", ""),
            })
            counts[tgt] += 1

    if unmapped:
        print(f"  WARN: {len(unmapped)} assignments unmapped (batch modes missing from map):")
        from collections import Counter
        for src_mode, n in Counter(m for m, _ in unmapped).most_common(5):
            print(f"    {src_mode}: {n}")

    # Patch n_assigned into modes
    final_modes = []
    for m in parsed.get("modes", []):
        m2 = dict(m)
        m2["n_assigned"] = counts.get(m["name"], 0)
        final_modes.append(m2)

    return {
        "modes": final_modes,
        "assignments": final_assignments,
        "_batch_mode_map": batch_mode_map,
        "_unmapped_count": len(unmapped),
    }


def validate(parsed: dict, all_ids: set[str]) -> tuple[bool, str]:
    if "modes" not in parsed or "assignments" not in parsed:
        return False, "missing 'modes' or 'assignments'"
    mode_names = {m["name"] for m in parsed["modes"]}
    assigned_ids = []
    for a in parsed["assignments"]:
        assigned_ids.append(a["id"])
        if a["mode"] not in mode_names:
            return False, f"assignment uses undefined mode: {a['mode']}"
    assigned_set = set(assigned_ids)
    missing = all_ids - assigned_set
    extra = assigned_set - all_ids
    issues = []
    if missing:
        issues.append(f"missing {len(missing)}: {sorted(missing)[:3]}")
    if extra:
        issues.append(f"unknown {len(extra)}: {sorted(extra)[:3]}")
    if len(assigned_ids) != len(assigned_set):
        from collections import Counter
        dups = [t for t, n in Counter(assigned_ids).items() if n > 1]
        issues.append(f"duplicates {len(dups)}: {dups[:3]}")
    return (not issues), "; ".join(issues) if issues else "ok"


def main():
    labels = [json.loads(l) for l in open(LABELS_FILE)]
    input_records = make_input_records(labels)
    all_ids = {r["id"] for r in input_records}
    print(f"loaded {len(labels)} labels, {len(input_records)} usable")
    print(f"batching by {BATCH_SIZE}")

    batches = list(chunk(input_records, BATCH_SIZE))
    print(f"{len(batches)} batches\n")

    batch_results = []
    for i, b in enumerate(batches, 1):
        br = aggregate_batch(b, i)
        batch_results.append(br)

    final = merge_batches(batch_results)

    ok, msg = validate(final, all_ids)
    if not ok:
        print(f"\nWARN: validation failed: {msg}")

    OUT_MAP.write_text(json.dumps({
        "model": MODEL,
        "n_labels": len(input_records),
        "n_batches": len(batches),
        "validation": {"ok": ok, "message": msg},
        "modes": final.get("modes", []),
        "assignments": final.get("assignments", []),
        "_batch_results": batch_results,
    }, ensure_ascii=False, indent=2))

    md = ["# Canonical Failure/Success Mode Taxonomy (v1)\n",
          f"Generated from {len(input_records)} LLM-labeled trajectories via {MODEL}, "
          f"split into {len(batches)} batches and merged.\n",
          f"Total unified modes: {len(final.get('modes', []))}\n",
          "## Modes\n"]
    for m in final.get("modes", []):
        md.append(f"### `{m['name']}` (n={m.get('n_assigned', '?')})\n")
        md.append(m.get("definition", ""))
        merged_from = m.get("merged_from_batch_modes") or m.get("merged_from", [])
        if merged_from:
            md.append(f"\n*Merged from batch modes: {', '.join(merged_from)}*\n")
    OUT_DEF.write_text("\n".join(md))

    print(f"\nsaved:")
    print(f"  {OUT_MAP}")
    print(f"  {OUT_DEF}")
    print(f"\nFinal modes:")
    for m in final.get("modes", []):
        print(f"  {m['name']:<45} n={m.get('n_assigned', '?')}")


if __name__ == "__main__":
    main()
