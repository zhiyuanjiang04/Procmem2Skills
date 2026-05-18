"""Stage 3: paired-compare labeling.

For each (benchmark, setting, task) where multiple arms (raw, workflow, skill)
have a trial, send the trio to Claude and let it directly compare them using
the v1 canonical taxonomy as shared vocabulary.

Outputs one record per (task, setting) triple with:
  - per_arm mode + evidence
  - deltas (workflow_vs_raw, skill_vs_raw, skill_vs_workflow)
  - skill_mechanism / workflow_mechanism

Env vars:
  PARALLEL=N             concurrent calls (default 1)
  PILOT=1                run pilot mode (24 triples, skillsbench 5s0f only)
  SETTINGS=5s0f,0s5f     comma-separated settings to include (default: all 6 + raw)
  BENCHMARKS=...         comma-separated benchmarks (default: all 3)
  PICK_TRIAL=first       which trial per (task, arm) (default: first by trial_id)
  SEED=0                 random seed for trial picking when PICK_TRIAL=random
"""
import json
import os
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_BASE = ROOT.parent.parent
MANIFEST = ROOT / "outputs" / "manifest.jsonl"
TAXONOMY = ROOT / "outputs" / "canonical_mode_map.v1.json"
OUT_PILOT = ROOT / "outputs" / "pair_labels_pilot.jsonl"
OUT_MAIN = ROOT / "outputs" / "pair_labels_v1.jsonl"
CACHE = ROOT / "outputs" / "_pair_cache"

MODEL = "claude-sonnet-4-6"
TIMEOUT = 600
INSTRUCTION_MAX = 3000
SKILL_MAX = 3000
CODEX_HEAD = 4000
CODEX_TAIL = 8000


def load_taxonomy() -> str:
    """Return formatted v1 mode list for the prompt."""
    d = json.loads(TAXONOMY.read_text())
    lines = []
    for m in d.get("modes", []):
        lines.append(f"- `{m['name']}` (n={m.get('n_assigned', '?')}): {m.get('definition', '')}")
    return "\n".join(lines)


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


def build_triples(manifest_records: list[dict], settings: set[str], benchmarks: set[str], pick: str, seed: int) -> list[dict]:
    """Group manifest records by (benchmark, setting, task) → pick one trial per arm."""
    import random
    rng = random.Random(seed)

    # Index: (bench, setting, task, arm) → list of trials
    idx = defaultdict(list)
    raw_idx = defaultdict(list)  # (bench, task) → raw trials
    for r in manifest_records:
        if r["benchmark"] not in benchmarks:
            continue
        if not r.get("codex_path"):
            continue
        if r["arm"] == "raw":
            raw_idx[(r["benchmark"], r["task_name"])].append(r)
        else:
            if r["setting"] not in settings:
                continue
            idx[(r["benchmark"], r["setting"], r["task_name"], r["arm"])].append(r)

    def pick_one(trials: list[dict]) -> dict | None:
        if not trials:
            return None
        if pick == "random":
            return rng.choice(trials)
        return sorted(trials, key=lambda x: x.get("trial_id", ""))[0]

    # Group by (bench, setting, task), require workflow AND skill arms present
    triples = []
    by_setting_task = defaultdict(dict)
    for (bench, setting, task, arm), trials in idx.items():
        by_setting_task[(bench, setting, task)][arm] = trials

    for (bench, setting, task), arms in by_setting_task.items():
        if "workflow" not in arms or "skill" not in arms:
            continue
        wf = pick_one(arms["workflow"])
        sk = pick_one(arms["skill"])
        rw = pick_one(raw_idx.get((bench, task), []))
        triples.append({
            "benchmark": bench,
            "setting": setting,
            "task_name": task,
            "raw": rw,
            "workflow": wf,
            "skill": sk,
        })

    return triples


def build_prompt(triple: dict, taxonomy_str: str) -> str:
    # Use workflow's instruction.md (contains injected workflow text)
    # else fall back to skill's, then raw's
    instr_src = triple["workflow"] or triple["skill"] or triple["raw"]
    instruction = read_file(instr_src.get("instruction_path"), INSTRUCTION_MAX) if instr_src else ""

    skill_md = read_file(triple["skill"].get("skill_path"), SKILL_MAX) if triple["skill"] else ""

    parts = []
    parts.append(
        "You compare 3 agent trajectories on the same task across 3 conditions "
        "(raw / workflow-injected / skill-injected) and produce a structured paired analysis.\n"
    )
    parts.append("---TASK INSTRUCTION (may contain injected workflow text if from workflow arm)---")
    parts.append(instruction or "(no instruction available)")
    parts.append("")
    parts.append("---V1 SHARED MODE TAXONOMY (use these names in `mode` fields)---")
    parts.append(taxonomy_str)
    parts.append("")

    for arm in ("raw", "workflow", "skill"):
        rec = triple.get(arm)
        if rec is None:
            parts.append(f"[ARM={arm}] MISSING (not run for this task)")
            parts.append("")
            continue
        codex = read_file(rec.get("codex_path"))
        codex_excerpt = head_tail(codex, CODEX_HEAD, CODEX_TAIL) if codex else "(no codex.txt)"
        parts.append(f"[ARM={arm}] status={rec['status']} reward={rec['reward']} exception={rec.get('exception_type')}")
        if arm == "skill":
            parts.append("---INJECTED SKILL.md---")
            parts.append(skill_md or "(no SKILL.md)")
        parts.append("---codex.txt (head + tail)---")
        parts.append(codex_excerpt)
        parts.append("")

    parts.append("""\
Output STRICT JSON only (no markdown fences). Schema:

{
  "per_arm": {
    "raw":      {"mode": "<v1 mode name or null if MISSING>", "status": "...", "evidence_quote": "<=160 chars verbatim or empty>"},
    "workflow": {...},
    "skill":    {...}
  },
  "deltas": {
    "workflow_vs_raw": {
      "what_changed": "1-2 sentences",
      "net_effect": "fixed" | "regressed" | "unchanged" | "mixed" | "not_comparable",
      "fixed_mode": [],
      "introduced_mode": []
    },
    "skill_vs_raw":       {...same shape...},
    "skill_vs_workflow":  {...same shape...}
  },
  "skill_mechanism": "knowledge_injection | procedural_anchor | failure_warning | none | counterproductive",
  "skill_mechanism_reason": "one sentence",
  "workflow_mechanism": "knowledge_injection | procedural_anchor | failure_warning | none | counterproductive",
  "workflow_mechanism_reason": "one sentence",
  "confidence": "high | medium | low"
}

Rules:
- Use exact v1 mode names from the taxonomy (snake_case). If a trial does not fit any, use "other".
- For MISSING arms, set per_arm.<arm> = null and deltas involving that arm to net_effect="not_comparable".
- fixed_mode / introduced_mode are mode names that the comparison arm eliminated or newly caused respectively (relative to the baseline arm).
- evidence_quote must be verbatim from codex.txt / instruction.md / SKILL.md.
- skill_mechanism = "none" if skill content was not used by agent at all.
- skill_mechanism = "counterproductive" if skill made the run worse than raw.
""")
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


def label_triple(triple: dict, taxonomy_str: str) -> dict:
    key = re.sub(r"[^A-Za-z0-9_-]", "_", f"{triple['benchmark']}_{triple['setting']}_{triple['task_name']}")
    cache_file = CACHE / f"{key}.json"
    if cache_file.exists():
        return {**json.loads(cache_file.read_text()), "_cached": True}

    prompt = build_prompt(triple, taxonomy_str)
    raw, meta = call_claude(prompt)
    parsed = parse_response(raw)
    out = {
        "benchmark": triple["benchmark"],
        "setting": triple["setting"],
        "task_name": triple["task_name"],
        "arms_present": [a for a in ("raw", "workflow", "skill") if triple.get(a) is not None],
        "trial_ids": {a: (triple[a].get("trial_id") if triple.get(a) else None) for a in ("raw", "workflow", "skill")},
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
    CACHE.mkdir(parents=True, exist_ok=True)
    pilot = os.environ.get("PILOT", "0") == "1"
    parallel = int(os.environ.get("PARALLEL", "1"))
    seed = int(os.environ.get("SEED", "0"))
    pick = os.environ.get("PICK_TRIAL", "first")

    if pilot:
        settings = {"5s0f"}
        benchmarks = {"skillsbench"}
        out_path = OUT_PILOT
    else:
        settings_env = os.environ.get("SETTINGS")
        settings = set(settings_env.split(",")) if settings_env else {"0s5f", "1s4f", "2s3f", "3s2f", "4s1f", "5s0f"}
        benchmarks_env = os.environ.get("BENCHMARKS")
        benchmarks = set(benchmarks_env.split(",")) if benchmarks_env else {"skillsbench", "terminalbench2", "terminalbenchpro"}
        out_path = OUT_MAIN

    print(f"mode: {'PILOT' if pilot else 'MAIN'}  parallel={parallel}  pick={pick}  seed={seed}")
    print(f"settings={sorted(settings)}  benchmarks={sorted(benchmarks)}")

    records = [json.loads(l) for l in open(MANIFEST)]
    triples = build_triples(records, settings, benchmarks, pick, seed)
    print(f"built {len(triples)} triples (workflow + skill arms both present)")
    n_with_raw = sum(1 for t in triples if t["raw"])
    print(f"  of which {n_with_raw} also have matching raw trial")

    taxonomy_str = load_taxonomy()

    def task_fn(idx_t):
        idx, t = idx_t
        try:
            return idx, t, label_triple(t, taxonomy_str), "ok"
        except Exception as e:
            return idx, t, {"error": str(e)}, "error"

    results: list[dict | None] = [None] * len(triples)
    n_done = 0
    n_cached = 0
    n_error = 0
    total_cost = 0.0
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(task_fn, (i, t)) for i, t in enumerate(triples)]
        for f in as_completed(futures):
            idx, t, out, status = f.result()
            results[idx] = out
            n_done += 1
            tag_arms = ",".join(out.get("arms_present", [])) if status == "ok" else ""
            if status == "error":
                n_error += 1
                print(f"[{n_done}/{len(triples)}] {t['benchmark']}/{t['setting']}/{t['task_name']} ERROR: {out.get('error', '')[:120]}",
                      flush=True)
            else:
                if out.get("_cached"):
                    n_cached += 1
                cost = (out.get("_meta") or {}).get("cost_usd") or 0
                total_cost += cost
                lbl = out.get("labels") or {}
                net = (lbl.get("deltas") or {}).get("skill_vs_raw", {}).get("net_effect", "?")
                mech = lbl.get("skill_mechanism", "?")
                tag = "[CACHED]" if out.get("_cached") else ""
                print(f"[{n_done}/{len(triples)}] {t['benchmark']}/{t['setting']}/{t['task_name'][:30]:<30} arms={tag_arms:<22} skill_vs_raw={net:<14} mech={mech:<22} {tag} cost=${cost:.3f}",
                      flush=True)

    with open(out_path, "w") as f:
        for r in results:
            if r is not None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nsaved {len(results)} paired labels → {out_path}")
    print(f"  cached: {n_cached}, errors: {n_error}, total cost: ${total_cost:.3f}")


if __name__ == "__main__":
    main()
