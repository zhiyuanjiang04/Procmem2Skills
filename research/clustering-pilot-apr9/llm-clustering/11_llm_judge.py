"""Independent LLM-as-judge over the final clusters.

This is a SEPARATE validation pass: a fresh Claude call sees only the cluster
+ candidate skill_concept + the member tasks (instruction + solution.sh), and
votes acceptable / borderline / not-acceptable with a one-sentence reason.

Used for: cross-checking the refinement decisions, and providing the headline
quality number reported alongside llm_clusters_clean.json.

Supports parallelism via PARALLEL env var.
"""
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent
CLUSTERS_FILE = ROOT / "outputs" / "llm_clusters_clean.json"
TASKS_JSONL = ROOT.parent / "data" / "tasks.jsonl"
SOLUTIONS_DIR = ROOT.parent.parent / "terminal-bench" / "original-tasks"
OUT_FILE = ROOT / "outputs" / "llm_judge.json"
JUDGE_CACHE_DIR = ROOT / "outputs" / "_judge_cache"

MODEL = "claude-sonnet-4-6"
MAX_INSTR_CHARS = 3000
MAX_SOLUTION_CHARS = 2500


PROMPT_TMPL = """\
You are an independent reviewer. Another LLM produced the cluster below by
grouping terminal-bench tasks under a single candidate skill. Your job is to
judge — without trusting the original LLM's reasoning — whether one reusable
skill could realistically cover all these tasks at execution time.

Candidate skill concept:
  {skill_concept}

Tasks in this cluster:
{task_blocks}

Apply this standard:
  - ACCEPT:    A single procedure (steps + preconditions + failure modes)
               drawn from the candidate skill would meaningfully help an agent
               solve every task here. Tasks share the SAME procedure shape and
               overlapping tools.
  - BORDERLINE: The skill helps the majority but at least one task would need
               substantial deviation, or the skill must be written so generically
               it loses its bite.
  - REJECT:    The tasks are only superficially related; one skill cannot serve
               them without becoming useless.

Output STRICT JSON only:
{{
  "verdict": "ACCEPT" | "BORDERLINE" | "REJECT",
  "verdict_reason": "one or two sentences explaining the verdict",
  "weakest_link": "task_id of the task that fits the candidate skill the worst (or null if all fit equally well)",
  "weakest_link_reason": "one sentence — why this task is the weakest"
}}
"""


def call_claude(prompt: str, timeout: int = 600) -> tuple[str, dict]:
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


def load_solution(task_id: str) -> str:
    sol = SOLUTIONS_DIR / task_id / "solution.sh"
    if sol.exists():
        try:
            return sol.read_text(errors="replace")
        except Exception:
            return ""
    return ""


def format_task_block(task_id: str, instr: str, sol: str) -> str:
    instr = (instr or "").strip()[:MAX_INSTR_CHARS]
    sol = (sol or "").strip()[:MAX_SOLUTION_CHARS]
    return (
        f"--- TASK: {task_id} ---\n"
        f"Instruction:\n{instr}\n\n"
        f"Solution.sh (truncated):\n{sol}\n"
    )


def judge_one(cluster: dict, task_data: dict) -> dict:
    blocks = []
    for tid in cluster["member_ids"]:
        meta = task_data.get(tid, {})
        blocks.append(format_task_block(tid, meta.get("instruction", ""), load_solution(tid)))
    prompt = PROMPT_TMPL.format(
        skill_concept=cluster["skill_concept"],
        task_blocks="\n".join(blocks),
    )
    raw, meta = call_claude(prompt)
    parsed = parse_response(raw)
    return {
        "cluster_id": cluster["id"],
        "skill_concept": cluster["skill_concept"],
        "n_tasks": len(cluster["member_ids"]),
        "verdict": parsed.get("verdict"),
        "verdict_reason": parsed.get("verdict_reason"),
        "weakest_link": parsed.get("weakest_link"),
        "weakest_link_reason": parsed.get("weakest_link_reason"),
        "_meta": {"duration_ms": meta.get("duration_ms"), "cost_usd": meta.get("total_cost_usd")},
    }


def main(limit: int | None = None):
    final = json.loads(CLUSTERS_FILE.read_text())
    task_data = {json.loads(l)["task_id"]: json.loads(l) for l in open(TASKS_JSONL)}

    clusters = final["clusters"]
    if limit:
        clusters = clusters[:limit]
    print(f"judging {len(clusters)} clusters (model={MODEL})")

    JUDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parallel = int(os.environ.get("PARALLEL", "1"))
    print(f"judging with PARALLEL={parallel}")

    def task_fn(idx_cluster):
        idx, c = idx_cluster
        safe_cid = c["id"].replace("/", "_")
        cache_file = JUDGE_CACHE_DIR / f"{safe_cid}.json"
        if cache_file.exists():
            return idx, c, json.loads(cache_file.read_text()), "cached"
        try:
            j = judge_one(c, task_data)
            cache_file.write_text(json.dumps(j, ensure_ascii=False, indent=2))
            return idx, c, j, "ok"
        except Exception as e:
            return idx, c, {"cluster_id": c["id"], "error": str(e)}, "error"

    judgments = [None] * len(clusters)
    total_cost = 0.0
    n_done = 0
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(task_fn, (i, c)) for i, c in enumerate(clusters)]
        for f in as_completed(futures):
            idx, c, j, status = f.result()
            judgments[idx] = j
            n_done += 1
            size = len(c["member_ids"])
            if status == "error":
                print(f"[{n_done}/{len(clusters)}] {c['id']} ({size}) ERROR: {j.get('error', '')[:120]}", flush=True)
            else:
                cost = (j.get("_meta") or {}).get("cost_usd") or 0
                total_cost += cost
                tag = "[CACHED]" if status == "cached" else ""
                print(f"[{n_done}/{len(clusters)}] {c['id']} ({size}) {tag} verdict={j.get('verdict')} cost=${cost:.3f}", flush=True)

    # Aggregate
    verdict_counts = {}
    for j in judgments:
        v = j.get("verdict") or "ERROR"
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    out = {
        "model": MODEL,
        "n_clusters_judged": len(judgments),
        "verdict_counts": verdict_counts,
        "total_cost_usd": round(total_cost, 4),
        "judgments": judgments,
    }
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print(f"\nsaved → {OUT_FILE}")
    print(f"  verdict counts: {verdict_counts}")
    print(f"  total cost: ${total_cost:.3f}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=n)
