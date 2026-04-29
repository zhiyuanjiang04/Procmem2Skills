"""Second-tier refinement: for each coarse cluster, send full instruction +
solution.sh to Claude and ask keep/split/remove_some.

This step produces the final llm_clusters.json (the main deliverable).
Optionally parallel via PARALLEL env var (default 1). Each cluster cached
to disk so resumes are cheap.
"""
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent
COARSE_FILE = ROOT / "outputs" / "llm_coarse.json"
TASKS_JSONL = ROOT.parent / "data" / "tasks.jsonl"
SOLUTIONS_DIR = ROOT.parent.parent / "terminal-bench" / "original-tasks"
OUT_FILE = ROOT / "outputs" / "llm_clusters.json"
REFINE_CACHE_DIR = ROOT / "outputs" / "_refine_cache"

MODEL = "claude-sonnet-4-6"
MAX_INSTR_CHARS = 4000          # cap per task to keep prompt size sane
MAX_SOLUTION_CHARS = 3000


PROMPT_TMPL = """\
You are refining a candidate cluster of terminal-bench tasks. The first-pass
clustering grouped these tasks because they were thought to share a single
reusable skill. Your job is to verify or correct that grouping using the full
task instructions and oracle solutions.

Candidate skill concept (from coarse pass):
  {skill_concept}

Coarse-pass reasoning:
  {reasoning}

Tasks in this candidate cluster:
{task_blocks}

For each task, judge whether the candidate skill — if implemented as a single
reusable procedure — would actually help solve it:
  - YES:     the skill's procedure directly addresses what the task needs
  - PARTIAL: skill helps with some steps but not enough to solve the task alone
  - NO:      task needs a fundamentally different procedure

Then output the cluster decision:
  - "keep":         all tasks really are covered by the candidate skill
  - "split":        the cluster contains 2+ distinct skills; output the splits
  - "remove_some":  candidate skill works for most, but some tasks should go to unclustered

Output STRICT JSON only (no markdown fences):
{{
  "decision": "keep" | "split" | "remove_some",
  "task_judgments": [
    {{"task_id": "...", "applicable": "YES" | "PARTIAL" | "NO", "reason": "one short sentence"}}
  ],
  "final_groups": [
    {{
      "skill_concept": "...",
      "member_ids": ["..."],
      "reasoning": "..."
    }}
  ],
  "removed_to_unclustered": ["..."]
}}

Rules:
  - "task_judgments" must include EVERY task above, exactly once.
  - "final_groups" describes the post-refinement groups. For "keep", produce one group with all tasks.
    For "split", produce 2+ groups, each with its own skill_concept. For "remove_some", produce one
    group with the kept tasks and list the rest in "removed_to_unclustered".
  - Every task must end up either in some final_group or in removed_to_unclustered.
  - skill_concept must be specific (a real procedure), not generic ("uses python", "involves files").
"""


def call_claude(prompt: str, model: str = MODEL, timeout: int = 600) -> tuple[str, dict]:
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
        f"Solution.sh (truncated to first {MAX_SOLUTION_CHARS} chars):\n{sol}\n"
    )


def refine_one_cluster(cluster: dict, task_data: dict) -> dict:
    task_blocks = []
    for tid in cluster["member_ids"]:
        meta = task_data.get(tid, {})
        instr = meta.get("instruction", "")
        sol = load_solution(tid)
        task_blocks.append(format_task_block(tid, instr, sol))
    blocks_str = "\n".join(task_blocks)

    prompt = PROMPT_TMPL.format(
        skill_concept=cluster["skill_concept"],
        reasoning=cluster["reasoning"],
        task_blocks=blocks_str,
    )
    raw, meta = call_claude(prompt)
    parsed = parse_response(raw)
    return {
        "source_coarse_id": cluster["id"],
        "coarse_skill_concept": cluster["skill_concept"],
        "decision": parsed.get("decision"),
        "task_judgments": parsed.get("task_judgments", []),
        "final_groups": parsed.get("final_groups", []),
        "removed_to_unclustered": parsed.get("removed_to_unclustered", []),
        "_meta": {
            "duration_ms": meta.get("duration_ms"),
            "cost_usd": meta.get("total_cost_usd"),
        },
    }


def main():
    coarse = json.loads(COARSE_FILE.read_text())
    task_data = {json.loads(l)["task_id"]: json.loads(l) for l in open(TASKS_JSONL)}
    print(f"loaded {len(coarse['clusters'])} coarse clusters, {len(task_data)} task records")

    REFINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parallel = int(os.environ.get("PARALLEL", "1"))
    print(f"refining with PARALLEL={parallel}")

    def task_fn(idx_cluster):
        idx, cluster = idx_cluster
        cid = cluster["id"]
        size = len(cluster["member_ids"])
        safe_cid = cid.replace("/", "_")
        cache_file = REFINE_CACHE_DIR / f"{safe_cid}.json"
        if cache_file.exists():
            res = json.loads(cache_file.read_text())
            return idx, cluster, res, "cached"
        try:
            res = refine_one_cluster(cluster, task_data)
            cache_file.write_text(json.dumps(res, ensure_ascii=False, indent=2))
            return idx, cluster, res, "ok"
        except Exception as e:
            return idx, cluster, {"source_coarse_id": cid, "error": str(e)}, "error"

    refined_results = [None] * len(coarse["clusters"])
    total_cost = 0.0
    n_done = 0
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(task_fn, (i, c)) for i, c in enumerate(coarse["clusters"])]
        for f in as_completed(futures):
            idx, cluster, res, status = f.result()
            refined_results[idx] = res
            n_done += 1
            cid = cluster["id"]
            size = len(cluster["member_ids"])
            if status == "error":
                print(f"[{n_done}/{len(coarse['clusters'])}] {cid} ({size}) ERROR: {res.get('error', '')[:120]}", flush=True)
            else:
                cost = (res.get("_meta") or {}).get("cost_usd") or 0
                total_cost += cost
                ng = len(res.get("final_groups", []))
                removed = len(res.get("removed_to_unclustered", []))
                tag = "[CACHED]" if status == "cached" else ""
                print(f"[{n_done}/{len(coarse['clusters'])}] {cid} ({size}) {tag} "
                      f"decision={res.get('decision')} groups={ng} removed={removed} cost=${cost:.3f}",
                      flush=True)

    # Build flat final cluster list
    final_clusters = []
    final_unclustered = list(coarse.get("unclustered", []))
    counter = 1
    for r in refined_results:
        if "error" in r:
            continue
        coarse_id = r["source_coarse_id"]
        for j, g in enumerate(r["final_groups"], 1):
            final_clusters.append({
                "id": f"{coarse_id}.{j}",
                "skill_concept": g.get("skill_concept", ""),
                "member_ids": g.get("member_ids", []),
                "reasoning": g.get("reasoning", ""),
                "task_judgments": [
                    j2 for j2 in r["task_judgments"]
                    if j2.get("task_id") in g.get("member_ids", [])
                ],
                "source_coarse_id": coarse_id,
            })
            counter += 1
        final_unclustered.extend(r.get("removed_to_unclustered", []))

    out = {
        "model": MODEL,
        "n_input_tasks": sum(len(c["member_ids"]) for c in coarse["clusters"]) + len(coarse.get("unclustered", [])),
        "n_final_clusters": len(final_clusters),
        "n_unclustered": len(set(final_unclustered)),
        "total_cost_usd": round(total_cost, 4),
        "clusters": final_clusters,
        "unclustered": sorted(set(final_unclustered)),
        "_per_coarse_results": refined_results,
    }
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print(f"\nsaved → {OUT_FILE}")
    print(f"  {out['n_final_clusters']} final clusters, {out['n_unclustered']} unclustered")
    print(f"  total cost: ${total_cost:.3f}")


if __name__ == "__main__":
    main()
