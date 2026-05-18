"""Walk extracted/ (selective extract from outputs.tar.gz) and produce manifest.jsonl
with one record per trial.

Each record captures: benchmark / setting / arm / task / status / reward + pointers
to the local result.json and codex.txt files.

Path layout (after extraction):
  extracted/codex-gpt-5-3-codex/eval/<bench>/<setting>/runs/<run-name>/
    skill/<bench-name>/results/harbor-jobs/<job>/<task>__<id>/
      result.json
      agent/codex.txt
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXTRACTED = ROOT.parent / "pm2s-traces" / "extracted"
OUT = ROOT / "outputs" / "manifest.jsonl"

# arm parsed from run-name: "native-{skill,workflow}-inject-..."
ARM_RE = re.compile(r"native-(skill|workflow)-inject-")
SETTING_RE = re.compile(r"^(raw|[0-5]s[0-5]f)$")


def parse_path(p: Path) -> dict | None:
    """Extract benchmark/setting/arm/run_name from result.json path."""
    parts = p.parts
    try:
        eval_idx = parts.index("eval")
    except ValueError:
        return None
    if eval_idx + 4 >= len(parts):
        return None
    bench = parts[eval_idx + 1]                 # skillsbench / terminalbench2 / terminalbenchpro
    setting = parts[eval_idx + 2]               # raw / 5s0f / ... / 0s5f
    if not SETTING_RE.match(setting):
        return None

    if setting == "raw":
        # raw layout: eval/<bench>/raw/harbor-jobs/<job>/<task>__<id>/result.json
        # no runs/ dir; treat the harbor-job folder as run_name
        arm = "raw"
        if eval_idx + 4 < len(parts) and parts[eval_idx + 3] == "harbor-jobs":
            run_name = parts[eval_idx + 4]
        else:
            run_name = "raw"
    else:
        if parts[eval_idx + 3] != "runs":
            return None
        run_name = parts[eval_idx + 4]
        m = ARM_RE.search(run_name)
        if not m:
            return None
        arm = m.group(1)                          # "skill" or "workflow"

    return {"benchmark": bench, "setting": setting, "arm": arm, "run_name": run_name}


def derive_codex_path(result_path: Path) -> Path | None:
    """Sibling agent/codex.txt for a given result.json."""
    candidate = result_path.parent / "agent" / "codex.txt"
    return candidate if candidate.exists() else None


def derive_instruction_path(result_path: Path, task_name: str | None) -> Path | None:
    """Locate instruction.md for this trial.

    Layout: <run-root>/<arm>/<bench-name>/prepared-tasks/<task>/instruction.md
    where result.json lives at:
    <run-root>/<arm>/<bench-name>/results/harbor-jobs/<job>/<task>__<id>/result.json
    """
    if not task_name:
        return None
    parts = result_path.parts
    try:
        results_idx = parts.index("results")
    except ValueError:
        # raw layout: eval/<bench>/raw/harbor-jobs/<job>/<task>__<id>/result.json
        # Try sibling prepared-tasks under raw root
        try:
            raw_idx = parts.index("raw")
            run_root = Path(*parts[: raw_idx + 1])
            cand = run_root / "prepared-tasks" / task_name / "instruction.md"
            return cand if cand.exists() else None
        except ValueError:
            return None
    run_root_with_bench = Path(*parts[:results_idx])
    cand = run_root_with_bench / "prepared-tasks" / task_name / "instruction.md"
    return cand if cand.exists() else None


def build_skill_index(extracted_root: Path) -> dict[tuple[str, str, str], Path]:
    """Index SKILL.md by (benchmark, setting, task).

    Skills repository layout:
      codex-gpt-5-3-codex/skills/<bench>/<setting>/<task>/SKILL.md
    """
    index = {}
    skills_root = extracted_root / "codex-gpt-5-3-codex" / "skills"
    if not skills_root.exists():
        return index
    for sk in skills_root.rglob("SKILL.md"):
        rel_parts = sk.relative_to(skills_root).parts
        if len(rel_parts) >= 4:
            bench, setting, task = rel_parts[0], rel_parts[1], rel_parts[2]
            index[(bench, setting, task)] = sk
    return index


def load_result(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(errors="replace"))
    except Exception:
        return None


def build_record(result_path: Path, skill_index: dict | None = None) -> dict | None:
    meta = parse_path(result_path)
    if not meta:
        return None
    data = load_result(result_path)
    if not data:
        return None
    rewards = (data.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    status = "success" if (isinstance(reward, (int, float)) and reward > 0) else "failure"
    exc = data.get("exception_info")
    exc_type = exc.get("type") if isinstance(exc, dict) else None

    # Duration
    started = data.get("started_at")
    finished = data.get("finished_at")
    duration_sec = None
    if started and finished:
        try:
            from datetime import datetime
            t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(finished.replace("Z", "+00:00"))
            duration_sec = (t1 - t0).total_seconds()
        except Exception:
            pass

    config = data.get("config") or {}
    agent_cfg = config.get("agent") or {}
    codex_path = derive_codex_path(result_path)
    task_name = data.get("task_name")
    instruction_path = derive_instruction_path(result_path, task_name)

    # Agent execution metrics (for cost/time comparison across arms)
    agent_result = data.get("agent_result") or {}
    agent_input_tokens = agent_result.get("n_input_tokens")
    agent_cache_tokens = agent_result.get("n_cache_tokens")
    agent_output_tokens = agent_result.get("n_output_tokens")
    agent_cost_usd = agent_result.get("cost_usd")
    # Per-phase durations
    ph = {k: data.get(k) or {} for k in
          ("environment_setup", "agent_setup", "agent_execution", "verifier")}
    def phase_dur(p):
        from datetime import datetime
        s, f = p.get("started_at"), p.get("finished_at")
        if not s or not f:
            return None
        try:
            return (datetime.fromisoformat(f.replace("Z", "+00:00"))
                    - datetime.fromisoformat(s.replace("Z", "+00:00"))).total_seconds()
        except Exception:
            return None
    agent_execution_sec = phase_dur(ph["agent_execution"])
    agent_setup_sec = phase_dur(ph["agent_setup"])
    env_setup_sec = phase_dur(ph["environment_setup"])
    verifier_sec = phase_dur(ph["verifier"])

    skill_path = None
    if skill_index is not None and meta["arm"] == "skill":
        skill_path = skill_index.get((meta["benchmark"], meta["setting"], task_name))

    base = ROOT.parent.parent  # /Users/hudx/Desktop/tb-work
    return {
        "trial_id": data.get("id"),
        "task_name": task_name,
        "trial_name": data.get("trial_name"),
        "benchmark": meta["benchmark"],
        "setting": meta["setting"],
        "arm": meta["arm"],
        "run_name": meta["run_name"],
        "status": status,
        "reward": reward,
        "exception_type": exc_type,
        "agent_name": (data.get("agent_info") or {}).get("name"),
        "agent_version": (data.get("agent_info") or {}).get("version"),
        "model": agent_cfg.get("model_name"),
        "started_at": started,
        "duration_sec": duration_sec,
        "agent_input_tokens": agent_input_tokens,
        "agent_cache_tokens": agent_cache_tokens,
        "agent_output_tokens": agent_output_tokens,
        "agent_cost_usd": agent_cost_usd,
        "agent_execution_sec": agent_execution_sec,
        "agent_setup_sec": agent_setup_sec,
        "env_setup_sec": env_setup_sec,
        "verifier_sec": verifier_sec,
        "result_path": str(result_path.relative_to(base)),
        "codex_path": str(codex_path.relative_to(base)) if codex_path else None,
        "instruction_path": str(instruction_path.relative_to(base)) if instruction_path else None,
        "skill_path": str(skill_path.relative_to(base)) if skill_path else None,
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"building skill index from {EXTRACTED}/codex-gpt-5-3-codex/skills...")
    skill_index = build_skill_index(EXTRACTED)
    print(f"  {len(skill_index)} (bench, setting, task) → SKILL.md mappings")

    print(f"scanning {EXTRACTED}...")
    result_files = list(EXTRACTED.rglob("result.json"))
    print(f"found {len(result_files)} result.json files")

    records = []
    skipped = 0
    no_codex = 0
    no_instruction = 0
    skill_arm_no_skill = 0
    for p in result_files:
        rec = build_record(p, skill_index)
        if rec is None:
            skipped += 1
            continue
        if not rec["codex_path"]:
            no_codex += 1
        if not rec["instruction_path"]:
            no_instruction += 1
        if rec["arm"] == "skill" and not rec["skill_path"]:
            skill_arm_no_skill += 1
        records.append(rec)

    with open(OUT, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(records)} records → {OUT}")
    print(f"  skipped: {skipped}  (path didn't match expected layout or unreadable)")
    print(f"  without codex.txt: {no_codex}")
    print(f"  without instruction.md: {no_instruction}")
    print(f"  skill-arm trials missing skill: {skill_arm_no_skill}")

    # quick stats
    from collections import Counter
    by_bench = Counter(r["benchmark"] for r in records)
    by_setting = Counter(r["setting"] for r in records)
    by_arm = Counter(r["arm"] for r in records)
    by_status = Counter(r["status"] for r in records)
    print("\n--- distribution ---")
    print(f"by benchmark: {dict(by_bench)}")
    print(f"by setting:   {dict(by_setting)}")
    print(f"by arm:       {dict(by_arm)}")
    print(f"by status:    {dict(by_status)}")

    # cross-tab
    print("\n--- (benchmark, setting, arm) → trials ---")
    cross = Counter((r["benchmark"], r["setting"], r["arm"]) for r in records)
    for k in sorted(cross.keys()):
        print(f"  {k}: {cross[k]}")


if __name__ == "__main__":
    main()
