# Execution-Pass-Rate Eval (Docker-blocked on Anvil)

End-to-end TB/SB task-execution pass-rate harness for the (n_noise × dataset)
grid. The model is given a pool of `|GT| + n_noise` skills, picks one, the
agent loads its `SKILL.md` content, executes shell commands in a container,
and the task's `tests/` decides pass/fail.

## Status

**Pipeline code: scaffolded.** **Real execution: blocked** — Anvil has no
Docker (only Apptainer 1.4); `harbor`/`tb` runners both require dockerd.

Until a Docker host is provisioned, use the **selection-proxy** harness at
`testsets/skill_selection_eval/run_pass_rate.py`, which measures whether the
model picked the GT skill from the same pool (numbers in
`testsets/eval_passrate_sonnet46/passrate_table.md`).

## Spec

- Step 1: ALL GT skills always placed in pool.
- Step 2: random distractors from ClawHub corpus fill `n_noise` slots.
- `n_noise ∈ {0, 1, 5, 10, 20, 50}` → pool sizes ≈ `{|GT|, |GT|+1, ..., |GT|+50}`.
- Datasets: SB (88) + TB-validated (62) = 150 tasks.
- Total trials: 150 × 6 = 900.

## Pipeline (when Docker is available)

```
testsets/exec_eval/
  pool_to_skill_dir.py    # take pool spec → materialize /tmp/eval/<tid>/skills/<slug>/SKILL.md
  build_noise_stubs.py    # generate description-only SKILL.md stubs for noise skills
  run_trial.sh            # harbor run -p <task> -a terminus-2-skills -m sonnet-4-6 --skill-dir <pool>
  orchestrate.py          # 900-trial driver, asyncio.Semaphore(K), --resume
  aggregate.py            # JSONL → pass_rate table by (dataset, n_noise)
```

### Per-trial flow (designed)

```
1. pool_to_skill_dir(task, n_noise, seed)
   - copies <task>/environment/skills/<gt_slug>/  → /tmp/eval/<trial>/skills/
   - generates noise-stub SKILL.md from corpus description for each distractor
   - returns trial_dir
2. harbor run -p <task_dir> -a terminus-2-skills -m claude-sonnet-4-6 \
     --skill-dir /tmp/eval/<trial>/skills/ --skill-format xml --temperature 0.0
3. Parse harbor output → {pass: bool, n_turns, picked_skill, wall_clock}
4. Append JSONL row to runs/exec_eval_<dataset>.jsonl
```

### Isolation guarantees

- Each trial has its own `trial_dir` under `/tmp/eval/<trial_id>/`.
- Each trial spawns a fresh container (harbor builds Dockerfile per run).
- No shared state between trials.
- `--resume`: skip any `(task_id, n_noise, seed)` already in the JSONL output.

## What's needed to launch

1. A host with Docker + docker-compose installed.
2. `pip install harbor` (from skillsbench_repo's pyproject — pulls
   `harbor @ git+https://github.com/laude-institute/harbor.git`).
3. Sonnet 4.6 access via `LITELLM_API_KEY` env var (harbor uses LiteLLM).
4. Run: `bash testsets/exec_eval/run_exec_eval.sh` (script at this path
   creates `/tmp/eval/<trial_dir>/` and calls `harbor run` per trial).

Estimated cost on a real run:
- ~900 trials × Sonnet 4.6 multi-turn agent (~5-15 turns avg, smaller pools
  than the original 500 grid). Pool max=51 keeps XML cheap.
- API cost: ~$0.10–$0.50/trial → **$100-$500 total**.
- Wall-clock: 6–12 hours at conc=4-6 on a single Docker host.

## Why the LLM-as-judge fallback isn't used here

This was considered: feed Opus 4.7 the (task, picked-skill-SKILL.md) and
ask "would the agent likely complete this with this guidance?". Rejected
because:
- The user explicitly asked for execution pass-rate, not yet-another LLM
  judgement. The selection-proxy at `skill_selection_eval/run_pass_rate.py`
  is the most defensible cheap proxy; a third layer of judge-on-judge is
  weaker signal.
- Memory `feedback_retrieval_isolation.md` warns against
  performance-as-GT-via-LLM in primary metrics.

When real execution runs, we'll have ground-truth pass/fail.
