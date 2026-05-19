# Sonnet 4.6 pass-rate (selection-proxy) eval

## What this measures

Per-trial: model sees `<available_skills>` with `|GT| + n_noise` candidates
(all GT + random distractors from ClawHub), picks ONE skill via
`<tool_call name="skill"><name>…</name></tool_call>`. The trial "passes" if
the picked skill is in the task's GT set.

## Why this is a pass-rate proxy (not the real thing)

True execution pass-rate requires running each trial in a container with the
terminus-2-skills agent, loading the picked SKILL.md, executing shell
commands, and running tests. **Docker is not installed on Anvil**, so the
end-to-end harness is blocked. Pipeline scaffolding lives in
`testsets/exec_eval/` and is ready to fire on a Docker host.

This selection-proxy is the strongest no-Docker signal: with the right skill
loaded, the agent has the procedural guidance designed for the task; with a
wrong/stub skill loaded, the agent has misleading or empty guidance. So this
hit-rate **upper-bounds** real execution pass-rate. (Execution can fail
downstream even with the right skill, but cannot succeed without it on
GT-aligned tasks.)

## Spec (user 2026-05-13)

- **Step 1**: All GT skills always placed in pool (multi-GT preserved).
- **Step 2**: Random distractors fill remaining slots.
- **Pool size** = `|GT| + n_noise`. Caps at ~55 (SB max-GT=7, TB max-GT=5).
- **n_noise sweep**: `{0, 1, 5, 10, 20, 50}` — GT-only, GT+1, ..., GT+50.
- **Datasets**: SkillsBench 88 + TerminalBench-validated 62.
- **Seed**: 0. Single-seed run; variance bars deferred.
- **Backend**: `claude -p` Max plan, Sonnet 4.6.

## Files

- `sb_passrate.jsonl` — 528 SB trials (88 × 6 n_noise).
- `tb_passrate.jsonl` — 372 TB trials (62 × 6 n_noise).
- `passrate_table.json` — aggregated cells.
- `passrate_table.md` — markdown table (the one to drop into reports).
- `run.log` — stderr from the orchestrator.

## Reproduce

```bash
cd /anvil/projects/x-cis260386/william/procmem2skills/procmem2skills
bash testsets/run_pass_rate_sonnet46.sh           # full
bash testsets/run_pass_rate_sonnet46.sh --smoke   # 5 tasks/dataset
bash testsets/run_pass_rate_sonnet46.sh --resume  # skip done trials
```

Concurrency=6, conservative for Max-plan rate limits. ~25 min wall-clock
for the full 900-trial grid.
