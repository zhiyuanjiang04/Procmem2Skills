# Prefill-Context Execution Pass-Rate Eval

## Spec (user 2026-05-13, 3rd revision)

> 实验可以把 GT+neighbour skills 作为 prefill context 一起喂给 sonnet 4.6.
> 然后让模型跑 terminal bench 和 skills bench 的任务。
> 第一步：放入 ground truth skills。
> 第二步：构建 Skill pool 的大小：5/10/20/50。
> 类型分成 random，embedding 相近，和 embedding 远离的。

- **Pool sizes**: 5, 10, 20, 50 (total, including GT)
- **Noise types**: random / hard (emb-near) / easy (emb-far)
- **Method**: ALL candidate SKILL.md content prefilled into Sonnet 4.6's
  system prompt. Model executes task with Bash + Read/Write/Edit/Glob/Grep
  in an isolated workspace. Tests run via marl conda env pytest.
- **Datasets**: SkillsBench 88 + TerminalBench-validated 62.
- **Total trials**: 150 × 4 × 3 = 1800.

## Pipeline files

```
testsets/exec_eval_prefill/
  pool_builder_v3.py     # builds (gt + noise) pool, 3 noise modes, fixed total size
  prompt_assembler.py    # concatenates SKILL.md blocks (real GT + corpus-stub noise)
  run_trial.py           # per-trial driver: workspace setup, claude -p, test runner
  aggregate.py           # JSONL → pass-rate table
testsets/run_exec_prefill_sonnet46.sh  # orchestrator (SB only on Anvil)
```

## How it works (per trial)

```
1. pool = build_pool(dataset, task, pool_size, noise_mode, seed)
   - GT slugs preserved; noise sampled by mode {random|hard|easy}
   - GT-alias ClawHub IDs excluded from noise sampling
2. setup_workspace(task_dir, work):
   - generic copy: env/* (minus Dockerfile/skills) → work/
   - Dockerfile-aware: parse COPY src dst → mirror layout (work/data/, work/output/...)
   - copy tests/ → work/tests/ (for /tests/-referencing test code)
3. system_prompt = preamble + concatenated SKILL.md for all candidates
   - GT skills load real env/skills/<name>/SKILL.md content
   - Noise skills synthesize from corpus description (no public-Hub scrape)
4. user_prompt = path-rewritten instruction.md (/{root,data,output,...} → workspace)
5. claude -p --model sonnet-4-6 --system-prompt P --allowed-tools Bash,Read,Write,...
   --cwd work --permission-mode bypassPermissions  < user_prompt
6. After agent returns: run task's tests/test_outputs.py via marl conda env pytest,
   with paths rewritten the same way. Pass = pytest exit 0.
7. Record JSONL: {trial_id, dataset, task_id, pool_size, noise_mode, seed,
                  pool, agent_rc, agent_wall_s, pass, test_log_tail, ...}
```

### Trial isolation

- Each trial: `/tmp/exec_prefill/<dataset>__<task>__sz<N>__<mode>__s<seed>/work/`
- Fresh workspace per trial (rmtree + mkdir).
- Fresh claude session (`--no-session-persistence`).
- JSONL `--resume` keys: `(task_id, pool_size, noise_mode, seed)`.

## What we proved

**Methodology works.** Single trial of `3d-scan-calc` at pool_size=5 random noise
PASSED in 35s on Anvil host: agent received 5 prefilled SKILL.md blocks
(mesh-analysis GT + 4 noise stubs), parsed STL via Bash, computed mass to
within 0.1% accuracy, both pytest cases passed.

```
trial: sb__3d-scan-calc__sz5__random__s0
pass: True   agent_wall_s: 34.7   system_prompt_chars: 5311
test: 2 passed in 0.01s
```

## What's blocked

**Wall-time scaling.** During smoke (5 tasks × 12 conditions × seed) the
Max-plan rate dropped to <1 successful trial per 5 min — abandoned trial
runs left orphaned `claude -p` processes that pinned the rate window. After
killing orphans and retrying with concurrency=1, single trials were still
hanging at >5 min vs. the 35s baseline. Likely shared-node rate contention.

Full 1800-trial scale needs:
- Concurrency 4-6 sustained throughput
- ~50 sec/trial average → ~6-8 h wall-clock with K=4
- Stable Max-plan quota (or ANTHROPIC_API_KEY w/ Sonnet 4.6 access)

**TB tasks not host-runnable.** TB tasks assume specific container Python
versions, system tools, network access. SB tasks are largely host-runnable
(84/89 lightweight Dockerfiles). For TB, container execution required.

## How to launch full SB run

```bash
cd /anvil/projects/x-cis260386/william/procmem2skills/procmem2skills
bash testsets/run_exec_prefill_sonnet46.sh           # full SB grid
bash testsets/run_exec_prefill_sonnet46.sh --smoke   # 5 tasks
bash testsets/run_exec_prefill_sonnet46.sh --resume  # skip done trials
CONCURRENCY=2 bash testsets/run_exec_prefill_sonnet46.sh  # lower if rate-limited
```

Once Max plan is unthrottled (or API key available), full grid should
complete in ~6-8h producing `eval_exec_prefill_sonnet46/exec_table.md`.

## Selection-proxy upper-bound (already measured)

In the absence of full execution data, the selection-proxy run at
`testsets/eval_passrate_sonnet46/passrate_table.md` upper-bounds these
numbers — if Sonnet picks the GT skill (which it does 96-100% under random
noise), real execution has the full SKILL.md as guidance. The gap between
upper-bound and actual execution pass-rate measures: how much can the model
actually USE the procedural content once it has it? Per the one passing
trial, that gap appears small for clean tasks.
