# SkillsBench Prefill-Context Execution Eval

End-to-end pipeline for evaluating whether injecting procedural skill documents
into an LLM's system prompt improves task execution pass-rate.

**Model**: Claude Sonnet 4.6 (`claude-sonnet-4-6`)
**Dataset**: SkillsBench (88 tasks, 14 lightweight tasks host-executable)
**Method**: Concatenate GT + noise SKILL.md blocks into system prompt, execute
task with Bash/Read/Write/Edit tools in isolated workspace, validate with pytest.

---

## Table of Contents

1. [Experiment Design](#experiment-design)
2. [Quick Start](#quick-start)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Experiment Phases](#experiment-phases)
5. [Already-Run Experiments](#already-run-experiments)
6. [Remaining Experiments](#remaining-experiments)
7. [Runner Scripts Reference](#runner-scripts-reference)
8. [Result Data Format](#result-data-format)
9. [Analysis & Report Generation](#analysis--report-generation)
10. [Troubleshooting](#troubleshooting)

---

## Experiment Design

### Research Question

Does prefilling GT skill documents into the system prompt improve task
execution pass-rate compared to baselines (no skills, noise-only, empty
framing)?

### Factorial Design

| Condition     | GT Skills | Noise Skills | Skill Prompt Frame | Description                          |
|---------------|-----------|--------------|--------------------|------------------------------------- |
| `noskill`     | No        | No           | No                 | Bare agent, no skill infrastructure  |
| `emptyframe`  | No        | No           | Yes                | Skill prompt framing, zero skills    |
| `noiseonly`   | No        | Yes (N=10)   | Yes                | Irrelevant skills only               |
| `gtonly`      | Yes       | No           | Yes                | GT skills only, no distractors       |
| `prefill`     | Yes       | Yes (varies) | Yes                | GT + noise at pool sizes 5/10/20/50  |

### Independent Variables

- **Pool size**: 5, 10, 20, 50 (total candidates including GT)
- **Noise mode**: `random` | `hard` (embedding-similar) | `easy` (embedding-distant)
- **Seed**: 0, 1, 2, 3, 4 (for variance estimation, N=5)

### Key Metrics

- **Pass-rate**: % of trials where pytest exits 0
- **Wall-time**: Agent execution time per trial (seconds)
- **Task-level pass-rate**: Per-task breakdown across conditions

---

## Quick Start

### Prerequisites

- Docker with `sg docker` group access (or run as docker-capable user)
- Claude Code CLI credentials at `~/.claude/` and `~/.claude.json`
- ~2GB disk for Docker image build

### 1. Build the Docker Image

```bash
cd skills-testset-eval
docker compose build
# Image name: skills-testset-eval-eval-pipeline
```

### 2. Run Smoke Test (5 tasks, ~30 min)

```bash
docker compose up
# Runs 5 tasks x 4 pool sizes x 3 noise modes x 1 seed = 60 trials
# Output: results/sb_exec.jsonl
```

### 3. Run Full Experiment (see Phase Scripts below)

```bash
# Phase P0: Prefill N=5 (seeds 1-4, 3 modes, 4 pool sizes)
nohup bash run_prefill_n5.sh 20 2 3 > /tmp/prefill_n5.log 2>&1 &

# Phase B: Factorial controls (emptyframe + noiseonly, 5 seeds)
nohup bash run_phase_b.sh 20 4 3 > /tmp/phase_b.log 2>&1 &
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Orchestrator (bash scripts)                  │
│  run_prefill_n5.sh / run_phase_b.sh / run_baselines_n5.sh       │
│  - Retry loop with stall detection (15 stalls → skip mode)      │
│  - Rate-limit-aware counting (filters RL'd trials from totals)  │
│  - Parallel container sharding via --skip-trials offset          │
└─────────────┬───────────────────────────────────────────────────┘
              │ sg docker -c "docker run ..."
              v
┌─────────────────────────────────────────────────────────────────┐
│                Docker Container (per shard)                       │
│  python -m exec_eval_prefill.run_trial                           │
│  --dataset sb --tasks ... --out /workspace/results/...           │
│  --pool-sizes ... --noise-modes ... --seeds ... --resume         │
└─────────────┬───────────────────────────────────────────────────┘
              │
              v
┌─────────────────────────────────────────────────────────────────┐
│               Per-Trial Pipeline (run_trial.py)                  │
│                                                                  │
│  1. build_pool()          pool_builder_v3.py                     │
│     GT preserved, noise by mode, exact pool_size                 │
│                                                                  │
│  2. setup_workspace()     Isolated /tmp/exec_prefill/<trial>/    │
│     Copy env files, mirror Dockerfile COPY, patch paths          │
│                                                                  │
│  3. build_system_prompt() prompt_assembler.py                    │
│     Preamble + <skill name="X">SKILL.md content</skill> x N     │
│                                                                  │
│  4. claude -p             Execute task with Bash tools           │
│     --model claude-sonnet-4-6 --cwd workspace                   │
│                                                                  │
│  5. pytest                Validate with task's test suite        │
│     WORKDIR=workspace, path-patched assertions                   │
│                                                                  │
│  6. Emit JSONL            Append to shared results file          │
│     {task_id, pool_size, noise_mode, seed, pass, wall_s, ...}   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Source Files

| File | Purpose |
|------|---------|
| `testsets/exec_eval_prefill/run_trial.py` | Per-trial driver: workspace setup, claude invocation, test runner |
| `testsets/exec_eval_prefill/pool_builder_v3.py` | Build candidate pools (GT + noise) with exact sizes |
| `testsets/exec_eval_prefill/prompt_assembler.py` | Concatenate SKILL.md blocks into system prompt |
| `testsets/exec_eval_prefill/aggregate.py` | JSONL → pass-rate table (Markdown + JSON) |
| `testsets/skill_selection_eval/retrieve.py` | FAISS-based embedding retrieval for hard/easy noise |
| `testsets/skill_selection_eval/pool_builder.py` | Shared utilities: `_stable_seed`, `_load_clawhub_embeddings` |

### Data Dependencies

| Asset | Path (in container) | Size |
|-------|---------------------|------|
| Skill corpus | `/workspace/procmem2skills/data/processed/skill_corpus.jsonl` | 16 MB (47,231 skills) |
| Embeddings | `/workspace/procmem2skills/data/embeddings/skill_embeddings.npy` | 72 MB (47K x 384 float32) |
| FAISS index | `/workspace/procmem2skills/data/embeddings/index/index.faiss` | ~5 MB |
| Skill metadata | `/workspace/procmem2skills/data/embeddings/skill_metadata.jsonl` | 5.6 MB |
| SB task specs | `/workspace/procmem2skills/testsets/data/skillsbench_tasks.jsonl` | 136 KB (88 tasks) |
| SB task dirs | `/workspace/skillsbench_repo/tasks/<task_id>/` | ~400 MB |

---

## Experiment Phases

### Phase Overview

```
Phase P0: Prefill N=5        672 configs   ← resolves pseudoreplication (seeds 1-4)
Phase B:  Factorial controls  140 configs   ← emptyframe + noiseonly baselines
Phase A:  Baselines N=5       140 configs   ← noskill + gtonly baselines
Phase C:  Original 20-task    168 configs   ← seed=0, 20-task exploratory run
```

### Phase P0: Prefill N=5 (IN PROGRESS)

**Purpose**: Add seeds 1-4 to existing seed=0 data for true variance estimation.
Resolves pseudoreplication critique — 12 configs per condition with seed=0 are
NOT independent samples.

**Design**: 14 tasks x 4 seeds x 4 pool sizes x 3 noise modes = 672 configs

| Parameter | Values |
|-----------|--------|
| Tasks | 14 lightweight SB tasks (6/20 skipped: heavy Dockerfile) |
| Seeds | 1, 2, 3, 4 |
| Pool sizes | 5, 10, 20, 50 |
| Noise modes | random, hard, easy |

**Runner**: `run_prefill_n5.sh`
```bash
# Usage: run_prefill_n5.sh [LIMIT] [CONCURRENCY] [TRIALS_PER_SHARD]
nohup bash run_prefill_n5.sh 20 2 3 > /tmp/prefill_n5.log 2>&1 &
```

**Output**: `results/sb_prefill_n5.jsonl`

### Phase B: Factorial Controls (COMPLETE)

**Purpose**: Test prompt framing effect (emptyframe) and noise priming effect
(noiseonly) to complete the factorial 2x2 design.

**Design**: 14 tasks x 5 seeds x 1 pool size per mode = 140 configs total

| Condition | Pool Size | Seeds | Configs |
|-----------|-----------|-------|---------|
| emptyframe | 0 (no skills) | 0-4 | 70 |
| noiseonly | 10 (noise only, no GT) | 0-4 | 70 |

**Runner**: `run_phase_b.sh`
```bash
nohup bash run_phase_b.sh 20 4 3 > /tmp/phase_b.log 2>&1 &
```

**Output**: `results/sb_phase_b.jsonl`

### Phase A: Baselines N=5 (COMPLETE)

**Purpose**: N=5 seed replication of noskill and gtonly baselines.

| Condition | Seeds | Configs |
|-----------|-------|---------|
| noskill | 0-4 | 70 |
| gtonly | 0-4 | 70 |

**Runner**: `run_baselines_n5.sh`

**Output**: `results/sb_baselines_n5.jsonl`

### Phase C: Original 20-task (COMPLETE)

**Purpose**: Exploratory run with seed=0 across 20 tasks.

| Parameter | Values |
|-----------|--------|
| Tasks | 20 (including 6 heavy-Dockerfile tasks) |
| Seeds | 0 |
| Pool sizes | 5, 10, 20, 50 |
| Noise modes | random, hard, easy |

**Runner**: `run_parallel.sh` / `run_sequential.sh`

**Output**: `results/sb_exec_20t.jsonl`

---

## Already-Run Experiments

### Results Summary (as of 2026-05-17)

| Result File | Condition(s) | Clean Configs | Status |
|-------------|-------------|---------------|--------|
| `sb_exec_20t.jsonl` | random/hard/easy (seed=0, 20 tasks) | 168 (56 per mode) | COMPLETE |
| `sb_baselines_n5.jsonl` | noskill, gtonly (5 seeds, 14 tasks) | 140 (70 each) | COMPLETE |
| `sb_phase_b.jsonl` | emptyframe, noiseonly (5 seeds, 14 tasks) | 140 (70 each) | COMPLETE |
| `sb_prefill_n5.jsonl` | random/hard/easy (seeds 1-4, 14 tasks) | 149/672 | IN PROGRESS |

### Preliminary Pass-Rates (from completed phases)

| Condition | N (configs) | Pass-Rate | Source |
|-----------|------------|-----------|--------|
| noskill | 70 | ~41% | sb_baselines_n5.jsonl |
| emptyframe | 70 | ~39% | sb_phase_b.jsonl |
| noiseonly | 70 | ~41% | sb_phase_b.jsonl |
| gtonly | 70 | ~53% | sb_baselines_n5.jsonl |
| prefill (random) | 56 | ~59% | sb_exec_20t.jsonl |
| prefill (hard) | 56 | ~59% | sb_exec_20t.jsonl |
| prefill (easy) | 56 | ~59% | sb_exec_20t.jsonl |

### Task Difficulty Tiers (from 20-task run)

| Tier | Tasks | Typical Pass-Rate |
|------|-------|-------------------|
| Always-pass | citation-check, 3d-scan-calc, dns-enum | >80% all conditions |
| Skill-sensitive | adaptive-cruise-control, dapt-intrusion-detection | 0% noskill → 60%+ prefill |
| Always-fail | azure-bgp-oscillation-route-leak, civ6-adjacency-optimizer | 0% all conditions |
| Skipped | data-to-d3, court-form-filling, + 4 others | Heavy Dockerfile |

---

## Remaining Experiments

### Priority 1: Complete Phase P0 (Prefill N=5)

```bash
# Check current progress:
python3 -c "
import json, re
rl_pat = re.compile(r'hit your limit|you.ve hit|resets \d+:\d+')
modes = {}
for line in open('results/sb_prefill_n5.jsonl'):
    try:
        r = json.loads(line)
        m = r.get('noise_mode','?')
        stdout = (r.get('agent_stdout_tail') or '').lower()
        if rl_pat.search(stdout): continue
        if r.get('skipped'): continue
        key = (r.get('task_id'), r.get('pool_size'), r.get('seed'))
        if m not in modes: modes[m] = set()
        modes[m].add(key)
    except: pass
for m in sorted(modes):
    print(f'{m}: {len(modes[m])}/224 clean')
print(f'Total: {sum(len(v) for v in modes.values())}/672')
"

# Resume if runner died:
nohup bash run_prefill_n5.sh 20 2 3 > /tmp/prefill_n5.log 2>&1 &
```

**Target**: 672 clean configs (224 per mode = 14 tasks x 4 seeds x 4 pools)

**Important notes**:
- Use `concurrency=2` to avoid rate limits (concurrency=4 causes API throttling)
- The runner processes modes sequentially: random → hard → easy
- `--resume` deduplicates against existing results in the JSONL file
- 6/20 tasks are automatically skipped (heavy Dockerfile); achievable is 14 tasks
- Rate-limit detection: trials with `"hit your limit"` in stdout are excluded from clean count

### Priority 2: Merge Seed-0 Data

After Phase P0 completes, merge seed=0 data from `sb_exec_20t.jsonl` into
the N=5 dataset for the 14 lightweight tasks. This gives seed=0 through seed=4
for complete 5-seed coverage.

```bash
# Extract seed=0 lightweight-task data from 20-task run:
python3 -c "
import json
lightweight = set()  # populate from run_trial.py is_lightweight() filter
for line in open('results/sb_exec_20t.jsonl'):
    r = json.loads(line)
    if r.get('seed') == 0 and r['task_id'] in lightweight:
        print(json.dumps(r))
" >> results/sb_prefill_n5.jsonl
```

### Priority 3: Generate Final Report

```bash
python3 generate_20t_report_v5.py
# Output: results/prefill_eval_report_v5.pdf
```

---

## Runner Scripts Reference

All runners share the same pattern:
1. Spawn Docker containers with `sg docker -c "docker run ..."`
2. Each container runs `python -m exec_eval_prefill.run_trial` with specific args
3. Retry loop with stall detection (15 consecutive stalls → skip mode)
4. Rate-limit filtering in progress counting

### Common Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `$1` (LIMIT) | 20 | Number of tasks to attempt |
| `$2` (CONCURRENCY) | 4 | Parallel Docker containers |
| `$3` (TRIALS_PER) | 3 | Trials per container shard |

### Script Matrix

| Script | Modes | Seeds | Pool Sizes | Output File |
|--------|-------|-------|------------|-------------|
| `run_prefill_n5.sh` | random, hard, easy | 1-4 | 5,10,20,50 | sb_prefill_n5.jsonl |
| `run_prefill_n5_hard_easy.sh` | hard, easy | 1-4 | 5,10,20,50 | sb_prefill_n5.jsonl |
| `run_phase_b.sh` | emptyframe, noiseonly | 0-4 | 0, 10 | sb_phase_b.jsonl |
| `run_baselines_n5.sh` | noskill, gtonly | 0-4 | 0, varies | sb_baselines_n5.jsonl |
| `run_baselines_20t.sh` | noskill, gtonly | 0 | 0, varies | sb_baselines_20t.jsonl |
| `run_parallel.sh` | random, hard, easy | 0 | 5,10,20,50 | sb_exec_20t.jsonl |
| `run_sequential.sh` | random, hard, easy | 0 | 5,10,20,50 | sb_exec_20t.jsonl |

### Running a Custom Experiment

To run a specific condition not covered by existing scripts:

```bash
sg docker -c "docker run --rm \
  -v $HOME/.claude:/home/evaluser/.claude \
  -v $HOME/.claude.json:/home/evaluser/.claude.json \
  -v $PWD/results:/workspace/results \
  -w /workspace/procmem2skills \
  -e PYTHONPATH=/workspace/procmem2skills/testsets \
  -e SKILL_CORPUS_PATH=/workspace/procmem2skills/data/processed/skill_corpus.jsonl \
  -e SKILL_INDEX_PATH=/workspace/procmem2skills/data/embeddings/index/index.faiss \
  -e SKILL_META_PATH=/workspace/procmem2skills/data/embeddings/skill_metadata.jsonl \
  -e SKILL_EMBEDDINGS_PATH=/workspace/procmem2skills/data/embeddings/skill_embeddings.npy \
  skills-testset-eval-eval-pipeline \
  -c 'python -u -m exec_eval_prefill.run_trial \
    --dataset sb \
    --tasks /workspace/procmem2skills/testsets/data/skillsbench_tasks.jsonl \
    --out /workspace/results/YOUR_OUTPUT.jsonl \
    --work-root /tmp/exec_prefill \
    --model claude-sonnet-4-6 \
    --pool-sizes 5 10 20 50 \
    --noise-modes random hard easy \
    --seeds 0 1 2 3 4 \
    --concurrency 1 \
    --python python \
    --limit 20 \
    --max-trials 3 \
    --skip-trials 0 \
    --resume 2>&1'"
```

### run_trial.py CLI Arguments

| Argument | Description |
|----------|-------------|
| `--dataset` | `sb` (SkillsBench) or `tb` (TerminalBench) |
| `--tasks` | Path to task JSONL file |
| `--out` | Output JSONL path |
| `--work-root` | Temp directory for trial workspaces |
| `--model` | Claude model ID |
| `--pool-sizes` | Space-separated pool sizes (e.g., `5 10 20 50`) |
| `--noise-modes` | Space-separated noise modes (e.g., `random hard easy`) |
| `--seeds` | Space-separated seed values |
| `--concurrency` | Max parallel trials within container |
| `--python` | Python binary for pytest (default: `python`) |
| `--limit` | Max tasks to process |
| `--max-trials` | Max trials per container invocation |
| `--skip-trials` | Offset for parallel sharding (container N skips first N*TRIALS_PER) |
| `--resume` | Skip already-completed (task_id, pool_size, noise_mode, seed) tuples |

---

## Result Data Format

Each line in the JSONL output files is a JSON object:

```json
{
  "trial_id": "sb__3d-scan-calc__sz5__random__s0",
  "dataset": "sb",
  "task_id": "3d-scan-calc",
  "pool_size": 5,
  "noise_mode": "random",
  "seed": 0,
  "gt_slugs": ["mesh-analysis"],
  "gt_positions": [2],
  "n_candidates": 5,
  "model": "claude-sonnet-4-6",
  "agent_rc": 0,
  "agent_wall_s": 34.7,
  "pass": true,
  "test_log_tail": "2 passed in 0.01s",
  "agent_stdout_tail": "...",
  "system_prompt_chars": 5311,
  "skipped": false
}
```

### Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `pass` | bool | `true` if pytest exit code = 0 |
| `skipped` | bool | `true` if task was skipped (heavy Dockerfile, TB, etc.) |
| `agent_stdout_tail` | string | Last ~2KB of claude stdout (check for rate-limit messages) |
| `noise_mode` | string | `random`, `hard`, `easy`, `noskill`, `gtonly`, `emptyframe`, `noiseonly` |
| `seed` | int | Random seed for pool construction |

### Counting Clean Results

Rate-limited trials must be excluded. Use this pattern:

```python
import json, re
rl_pat = re.compile(r"hit your limit|you've hit|resets \d+:\d+")
seen = set()
for line in open("results/sb_prefill_n5.jsonl"):
    r = json.loads(line)
    stdout = (r.get("agent_stdout_tail") or "").lower()
    if rl_pat.search(stdout): continue  # rate-limited
    if r.get("skipped"): continue       # skipped task
    seen.add((r["task_id"], r["pool_size"], r["noise_mode"], r["seed"]))
print(f"Clean configs: {len(seen)}")
```

---

## Analysis & Report Generation

### Report Scripts

| Script | Input Files | Output |
|--------|-------------|--------|
| `generate_20t_report_v5.py` | sb_exec_20t.jsonl, sb_baselines_n5.jsonl, sb_phase_b.jsonl | prefill_eval_report_v5.pdf |
| `analysis_paired.py` | sb_exec_20t.jsonl, sb_baselines_20t.jsonl | Paired difference analysis |
| `analysis_error_taxonomy.py` | sb_exec_20t.jsonl | Error categorization |

### Aggregation

```bash
# Generate pass-rate table from any JSONL file:
python -m exec_eval_prefill.aggregate \
  --results results/sb_prefill_n5.jsonl \
  --json results/prefill_n5_table.json \
  --md results/prefill_n5_table.md
```

---

## Troubleshooting

### Rate Limiting

**Symptom**: Stall count increases, `agent_stdout_tail` contains "hit your limit"

**Fix**: Reduce concurrency. `concurrency=2` is safe; `concurrency=4` may trigger
rate limits depending on API quota.

```bash
# Check rate-limited trial count:
python3 -c "
import json, re
rl = re.compile(r'hit your limit|you.ve hit|resets \d+:\d+')
count = sum(1 for l in open('results/sb_prefill_n5.jsonl')
            if rl.search(json.loads(l).get('agent_stdout_tail','').lower()))
print(f'Rate-limited trials: {count}')
"
```

### Runner Crashed / Died

All runners support `--resume` and can be safely restarted:

```bash
# Check if runner is alive:
ps aux | grep run_prefill_n5 | grep -v grep

# Restart:
nohup bash run_prefill_n5.sh 20 2 3 > /tmp/prefill_n5.log 2>&1 &
```

### Docker Image Not Found

```bash
# Rebuild:
cd skills-testset-eval
docker compose build
# Or directly:
docker build -t skills-testset-eval-eval-pipeline .
```

### Skipped Tasks

6/20 SkillsBench tasks are skipped due to heavy Dockerfile dependencies
(nodejs, playwright, poppler, etc.). This is by design — the `is_lightweight()`
filter in `run_trial.py` detects these. Achievable clean configs per mode
is 14 tasks, not 20.

### sg docker Permission

All `docker run` commands are wrapped in `sg docker -c "..."` for group
permission. If your user is already in the docker group, you can remove the
`sg docker -c` wrapper.
