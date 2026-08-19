# Clean Experiment Pipeline

This directory contains the reusable PM2Skills experiment code, prompts, configuration templates, and tests. It intentionally excludes trajectories, generated skills, logs, Docker state, API keys, OAuth tokens, and historical outputs.

The experiment stages are exposed as direct scripts under `scripts/`; there is no extra command wrapper.

## End-to-end map

The normal data flow is:

    raw trajectories -> mixed collection -> qualified workflows
        -> normal/no-hint skills or compact procedures
        -> workflow/skill/compact injection evaluation

Retrieval is a separate branch that starts from a completed skill pool:

    SkillsBench task-skill map + skill pool
        -> candidate manifests and pools
        -> Arm 1 embedding retrieval
        -> Arm 2 agent selection
        -> Arm 3 real execution and activated-skill parsing
        -> common precision/recall/F1/success summaries

For one controlled comparison, keep the following fields fixed across arms:
benchmark, task set, ground-truth skill set, candidate-pool mode, pool size, seed, agent, model, provider, and trial count. Change only the representation or retrieval arm being evaluated.

## Input/output contracts

- Raw and collection scripts consume Harbor task data and write trial directories containing result metadata and trajectories.
- Workflow induction consumes the raw/collection roots plus a status log and writes one workflow JSON and one metadata JSON.
- Skill and compact-procedure generation consume the workflow JSON and write condition-specific artifacts under separate output roots.
- Injection consumes trace roots and the selected workflow/skill/procedure roots and writes per-run reports, trial outputs, and condition summaries.
- Retrieval preparation writes manifests and candidate pools under the retrieval root; each retrieval arm then writes to its own benchmark/agent-model/mode/pool-size/seed directory.

Use explicit output paths and a unique run ID when changing the agent, model, provider, benchmark, or pool setting. Never compare two results only by directory modification time.

## Recommended smoke-test order

Before a long Docker batch:

1. Verify the API key or subscription with the provider-specific smoke test in Section 2.
2. Run one task with one trial and low concurrency.
3. Inspect one result JSON and one trajectory for the expected agent/model metadata.
4. Only then increase N_ATTEMPTS and N_CONCURRENT.
5. For retrieval Arm 3, first run one candidate manifest and confirm the activated-skill parser produces a non-empty per-trial record.

## 1. Configure external data and runtime

Run commands from the repository root:

    cd <pm2s-clean>

Set external roots in the private shell used to launch the experiment:

    export PROCMEM2SKILLS_ROOT=/path/to/procmem2skills
    export TASK_SOURCE_ROOT=/path/to/procmem2skills/benchmarks/harbor-datasets
    export SKILLSBENCH_ROOT=/path/to/skillsbench
    export PM2S_RESULTS_ROOT=/path/to/results
    export PM2S_RETRIEVAL_ROOT=/path/to/skill-retrieval-data

Harbor and its Python environment are resolved from `PROCMEM2SKILLS_ROOT`. Dataset files, retrieval pools, and all generated outputs stay outside this code directory.

## 2. Credentials and subscriptions

### API-key agents

Set a provider-specific variable only in the shell that launches the run. Pass the variable name, never the value:

    export GOOGLE_API_KEY=<private-value>

Gemini example:

    DATASET=terminal-bench@2.0 \
    AGENT=gemini-cli \
    MODEL=google/gemini-3.1-pro-preview \
    PROVIDER=google \
    API_KEY_ENV=GOOGLE_API_KEY \
    BASE_URL=https://generativelanguage.googleapis.com/v1beta \
    bash scripts/run_raw.sh

For Google runs, the launcher forwards the selected value to both environment names expected by Gemini CLI. The key is never written into manifests, prompts, source files, or result metadata. The same pattern applies to `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `UNIAPI_API_KEY`, and `ANTHROPIC_API_KEY`.

A 401 or 403 is an authentication or permission failure. A 429 is quota or rate limiting. Do not rotate credentials in the middle of a batch. Stop or quarantine that batch, fix the private shell environment, and start a fresh run ID. Collection and injection loops record a failed condition and continue to later conditions, but a quota error should be treated as a batch-level incident.

### Codex subscription

Authenticate Codex interactively as the same Unix user and runtime that launches Harbor:

    codex exec --model <model> --skip-git-repo-check -- "Reply OK only"

Do not copy OAuth files into this directory or convert a subscription token into an API-key variable. If the smoke command fails, do not launch a batch.

### Claude Code subscription

Use a private temporary `CLAUDE_CONFIG_DIR`, complete the interactive login, and smoke-test it before starting Harbor:

    export CLAUDE_CONFIG_DIR=/path/to/private-claude-config
    claude auth login --claudeai
    CLAUDE_CONFIG_DIR="$CLAUDE_CONFIG_DIR" claude -p "Reply OK only" --model <model> --no-session-persistence

For Claude Code real execution, set `AGENT=claude-code`, `MODEL=<model>`, and `USE_CLAUDE_CODE_OAUTH=1`. The real-execution script requires either `CLAUDE_CODE_CREDENTIALS_FILE` or `CLAUDE_CONFIG_DIR`; it performs a host-side smoke test before Docker jobs. Credential contents and passwords never enter this directory.

Safe diagnostics may report only whether a variable is set and its length. Do not print full values, shell history, process arguments, or Docker inspect output when credentials are present.

## 3. Raw trajectory runs

Direct entrypoint: `scripts/run_raw.sh`.

It reads configuration from environment variables and passes extra arguments to Harbor:

    DATASET=terminal-bench@2.0 \
    AGENT=gemini-cli \
    MODEL=google/gemini-3.1-pro-preview \
    PROVIDER=google \
    API_KEY_ENV=GOOGLE_API_KEY \
    BASE_URL=https://generativelanguage.googleapis.com/v1beta \
    N_ATTEMPTS=5 \
    N_CONCURRENT=10 \
    TASK_NAME=<task-name> \
    bash scripts/run_raw.sh

Important controls include `DATASET_PATH`, `RESULTS_ROOT`, `JOBS_DIR`, `TASK_NAME`, `EXCLUDE_TASK_NAME`, `N_TASKS`, `ENV_TYPE`, `PREPULL_IMAGES`, and the provider variables above. Use a unique result/job name for every agent-model-provider combination.

## 4. Mixed trace collection

Direct entrypoint: `scripts/collect_traces.sh`.

It adds trials only for tasks that still need success/failure evidence:

    DATASET=terminal-bench@2.0 \
    RAW_RESULTS_PATH=/path/to/raw-results \
    OUTPUT_PATH=/path/to/collect-rounds \
    AGENT=gemini-cli \
    MODEL=google/gemini-3.1-pro-preview \
    PROVIDER=google \
    API_KEY_ENV=GOOGLE_API_KEY \
    BASE_URL=https://generativelanguage.googleapis.com/v1beta \
    TRIALS_PER_ROUND=5 \
    MAX_ATTEMPT_BLOCKS=10 \
    TARGET_SUCCESS=5 \
    TARGET_FAILURE=5 \
    N_CONCURRENT=10 \
    bash scripts/collect_traces.sh

`TRIALS_PER_ROUND` is the number of new trials per task in one block. `MAX_ATTEMPT_BLOCKS` is the maximum number of blocks. `CONTINUE_FROM_EXISTING=1` resumes an existing collection; `CLEANUP_AFTER_ROUND=1` enables the configured Docker cleanup behavior.

## 5. Workflow induction

Direct entrypoint: `scripts/export_workflows.py`.

It selects status-qualified raw and collected traces and writes workflow records plus metadata:

    python3 scripts/export_workflows.py \
      --status-log /path/to/collect/status.log \
      --raw-root /path/to/raw \
      --collect-root /path/to/collect \
      --workflow-out /path/to/workflows.json \
      --metadata-out /path/to/workflows.meta.json \
      --success-per-task 5 \
      --failure-per-task 5 \
      --enable-cleaning

The workflow input is the shared source for normal skills, no-hint skills, and compact procedural baselines.

## 6. Workflow to SKILL.md

Direct entrypoint: `scripts/generate_skills.py`.

Normal skills include trace outcome labels:

    export GOOGLE_API_KEY=<private-value>
    python3 scripts/generate_skills.py \
      --workflow-input /path/to/workflows.json \
      --output-root /path/to/outputs \
      --benchmark terminalbench2 \
      --agent gemini-cli \
      --model google/gemini-3.1-pro-preview \
      --provider google \
      --api-key-env GOOGLE_API_KEY \
      --base-url https://generativelanguage.googleapis.com/v1beta \
      --skill-creator-model google/gemini-3.1-pro-preview \
      --system-prompt-file configs/skill_generator_system_prompt.txt \
      --hint-mode with-status \
      --resume

No-hint skills use the same workflows but hide outcome labels:

    python3 scripts/generate_skills.py \
      --workflow-input /path/to/workflows.json \
      --output-root /path/to/outputs \
      --benchmark terminalbench2 \
      --agent gemini-cli \
      --model google/gemini-3.1-pro-preview \
      --provider google \
      --api-key-env GOOGLE_API_KEY \
      --base-url https://generativelanguage.googleapis.com/v1beta \
      --skill-creator-model google/gemini-3.1-pro-preview \
      --system-prompt-file configs/skill_generator_system_prompt_no_hint.txt \
      --hint-mode no-hint \
      --resume

The output root and `--skills-subdir` should be kept separate for different benchmark, agent, model, and hint-mode settings.

## 7. Compact procedural baselines

Direct entrypoint: `scripts/generate_compact_procedures.py`.

It can generate short plans, test-first templates, and reusable scripts from the same workflow input:

    python3 scripts/generate_compact_procedures.py \
      --workflow-input /path/to/workflows.json \
      --source workflow \
      --output-root /path/to/compact-procedures \
      --benchmark terminalbench2 \
      --conditions 5s0f \
      --forms short-plan,test-first,script \
      --agent gemini-cli \
      --model google/gemini-3.1-pro-preview \
      --provider google \
      --api-key-env GOOGLE_API_KEY \
      --base-url https://generativelanguage.googleapis.com/v1beta \
      --procedure-creator-model google/gemini-3.1-pro-preview \
      --hint-mode no-hint \
      --resume

Use `--source instruction` with `--task-source-root` when the baseline should be generated from task instructions rather than workflow traces.

## 8. Workflow and skill injection

Main direct entrypoint: `scripts/run_eval.sh`. It calls `scripts/run_context_comparison.py` and supports `baseline`, `workflow`, `skill`, `short-plan`, `test-first`, and `script` arms.

Workflow injection:

    bash scripts/run_eval.sh \
      --trace-root /path/to/traces \
      --skills-root /path/to/skills \
      --output-root /path/to/outputs \
      --benchmark-output terminalbench2 \
      --provider google \
      --api-key-env GOOGLE_API_KEY \
      --base-url https://generativelanguage.googleapis.com/v1beta \
      --agent gemini-cli \
      --model google/gemini-3.1-pro-preview \
      --m-success 5 --n-failure 0 \
      --n-attempts 5 --n-concurrent 10 \
      --arms workflow \
      --run-id workflow-terminalbench2-gemini

Skill injection uses `--arms skill`. Run both controlled arms with `--arms workflow,skill`. Use `--workflow-hint-mode no-hint` only when the workflow source is the no-hint condition. Docker cleanup is enabled by default; use `--no-docker-cleanup` only when another supervisor owns cleanup. A failed setting is recorded and does not silently become a successful result.

## 9. Retrieval experiments

The retrieval pipeline has three arms and a shared directory layout:

    <retrieval-root>/<benchmark>/<agent-model>/
      embedding_based/<mode>/k<pool-size>/seed-<seed>/
      agent_pick/<mode>/k<pool-size>/seed-<seed>/
      real_execution/<mode>/k<pool-size>/seed-<seed>/

### 9.1 Build manifests and candidate pools

    python3 scripts/retrieval/build_skillsbench_manifests.py \
      --root /path/to/retrieval-data \
      --tasks-root /path/to/skillsbench/tasks \
      --benchmark skillsbench \
      --overwrite

    python3 scripts/retrieval/filter_noise_pool.py \
      --root /path/to/retrieval-data \
      --noise-pool /path/to/noise-pool \
      --output /path/to/retrieval-data/manifests/noise.jsonl \
      --english-only

    python3 scripts/retrieval/build_candidate_pools.py \
      --root /path/to/retrieval-data \
      --benchmark skillsbench \
      --pool-sizes 5,10,20,50,100 \
      --noise-modes random,similar,dissimilar \
      --seeds 42

### 9.2 Arm 1: embedding-based retrieval

Direct entrypoint: `scripts/retrieval/run_embedding_retrieval.py`.

Local embedding requires no API key:

    python3 scripts/retrieval/run_embedding_retrieval.py \
      --root /path/to/retrieval-data \
      --candidate-pool /path/to/candidate-pool.json \
      --method local-embedding \
      --local-model-path /path/to/Qwen3-Embedding-0.6B \
      --benchmark skillsbench \
      --agent embedding \
      --model Qwen3-Embedding-0.6B \
      --top-k 1

Use `--top-k 3` or `--top-k 5` for set-level precision, recall, and F1. Top-1 intentionally reports recall and F1 as NA. `--method lexical` is an API-free debugging baseline.

### 9.3 Arm 2: let agents pick

Direct entrypoint: `scripts/retrieval/run_agent_pick.py`.

This presents the task and candidate skill pool to the agent without executing the downstream task:

    python3 scripts/retrieval/run_agent_pick.py \
      --root /path/to/retrieval-data \
      --skills-manifest /path/to/skill-manifest.json \
      --benchmark skillsbench \
      --agent gemini-cli \
      --provider google \
      --api-key-env GOOGLE_API_KEY \
      --base-url https://generativelanguage.googleapis.com/v1beta/openai \
      --model google/gemini-3.1-pro-preview \
      --n-attempts 1 --n-concurrent 10

The output stores selected skills and set-level precision, recall, and F1 against the manifest ground truth.

### 9.4 Arm 3: real execution

Direct entrypoint: `scripts/retrieval/run_real_execution.sh`.

It injects the candidate pool into the execution environment, runs the downstream task, and parses the trajectory for activated skills:

    CANDIDATE_MANIFEST=/path/to/candidate-manifest.json \
    ROOT=/path/to/retrieval-data \
    BENCHMARK=skillsbench \
    AGENT=gemini-cli \
    MODEL=google/gemini-3.1-pro-preview \
    PROVIDER=google \
    API_KEY_ENV=GOOGLE_API_KEY \
    BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai \
    N_ATTEMPTS=5 N_CONCURRENT=10 \
    bash scripts/retrieval/run_real_execution.sh

For Claude Code OAuth, set `AGENT=claude-code`, `USE_CLAUDE_CODE_OAUTH=1`, and either `CLAUDE_CODE_CREDENTIALS_FILE` or `CLAUDE_CONFIG_DIR` in the private shell. The script smoke-tests the host CLI before launching Docker.

Summarize real execution and parsed activation:

    python3 scripts/retrieval/summarize_execution_retrieval.py \
      --root /path/to/retrieval-data \
      --candidate-manifest /path/to/candidate-manifest.json \
      --run-root /path/to/real-execution-run \
      --benchmark skillsbench \
      --agent gemini-cli \
      --model google/gemini-3.1-pro-preview

## 10. Optional transfer analyses

    python3 scripts/build_crossbench_task_similarity.py --help
    python3 scripts/run_crossbench_similarity_skill_eval.py --help
    python3 scripts/run_skill_transferability_eval.py --help

These scripts are kept separate from the three retrieval arms because they analyze cross-benchmark transfer rather than the retrieval protocol itself.

## 11. Reproducibility and safety

- Keep code here and data in explicit external roots.
- Never reuse a run ID for another agent, model, or provider.
- Keep normal and no-hint outputs separate.
- Use the same candidate manifest for all retrieval arms of one setting.
- Record agent, model, provider, pool mode, pool size, seed, trials, concurrency, and run ID.
- Check Docker capacity before concurrent runs.
- These entrypoints do not globally prune unrelated containers or images.
