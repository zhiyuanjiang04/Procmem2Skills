# Artifacts for local run @ 192.168.50.47

## Tarballs (in this directory)

| file | size | what's inside |
|---|---|---|
| `procmem2skills_pipeline.tar.gz` | 136M | All pipeline code + corpus + embeddings + results |
| `skillsbench_tasks.tar.gz` | 412M | SkillsBench task dirs (instruction/tests/env/solution) + terminus-2-skills agent |
| `tb_validated_tasks.tar.gz` | 5.0M | TerminalBench 62 validated task dirs |

## SHA256
fb0697b87cba6b414a5ae02e8d3562bae7fe958ec3170b5cb91379f3d483ea27  procmem2skills_pipeline.tar.gz
5034f3455b8dbd6382e4058df58470ab96772487cdc870010e1b3c0ac90a1251  skillsbench_tasks.tar.gz
19ebc5459ef74225535973cbcfa199955973d77d3ad52d1679a657db879ff8ee  tb_validated_tasks.tar.gz


## Layout after extracting all three

```
<your-local-root>/
├── procmem2skills/                          # from procmem2skills_pipeline.tar.gz
│   ├── testsets/
│   │   ├── exec_eval_prefill/               # NEW: prefill-context execution eval
│   │   │   ├── pool_builder_v3.py
│   │   │   ├── prompt_assembler.py
│   │   │   ├── run_trial.py
│   │   │   ├── aggregate.py
│   │   │   └── README.md
│   │   ├── exec_eval/                       # Docker-ready scaffold (alt path)
│   │   ├── skill_selection_eval/            # selection-proxy harness (already-run)
│   │   ├── data/
│   │   │   ├── skillsbench_tasks.jsonl      # 88 SB task specs (gt_skills, descs)
│   │   │   └── terminal_bench_validated.jsonl  # 62 TB task specs
│   │   ├── embeddings/                      # task / GT embeddings (small)
│   │   ├── eval_passrate_sonnet46/          # already-run selection-proxy results
│   │   ├── eval_exec_prefill_sonnet46/      # smoke results from prefill-context run
│   │   ├── run_exec_prefill_sonnet46.sh     # full SB grid orchestrator
│   │   ├── run_pass_rate_sonnet46.sh        # selection-proxy orchestrator
│   │   └── run_sonnet46.sh                  # original 3-noise 5/50/500 grid
│   └── data/
│       ├── processed/skill_corpus.jsonl     # 47,231 ClawHub skills (16M)
│       └── embeddings/                      # FAISS + BGE-small embeddings (145M)
│           ├── skill_metadata.jsonl
│           ├── skill_embeddings.npy         # 47K × 384 float32
│           └── index/index.faiss
│
├── skillsbench_repo/                        # from skillsbench_tasks.tar.gz
│   ├── tasks/                               # 89 SB tasks: instruction, env (Dockerfile),
│   │                                        # tests, solution, environment/skills/<gt>/
│   └── libs/terminus_agent/                 # harbor's terminus-2-skills agent
│
└── terminal-bench/                          # from tb_validated_tasks.tar.gz
    └── original-tasks/                      # 62 validated TB task dirs
```

## How to pull from your local server

```bash
# From 192.168.50.47:
cd /path/to/where/you/want/it
scp x-hluo4@anvil.rcac.purdue.edu:/anvil/projects/x-cis260386/william/procmem2skills/_artifacts_for_local/'*.tar.gz' .
tar -xzf procmem2skills_pipeline.tar.gz
tar -xzf skillsbench_tasks.tar.gz
tar -xzf tb_validated_tasks.tar.gz
```

This creates `procmem2skills/`, `skillsbench_repo/`, `terminal-bench/` side by side.
The pipeline's run_trial.py expects this exact layout (it resolves SB task dirs via
`../skillsbench_repo/tasks/<task_id>/` and TB via `../terminal-bench/original-tasks/`).

## How to run locally

### Setup
```bash
# Python env: needs faiss-cpu, numpy, sentence-transformers (for the BGE-small encoder
# used by hard/easy noise modes), pytest, anthropic (for SDK backend), and the
# `claude` CLI (for Max-plan / CLI backend).
python -m venv .venv && source .venv/bin/activate
pip install faiss-cpu numpy pandas sentence-transformers pytest anthropic

# Set corpus paths (the runner reads these env vars)
export SKILL_CORPUS_PATH=$PWD/procmem2skills/data/processed/skill_corpus.jsonl
export SKILL_INDEX_PATH=$PWD/procmem2skills/data/embeddings/index/index.faiss
export SKILL_META_PATH=$PWD/procmem2skills/data/embeddings/skill_metadata.jsonl
export SKILL_EMBEDDINGS_PATH=$PWD/procmem2skills/data/embeddings/skill_embeddings.npy

# Make `claude` CLI available, Max-plan logged in OR ANTHROPIC_API_KEY=...
```

### Run the execution eval (SB)

```bash
cd procmem2skills

# Smoke (5 tasks × 12 conditions = 60 trials, ~30-60 min depending on rate)
bash testsets/run_exec_prefill_sonnet46.sh --smoke

# Full (84 lightweight SB tasks × 4 sizes × 3 noise = 1008 trials, ~6-8h)
bash testsets/run_exec_prefill_sonnet46.sh

# Resume after interruption
bash testsets/run_exec_prefill_sonnet46.sh --resume

# Concurrency override (default 4; raise on bigger box, lower if rate-limited)
CONCURRENCY=8 bash testsets/run_exec_prefill_sonnet46.sh
```

Output: `testsets/eval_exec_prefill_sonnet46/sb_exec.jsonl` + `exec_table.md` + JSON.

### Run the already-defined selection-proxy eval (cheap baseline)

```bash
bash testsets/run_pass_rate_sonnet46.sh
```

Output: `testsets/eval_passrate_sonnet46/passrate_table.md`.

## Notes

1. **TB execution not yet implemented for the prefill-context path** — TB tasks
   assume Docker container environments. The pipeline's `run_trial.py` currently
   skips TB with reason "tb host-execution not implemented (needs container)".
   To enable: build per-task Apptainer/Docker images, run agent inside container,
   bind-mount workspace. Scaffolding in `testsets/exec_eval/`.

2. **SB heavy-Dockerfile filter**: 5/89 SB tasks have Dockerfiles needing
   nodejs/playwright/poppler/tex/etc. (e.g., `data-to-d3`). Pipeline skips these
   automatically with `reason=heavy Dockerfile`. To enable: install those system
   deps locally OR run in container.

3. **Path patcher**: `run_trial.py` rewrites `/root/`, `/data/`, `/output/`, `/logs/`,
   `/app/`, `/tests/`, `/home/`, `/workspace/` → workspace subdirs, and mirrors
   Dockerfile `COPY src dst` mappings into the workspace. Most SB tasks should
   work; very custom layouts may need extra rewrites.

4. **Smoke pre-run on Anvil saw rate-limit collisions** with other sessions; on
   a dedicated local box this should clear.
