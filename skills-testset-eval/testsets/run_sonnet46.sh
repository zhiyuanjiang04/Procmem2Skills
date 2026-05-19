#!/usr/bin/env bash
# Experiment pipeline: Sonnet 4.6 skill-selection eval (SB + TB)
#
# Design:
#   Step 1 — GT skills are placed in the pool for every trial.
#   Step 2 — Distractors fill remaining slots.
#   Pool sizes : 5 | 50 | 500
#   Distractor types: random | hard (embedding-near) | easy (embedding-far)
#
# Isolation: SB and TB run write to separate output files.
#            Each (task_id, pool_size, noise_mode, seed) is uniquely keyed;
#            --resume guarantees no trial is re-run if output already exists.
#
# Usage:
#   cd /anvil/projects/x-cis260386/william/procmem2skills/procmem2skills
#   bash testsets/run_sonnet46.sh [--smoke] [--resume]
#
# Set ANTHROPIC_API_KEY before running.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTSETS_DIR="$REPO_ROOT/testsets"
RUNS_DIR="$TESTSETS_DIR/runs"
DATA_DIR="$TESTSETS_DIR/data"
MODULE="skill_selection_eval"

# Use the project-wide marl venv (has faiss, sentence-transformers, anthropic).
PYTHON="${PYTHON:-/anvil/projects/x-cis260386/william/llm-colearning/envs/marl/bin/python}"

MODEL="claude-sonnet-4-6"
POOL_SIZES="5 50 500"
NOISE_MODES="random hard easy"
SEEDS="0"
CONCURRENCY=6
MAX_GT=1

SB_TASKS="$DATA_DIR/skillsbench_tasks.jsonl"
TB_TASKS="$DATA_DIR/terminal_bench_validated.jsonl"

# New results go in a dedicated folder — easy to git add / push independently.
EVAL_DIR="$TESTSETS_DIR/eval_sonnet46"
SB_OUT="$EVAL_DIR/sb_sonnet46.jsonl"
TB_OUT="$EVAL_DIR/tb_sonnet46.jsonl"
SUMMARY_OUT="$EVAL_DIR/sonnet46_table.json"

export PYTHONPATH="$TESTSETS_DIR:${PYTHONPATH:-}"

SMOKE=0
RESUME=""
LIMIT_ARG=""

for arg in "$@"; do
  case "$arg" in
    --smoke)  SMOKE=1; LIMIT_ARG="--limit 5" ;;
    --resume) RESUME="--resume" ;;
  esac
done

mkdir -p "$EVAL_DIR"

if [ "$SMOKE" -eq 1 ]; then
  echo "[SMOKE MODE] Limiting to 5 tasks per dataset"
  SB_OUT="$EVAL_DIR/sb_sonnet46_smoke.jsonl"
  TB_OUT="$EVAL_DIR/tb_sonnet46_smoke.jsonl"
  SUMMARY_OUT="$EVAL_DIR/sonnet46_smoke_table.json"
fi

cd "$REPO_ROOT"

# ----- Preflight: verify large data files exist -----
CORPUS="${SKILL_CORPUS_PATH:-$REPO_ROOT/data/processed/skill_corpus.jsonl}"
INDEX="${SKILL_INDEX_PATH:-$REPO_ROOT/data/embeddings/index/index.faiss}"
META="${SKILL_META_PATH:-$REPO_ROOT/data/embeddings/skill_metadata.jsonl}"
EMBEDDINGS="${SKILL_EMBEDDINGS_PATH:-$REPO_ROOT/data/embeddings/skill_embeddings.npy}"
MISSING=0
for f in "$CORPUS" "$INDEX" "$META" "$EMBEDDINGS"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing data file: $f"
    MISSING=1
  fi
done
if [ "$MISSING" -eq 1 ]; then
  echo ""
  echo "These large files are not tracked by git. Set env vars to override paths:"
  echo "  export SKILL_CORPUS_PATH=/path/to/skill_corpus.jsonl"
  echo "  export SKILL_INDEX_PATH=/path/to/index.faiss"
  echo "  export SKILL_META_PATH=/path/to/skill_metadata.jsonl"
  echo ""
  echo "Or symlink them into the expected locations:"
  echo "  mkdir -p data/processed data/embeddings/index"
  echo "  ln -s /path/to/skill_corpus.jsonl data/processed/skill_corpus.jsonl"
  echo "  ln -s /path/to/index.faiss data/embeddings/index/index.faiss"
  echo "  ln -s /path/to/skill_metadata.jsonl data/embeddings/skill_metadata.jsonl"
  exit 1
fi

export SKILL_CORPUS_PATH="$CORPUS"
export SKILL_INDEX_PATH="$INDEX"
export SKILL_META_PATH="$META"
export SKILL_EMBEDDINGS_PATH="$EMBEDDINGS"

echo "======================================================================"
echo "  Sonnet 4.6 Skill-Selection Eval — $(date '+%Y-%m-%d %H:%M')"
echo "  Model  : $MODEL"
echo "  Pools  : $POOL_SIZES"
echo "  Modes  : $NOISE_MODES"
echo "  Seeds  : $SEEDS"
echo "======================================================================"

# ----- Stage 1: SkillsBench (88 tasks) -----
echo ""
echo ">>> [1/2] SkillsBench — $(wc -l < "$SB_TASKS") tasks"
echo "    Output: $SB_OUT"

"$PYTHON" -m "$MODULE.run_unified" \
  --dataset sb \
  --tasks "$SB_TASKS" \
  --out "$SB_OUT" \
  --model "$MODEL" \
  --pool-sizes $POOL_SIZES \
  --noise-modes $NOISE_MODES \
  --seeds $SEEDS \
  --max-gt "$MAX_GT" \
  --concurrency "$CONCURRENCY" \
  $RESUME \
  $LIMIT_ARG

echo ">>> SkillsBench complete."

# ----- Stage 2: TerminalBench validated (62 tasks) -----
echo ""
echo ">>> [2/2] TerminalBench (validated) — $(wc -l < "$TB_TASKS") tasks"
echo "    Output: $TB_OUT"

"$PYTHON" -m "$MODULE.run_unified" \
  --dataset tb \
  --tasks "$TB_TASKS" \
  --out "$TB_OUT" \
  --model "$MODEL" \
  --pool-sizes $POOL_SIZES \
  --noise-modes $NOISE_MODES \
  --seeds $SEEDS \
  --concurrency "$CONCURRENCY" \
  $RESUME \
  $LIMIT_ARG

echo ">>> TerminalBench complete."

# ----- Stage 3: Generate table -----
echo ""
echo ">>> [3/3] Generating results table"

"$PYTHON" -m "$MODULE.report_table" \
  --results "$SB_OUT" "$TB_OUT" \
  --json "$SUMMARY_OUT"

echo ""
echo "======================================================================"
echo "  Done. Summary JSON: $SUMMARY_OUT"
echo "======================================================================"
