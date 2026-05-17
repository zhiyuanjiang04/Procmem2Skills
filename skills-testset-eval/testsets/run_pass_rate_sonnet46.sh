#!/usr/bin/env bash
# Sonnet 4.6 pass-rate-style eval over the n_noise grid.
#
# Spec (user 2026-05-13):
#   Step 1: GT skills always in the pool (all of them, multi-GT preserved).
#   Step 2: random distractors fill remaining slots.
#   Conditions: n_noise ∈ {0, 1, 5, 10, 20, 50}  (i.e. GT-only, GT+1, ..., GT+50)
#   Datasets: SkillsBench (88), TerminalBench-validated (62)
#
# Each (task_id, n_noise, seed) trial is independent. SB / TB write to separate
# output files. --resume skips completed trials.
#
# Pass-rate semantics (no-Docker proxy):
#   Reports the rate at which Sonnet 4.6 picks a GT skill from the pool. With
#   the correct skill loaded in a real terminus-2-skills run, the agent has the
#   strongest possible guidance for the task. So this hit-rate upper-bounds
#   true execution pass-rate. Real execution-pass requires Docker (not on Anvil)
#   plus the full harbor harness; pipeline scaffolding for that lives in
#   testsets/exec_eval/ (to be added).
#
# Usage:
#   cd /anvil/projects/x-cis260386/william/procmem2skills/procmem2skills
#   bash testsets/run_pass_rate_sonnet46.sh [--smoke] [--resume]
#
# Auth: claude -p Max plan (no ANTHROPIC_API_KEY needed).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTSETS_DIR="$REPO_ROOT/testsets"
DATA_DIR="$TESTSETS_DIR/data"
MODULE="skill_selection_eval"

PYTHON="${PYTHON:-/anvil/projects/x-cis260386/william/llm-colearning/envs/marl/bin/python}"

MODEL="claude-sonnet-4-6"
N_NOISE="0 1 5 10 20 50"
SEEDS="0"
CONCURRENCY=6

SB_TASKS="$DATA_DIR/skillsbench_tasks.jsonl"
TB_TASKS="$DATA_DIR/terminal_bench_validated.jsonl"

EVAL_DIR="$TESTSETS_DIR/eval_passrate_sonnet46"
SB_OUT="$EVAL_DIR/sb_passrate.jsonl"
TB_OUT="$EVAL_DIR/tb_passrate.jsonl"
SUMMARY_JSON="$EVAL_DIR/passrate_table.json"
SUMMARY_MD="$EVAL_DIR/passrate_table.md"

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
  echo "[SMOKE MODE] limit=5 tasks/dataset"
  SB_OUT="$EVAL_DIR/sb_passrate_smoke.jsonl"
  TB_OUT="$EVAL_DIR/tb_passrate_smoke.jsonl"
  SUMMARY_JSON="$EVAL_DIR/passrate_smoke_table.json"
  SUMMARY_MD="$EVAL_DIR/passrate_smoke_table.md"
fi

cd "$REPO_ROOT"

CORPUS="${SKILL_CORPUS_PATH:-$REPO_ROOT/data/processed/skill_corpus.jsonl}"
INDEX="${SKILL_INDEX_PATH:-$REPO_ROOT/data/embeddings/index/index.faiss}"
META="${SKILL_META_PATH:-$REPO_ROOT/data/embeddings/skill_metadata.jsonl}"
EMBEDDINGS="${SKILL_EMBEDDINGS_PATH:-$REPO_ROOT/data/embeddings/skill_embeddings.npy}"
for f in "$CORPUS" "$INDEX" "$META" "$EMBEDDINGS"; do
  [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
export SKILL_CORPUS_PATH="$CORPUS" SKILL_INDEX_PATH="$INDEX" SKILL_META_PATH="$META" SKILL_EMBEDDINGS_PATH="$EMBEDDINGS"

echo "======================================================================"
echo "  Sonnet 4.6 pass-rate (selection-proxy) eval — $(date '+%Y-%m-%d %H:%M')"
echo "  Model    : $MODEL"
echo "  n_noise  : $N_NOISE"
echo "  Seeds    : $SEEDS"
echo "  Datasets : SB($(wc -l < "$SB_TASKS")) + TB-validated($(wc -l < "$TB_TASKS"))"
echo "======================================================================"

echo
echo ">>> [1/3] SkillsBench"
"$PYTHON" -m "$MODULE.run_pass_rate" \
  --dataset sb --tasks "$SB_TASKS" --out "$SB_OUT" \
  --model "$MODEL" --n-noise $N_NOISE --seeds $SEEDS \
  --concurrency "$CONCURRENCY" $RESUME $LIMIT_ARG

echo
echo ">>> [2/3] TerminalBench (validated)"
"$PYTHON" -m "$MODULE.run_pass_rate" \
  --dataset tb --tasks "$TB_TASKS" --out "$TB_OUT" \
  --model "$MODEL" --n-noise $N_NOISE --seeds $SEEDS \
  --concurrency "$CONCURRENCY" $RESUME $LIMIT_ARG

echo
echo ">>> [3/3] Aggregate table"
"$PYTHON" -m "$MODULE.report_pass_rate" \
  --results "$SB_OUT" "$TB_OUT" \
  --json "$SUMMARY_JSON" --md "$SUMMARY_MD"

echo
echo "======================================================================"
echo "Summary JSON : $SUMMARY_JSON"
echo "Summary MD   : $SUMMARY_MD"
echo "======================================================================"
