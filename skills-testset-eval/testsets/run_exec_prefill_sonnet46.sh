#!/usr/bin/env bash
# Sonnet 4.6 prefill-context execution-pass-rate eval (SB on Anvil host).
#
# Spec (user 2026-05-13, 3rd revision):
#   Step 1: all GT skills in pool.
#   Step 2: pool sizes {5,10,20,50}, noise types {random, hard (near), easy (far)}.
#   Method: ALL candidate SKILL.md content prefilled into system prompt.
#           Sonnet 4.6 runs the task with Bash/Read/Write/Edit/Glob/Grep tools
#           in an isolated /tmp/exec_prefill/<trial>/work/ dir.
#           Tests run via marl conda env pytest after agent exits.
#
# Coverage:
#   SB:    84/89 lightweight tasks (5 skipped: heavy Docker setup — nodejs,
#          playwright, etc.). 84 × 4 × 3 = 1008 SB trials.
#   TB:    NOT RUN on host. Container required. Pipeline scaffold in
#          testsets/exec_eval/ ready for Docker host.
#
# Isolation: each trial has its own /tmp/exec_prefill/<trial>/ tree.
# Resume: --resume reads JSONL and skips completed (task,size,noise,seed).
#
# Usage:
#   cd /anvil/projects/x-cis260386/william/procmem2skills/procmem2skills
#   bash testsets/run_exec_prefill_sonnet46.sh [--smoke] [--resume]

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTSETS="$REPO_ROOT/testsets"
PYTHON="${PYTHON:-/anvil/projects/x-cis260386/william/llm-colearning/envs/marl/bin/python}"

MODEL="claude-sonnet-4-6"
POOL_SIZES="5 10 20 50"
NOISE_MODES="random hard easy"
SEEDS="0"
CONCURRENCY="${CONCURRENCY:-4}"

SB_TASKS="$TESTSETS/data/skillsbench_tasks.jsonl"
OUT_DIR="$TESTSETS/eval_exec_prefill_sonnet46"
SB_OUT="$OUT_DIR/sb_exec.jsonl"
SUMMARY_JSON="$OUT_DIR/exec_table.json"
SUMMARY_MD="$OUT_DIR/exec_table.md"
WORK_ROOT="${WORK_ROOT:-/tmp/exec_prefill}"

SMOKE=0; RESUME=""; LIMIT_ARG=""
for arg in "$@"; do
  case "$arg" in
    --smoke)  SMOKE=1; LIMIT_ARG="--limit 5" ;;
    --resume) RESUME="--resume" ;;
  esac
done
mkdir -p "$OUT_DIR" "$WORK_ROOT"
[ "$SMOKE" -eq 1 ] && { SB_OUT="$OUT_DIR/sb_exec_smoke.jsonl"; SUMMARY_JSON="$OUT_DIR/exec_smoke_table.json"; SUMMARY_MD="$OUT_DIR/exec_smoke_table.md"; }

cd "$REPO_ROOT"
export PYTHONPATH="$TESTSETS:${PYTHONPATH:-}"
export SKILL_CORPUS_PATH="${SKILL_CORPUS_PATH:-$REPO_ROOT/data/processed/skill_corpus.jsonl}"
export SKILL_INDEX_PATH="${SKILL_INDEX_PATH:-$REPO_ROOT/data/embeddings/index/index.faiss}"
export SKILL_META_PATH="${SKILL_META_PATH:-$REPO_ROOT/data/embeddings/skill_metadata.jsonl}"
export SKILL_EMBEDDINGS_PATH="${SKILL_EMBEDDINGS_PATH:-$REPO_ROOT/data/embeddings/skill_embeddings.npy}"

echo "======================================================================"
echo "  Sonnet 4.6 exec-prefill pass-rate eval — $(date '+%Y-%m-%d %H:%M')"
echo "  Pool sizes : $POOL_SIZES"
echo "  Noise types: $NOISE_MODES"
echo "  Concurrency: $CONCURRENCY"
echo "  Work root  : $WORK_ROOT"
echo "  Output     : $SB_OUT"
echo "======================================================================"
echo

"$PYTHON" -m exec_eval_prefill.run_trial \
  --dataset sb \
  --tasks "$SB_TASKS" \
  --out "$SB_OUT" \
  --work-root "$WORK_ROOT" \
  --model "$MODEL" \
  --pool-sizes $POOL_SIZES \
  --noise-modes $NOISE_MODES \
  --seeds $SEEDS \
  --concurrency "$CONCURRENCY" \
  --python "$PYTHON" \
  $RESUME $LIMIT_ARG

echo
echo ">>> aggregate"
"$PYTHON" -m exec_eval_prefill.aggregate \
  --results "$SB_OUT" --json "$SUMMARY_JSON" --md "$SUMMARY_MD"

echo
echo "Output: $SB_OUT"
echo "Table : $SUMMARY_MD"
