#!/usr/bin/env bash
# Execution-pass-rate orchestrator. REQUIRES DOCKER (not on Anvil).
#
# Spec (user 2026-05-13):
#   GT skills always in pool. n_noise ∈ {0,1,5,10,20,50}. random distractors.
#
# Usage:
#   bash testsets/exec_eval/run_exec_eval.sh sb         # SB only
#   bash testsets/exec_eval/run_exec_eval.sh tb         # TB only (needs SKILL.md scraping)
#   bash testsets/exec_eval/run_exec_eval.sh sb --dry-run  # Anvil-safe; just builds skill dirs
#   bash testsets/exec_eval/run_exec_eval.sh sb --run-cmd-only  # print harbor cmds
#
# Real execution prerequisites (verified at startup):
#   - docker, docker-compose installed
#   - pip install -e /path/to/skillsbench_repo (provides harbor)
#   - LITELLM_API_KEY set (Sonnet 4.6 via LiteLLM)

set -euo pipefail
DATASET="${1:-sb}"; shift || true
EXTRA_ARGS="$@"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TESTSETS="$REPO_ROOT/testsets"
PYTHON="${PYTHON:-/anvil/projects/x-cis260386/william/llm-colearning/envs/marl/bin/python}"

case "$DATASET" in
  sb) TASKS="$TESTSETS/data/skillsbench_tasks.jsonl" ;;
  tb) TASKS="$TESTSETS/data/terminal_bench_validated.jsonl" ;;
  *) echo "dataset must be sb or tb"; exit 1 ;;
esac

OUT_DIR="$TESTSETS/eval_exec_sonnet46"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/exec_${DATASET}.jsonl"
WORK="/tmp/exec_eval/$DATASET"

# Preflight: docker check (warn-not-error in --dry-run)
NEED_DOCKER=1
case " $EXTRA_ARGS " in *" --dry-run "*|*" --run-cmd-only "*) NEED_DOCKER=0 ;; esac
if [ "$NEED_DOCKER" -eq 1 ]; then
  command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found. Use --dry-run on Anvil."; exit 2; }
  command -v harbor >/dev/null 2>&1 || { echo "ERROR: harbor CLI not installed."; exit 2; }
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export SKILL_CORPUS_PATH="${SKILL_CORPUS_PATH:-$REPO_ROOT/data/processed/skill_corpus.jsonl}"

echo "==> exec-eval dataset=$DATASET tasks=$(wc -l < "$TASKS") out=$OUT"

"$PYTHON" -m testsets.exec_eval.orchestrate \
  --dataset "$DATASET" \
  --tasks "$TASKS" \
  --out "$OUT" \
  --work-root "$WORK" \
  --corpus "$SKILL_CORPUS_PATH" \
  --model claude-sonnet-4-6 \
  --n-noise 0 1 5 10 20 50 \
  --seeds 0 \
  --concurrency 4 \
  --resume \
  $EXTRA_ARGS
