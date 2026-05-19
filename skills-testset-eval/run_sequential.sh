#!/usr/bin/env bash
# Run trials one-at-a-time in fresh containers to avoid asyncio hang.
# Each container runs exactly 1 trial (--max-trials 1) then exits.
set -euo pipefail
cd "$(dirname "$0")"

LIMIT="${1:-5}"
NOISE_MODES="${2:-random hard easy}"
POOL_SIZES="${3:-5 10 20 50}"
OUT_FILE="${4:-results/sb_exec.jsonl}"
TOTAL_TRIALS=$((LIMIT * $(echo $POOL_SIZES | wc -w) * $(echo $NOISE_MODES | wc -w)))

echo "=== Sequential runner: limit=$LIMIT, noise=$NOISE_MODES, sizes=$POOL_SIZES ==="
echo "    output=$OUT_FILE, max_trials=$TOTAL_TRIALS"

STALL_COUNT=0
while true; do
    COMPLETED=$(wc -l < "$OUT_FILE" 2>/dev/null || echo 0)
    REMAINING=$((TOTAL_TRIALS - COMPLETED))
    if [ "$REMAINING" -le 0 ]; then
        echo "All $TOTAL_TRIALS trials done!"
        break
    fi
    echo "[$(date '+%H:%M:%S')] Trial $((COMPLETED+1))/$TOTAL_TRIALS ($REMAINING remaining)..."

    sg docker -c "docker run --rm \
      -v $HOME/.claude:/home/evaluser/.claude \
      -v $HOME/.claude.json:/home/evaluser/.claude.json \
      -v $PWD/results:/workspace/results \
      -e PYTHONPATH=/workspace/procmem2skills/testsets \
      -e SKILL_CORPUS_PATH=/workspace/procmem2skills/data/processed/skill_corpus.jsonl \
      -e SKILL_INDEX_PATH=/workspace/procmem2skills/data/embeddings/index/index.faiss \
      -e SKILL_META_PATH=/workspace/procmem2skills/data/embeddings/skill_metadata.jsonl \
      -e SKILL_EMBEDDINGS_PATH=/workspace/procmem2skills/data/embeddings/skill_embeddings.npy \
      -w /workspace/procmem2skills \
      skills-testset-eval-eval-pipeline \
      -c 'python -u -m exec_eval_prefill.run_trial \
        --dataset sb \
        --tasks /workspace/procmem2skills/testsets/data/skillsbench_tasks.jsonl \
        --out /workspace/$OUT_FILE \
        --work-root /tmp/exec_prefill \
        --model claude-sonnet-4-6 \
        --pool-sizes $POOL_SIZES \
        --noise-modes $NOISE_MODES \
        --seeds 0 \
        --concurrency 1 \
        --python python \
        --limit $LIMIT \
        --max-trials 1 \
        --resume 2>&1'" || echo "  [WARN] container exited non-zero"

    NEW_COMPLETED=$(wc -l < "$OUT_FILE" 2>/dev/null || echo 0)
    if [ "$NEW_COMPLETED" -le "$COMPLETED" ]; then
        STALL_COUNT=$((STALL_COUNT + 1))
        echo "  [WARN] No progress (stall $STALL_COUNT/3)"
        if [ "$STALL_COUNT" -ge 3 ]; then
            echo "  [ERROR] 3 consecutive stalls, aborting."
            break
        fi
    else
        STALL_COUNT=0
    fi
done

echo "=== Aggregating ==="
sg docker -c "docker run --rm \
  -v $PWD/results:/workspace/results \
  -e PYTHONPATH=/workspace/procmem2skills/testsets \
  -w /workspace/procmem2skills \
  skills-testset-eval-eval-pipeline \
  -c 'python -m exec_eval_prefill.aggregate \
    --results /workspace/$OUT_FILE \
    --json /workspace/results/exec_table.json \
    --md /workspace/results/exec_table.md && \
  cat /workspace/results/exec_table.md'" || true
echo "=== Done ==="
