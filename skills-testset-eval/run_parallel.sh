#!/usr/bin/env bash
# Parallel trial runner: launches CONCURRENCY containers with sharded trial offsets.
# Each container works on a different slice of remaining trials.
set -euo pipefail
cd "$(dirname "$0")"

LIMIT="${1:-20}"
NOISE="${2:-random hard easy}"
SIZES="${3:-5 10 20 50}"
OUT="${4:-results/sb_exec_20t.jsonl}"
CONCURRENCY="${5:-4}"
TRIALS_PER="${6:-3}"

TOTAL=$((LIMIT * $(echo $SIZES | wc -w) * $(echo $NOISE | wc -w)))
touch "$OUT"

echo "=== Parallel runner: limit=$LIMIT, conc=$CONCURRENCY, trials_per=$TRIALS_PER, total=$TOTAL ==="

run_one() {
    local SKIP=$1
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
        --out /workspace/$OUT \
        --work-root /tmp/exec_prefill \
        --model claude-sonnet-4-6 \
        --pool-sizes $SIZES \
        --noise-modes $NOISE \
        --seeds 0 \
        --concurrency 1 \
        --python python \
        --limit $LIMIT \
        --max-trials $TRIALS_PER \
        --skip-trials $SKIP \
        --resume 2>&1'" 2>/dev/null
}

count_unique() {
    python3 -c "
import json
seen = set()
for line in open('$OUT'):
    try:
        r = json.loads(line)
        seen.add((r.get('task_id'), r.get('pool_size'), r.get('noise_mode')))
    except: pass
print(len(seen))
" 2>/dev/null || echo 0
}

STALL=0
while true; do
    COMPLETED=$(count_unique)
    REMAINING=$((TOTAL - COMPLETED))
    if [ "$REMAINING" -le 0 ]; then
        echo "[$(date '+%H:%M:%S')] All $TOTAL unique configs done!"
        break
    fi
    LINES=$(wc -l < "$OUT" 2>/dev/null || echo 0)
    echo "[$(date '+%H:%M:%S')] $COMPLETED/$TOTAL unique configs done ($LINES lines), $REMAINING remaining. Launching $CONCURRENCY..."

    # Launch CONCURRENCY containers with different trial offsets
    pids=()
    for i in $(seq 0 $((CONCURRENCY - 1))); do
        SKIP=$((i * TRIALS_PER))
        if [ "$SKIP" -ge "$REMAINING" ]; then break; fi
        run_one "$SKIP" &
        pids+=($!)
    done

    # Wait for all to finish
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    NEW_COMPLETED=$(count_unique)
    if [ "$NEW_COMPLETED" -le "$COMPLETED" ]; then
        STALL=$((STALL + 1))
        echo "  [WARN] No progress (stall $STALL/10)"
        if [ "$STALL" -ge 10 ]; then
            echo "  [ERROR] 10 consecutive stalls, aborting."
            break
        fi
    else
        GAINED=$((NEW_COMPLETED - COMPLETED))
        echo "  +$GAINED unique configs completed"
        STALL=0
    fi
done

echo "=== Aggregating ==="
sg docker -c "docker run --rm \
  -v $PWD/results:/workspace/results \
  -e PYTHONPATH=/workspace/procmem2skills/testsets \
  -w /workspace/procmem2skills \
  skills-testset-eval-eval-pipeline \
  -c 'python -m exec_eval_prefill.aggregate \
    --results /workspace/$OUT \
    --json /workspace/results/exec_20t_table.json \
    --md /workspace/results/exec_20t_table.md && \
  cat /workspace/results/exec_20t_table.md'" || true
echo "=== Done ==="
