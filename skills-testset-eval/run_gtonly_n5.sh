#!/usr/bin/env bash
# Rerun GT-only N=5 baselines (fill rate-limited gaps)
set -euo pipefail
cd "$(dirname "$0")"

LIMIT="${1:-20}"
OUT="results/sb_baselines_n5.jsonl"
CONCURRENCY="${2:-4}"
TRIALS_PER="${3:-3}"
touch "$OUT"

echo "=== GT-only N=5 Rerun: limit=$LIMIT, conc=$CONCURRENCY ==="

run_one() {
    local SKIP=$1
    local SEEDS=$2
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
        --pool-sizes 0 \
        --noise-modes gtonly \
        --seeds $SEEDS \
        --concurrency 1 \
        --python python \
        --limit $LIMIT \
        --max-trials $TRIALS_PER \
        --skip-trials $SKIP \
        --resume 2>&1'" 2>/dev/null
}

count_clean_gtonly() {
    python3 -c "
import json
seen = set()
for line in open('$OUT'):
    try:
        r = json.loads(line)
        if r.get('noise_mode') != 'gtonly': continue
        stdout = (r.get('agent_stdout_tail') or '')
        is_rl = 'limit' in stdout.lower() or 'resets' in stdout.lower()
        if not is_rl:
            seen.add((r.get('task_id'), r.get('seed')))
    except: pass
print(len(seen))
" 2>/dev/null || echo 0
}

# Target: 20 tasks x 5 seeds = 100 clean configs
MODE_TOTAL=$((LIMIT * 5))
STALL=0

while true; do
    CLEAN=$(count_clean_gtonly)
    echo "[$(date '+%H:%M:%S')] gtonly clean: $CLEAN/$MODE_TOTAL"
    if [ "$CLEAN" -ge "$MODE_TOTAL" ]; then
        echo "All $MODE_TOTAL clean configs done!"
        break
    fi

    REMAINING=$((MODE_TOTAL - CLEAN))
    echo "  $REMAINING remaining. Launching $CONCURRENCY containers..."

    pids=()
    for i in $(seq 0 $((CONCURRENCY - 1))); do
        SKIP=$((i * TRIALS_PER))
        if [ "$SKIP" -ge "$REMAINING" ]; then break; fi
        run_one "$SKIP" "0 1 2 3 4" &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    NEW_CLEAN=$(count_clean_gtonly)
    if [ "$NEW_CLEAN" -le "$CLEAN" ]; then
        STALL=$((STALL + 1))
        echo "  [WARN] No progress (stall $STALL/15)"
        if [ "$STALL" -ge 15 ]; then
            echo "  [ERROR] 15 stalls, aborting."
            break
        fi
    else
        GAINED=$((NEW_CLEAN - CLEAN))
        echo "  +$GAINED clean configs"
        STALL=0
    fi
done

echo "=== GT-only N=5 Rerun Complete ==="
echo "Clean configs: $(count_clean_gtonly)/$MODE_TOTAL"
