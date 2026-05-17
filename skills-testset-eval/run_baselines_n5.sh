#!/usr/bin/env bash
# 20-task baselines with N=5 seeds: noskill + gtonly
set -euo pipefail
cd "$(dirname "$0")"

LIMIT="${1:-20}"
OUT="results/sb_baselines_n5.jsonl"
CONCURRENCY="${2:-4}"
TRIALS_PER="${3:-3}"
touch "$OUT"

# 2 modes x 5 seeds x LIMIT tasks
TOTAL=$((LIMIT * 2 * 5))

echo "=== N=5 Baselines: limit=$LIMIT, conc=$CONCURRENCY, total=$TOTAL ==="

run_one() {
    local MODE=$1
    local SKIP=$2
    local SEEDS=$3
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
        --noise-modes $MODE \
        --seeds $SEEDS \
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
        seen.add((r.get('task_id'), r.get('pool_size'), r.get('noise_mode'), r.get('seed')))
    except: pass
print(len(seen))
" 2>/dev/null || echo 0
}

count_mode() {
    local MODE=$1
    python3 -c "
import json
seen = set()
for line in open('$OUT'):
    try:
        r = json.loads(line)
        if r.get('noise_mode') == '$MODE':
            seen.add((r.get('task_id'), r.get('seed')))
    except: pass
print(len(seen))
" 2>/dev/null || echo 0
}

# Each mode: 20 tasks x 5 seeds = 100 configs
MODE_TOTAL=$((LIMIT * 5))

for MODE in noskill gtonly; do
    echo "=== Baseline: $MODE (N=5 seeds) ==="
    STALL=0
    while true; do
        COMPLETED=$(count_mode "$MODE")
        REMAINING=$((MODE_TOTAL - COMPLETED))
        if [ "$REMAINING" -le 0 ]; then
            echo "[$(date '+%H:%M:%S')] $MODE done! ($COMPLETED/$MODE_TOTAL)"
            break
        fi
        echo "[$(date '+%H:%M:%S')] $MODE: $COMPLETED/$MODE_TOTAL done, $REMAINING remaining. Launching $CONCURRENCY..."

        pids=()
        for i in $(seq 0 $((CONCURRENCY - 1))); do
            SKIP=$((i * TRIALS_PER))
            if [ "$SKIP" -ge "$REMAINING" ]; then break; fi
            run_one "$MODE" "$SKIP" "0 1 2 3 4" &
            pids+=($!)
        done

        for pid in "${pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done

        NEW_COMPLETED=$(count_mode "$MODE")
        if [ "$NEW_COMPLETED" -le "$COMPLETED" ]; then
            STALL=$((STALL + 1))
            echo "  [WARN] No progress (stall $STALL/10)"
            if [ "$STALL" -ge 10 ]; then
                echo "  [ERROR] 10 stalls, moving on."
                break
            fi
        else
            GAINED=$((NEW_COMPLETED - COMPLETED))
            echo "  +$GAINED unique configs"
            STALL=0
        fi
    done
done

echo "=== N=5 Baselines Complete ==="
TOTAL_UNIQUE=$(count_unique)
echo "Total unique configs: $TOTAL_UNIQUE/$TOTAL"
