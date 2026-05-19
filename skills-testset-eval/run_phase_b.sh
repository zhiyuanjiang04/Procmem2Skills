#!/usr/bin/env bash
# Phase B: factorial controls — emptyframe, noiseonly, N=5 seeds each
# P3 (emptyframe): tests prompt framing effect — HIGHEST PRIORITY
# P1 (noiseonly): tests noise priming effect
set -euo pipefail
cd "$(dirname "$0")"

LIMIT="${1:-20}"
OUT="results/sb_phase_b.jsonl"
CONCURRENCY="${2:-4}"
TRIALS_PER="${3:-3}"
touch "$OUT"

echo "=== Phase B: emptyframe + noiseonly, limit=$LIMIT, conc=$CONCURRENCY ==="

run_one() {
    local MODE=$1
    local SKIP=$2
    local SEEDS=$3
    local SIZES=$4
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
        --noise-modes $MODE \
        --seeds $SEEDS \
        --concurrency 1 \
        --python python \
        --limit $LIMIT \
        --max-trials $TRIALS_PER \
        --skip-trials $SKIP \
        --resume 2>&1'" 2>/dev/null
}

count_clean() {
    local MODE=$1
    python3 -c "
import json, re
rl_pat = re.compile(r\"hit your limit|you've hit|resets \d+:\d+\")
seen = set()
for line in open('$OUT'):
    try:
        r = json.loads(line)
        if r.get('noise_mode') != '$MODE': continue
        stdout = (r.get('agent_stdout_tail') or '').lower()
        if rl_pat.search(stdout): continue
        if r.get('skipped'): continue
        seen.add((r.get('task_id'), r.get('pool_size'), r.get('seed')))
    except: pass
print(len(seen))
" 2>/dev/null || echo 0
}

# P3: emptyframe — pool_size=0, 5 seeds = 20 tasks x 5 seeds = 100 configs
echo "=== P3: emptyframe (N=5 seeds) ==="
MODE_TOTAL=$((LIMIT * 5))
STALL=0
while true; do
    COMPLETED=$(count_clean "emptyframe")
    REMAINING=$((MODE_TOTAL - COMPLETED))
    if [ "$REMAINING" -le 0 ]; then
        echo "[$(date '+%H:%M:%S')] emptyframe done! ($COMPLETED/$MODE_TOTAL)"
        break
    fi
    echo "[$(date '+%H:%M:%S')] emptyframe: $COMPLETED/$MODE_TOTAL done, $REMAINING remaining."

    pids=()
    for i in $(seq 0 $((CONCURRENCY - 1))); do
        SKIP=$((i * TRIALS_PER))
        if [ "$SKIP" -ge "$REMAINING" ]; then break; fi
        run_one "emptyframe" "$SKIP" "0 1 2 3 4" "0" &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    NEW_COMPLETED=$(count_clean "emptyframe")
    if [ "$NEW_COMPLETED" -le "$COMPLETED" ]; then
        STALL=$((STALL + 1))
        echo "  [WARN] No progress (stall $STALL/15)"
        if [ "$STALL" -ge 15 ]; then
            echo "  [ERROR] 15 stalls, moving on."
            break
        fi
    else
        GAINED=$((NEW_COMPLETED - COMPLETED))
        echo "  +$GAINED clean configs"
        STALL=0
    fi
done

# P1: noiseonly — pool_size=10, 5 seeds = 20 tasks x 5 seeds = 100 configs
echo "=== P1: noiseonly (N=5 seeds, pool=10) ==="
MODE_TOTAL=$((LIMIT * 5))
STALL=0
while true; do
    COMPLETED=$(count_clean "noiseonly")
    REMAINING=$((MODE_TOTAL - COMPLETED))
    if [ "$REMAINING" -le 0 ]; then
        echo "[$(date '+%H:%M:%S')] noiseonly done! ($COMPLETED/$MODE_TOTAL)"
        break
    fi
    echo "[$(date '+%H:%M:%S')] noiseonly: $COMPLETED/$MODE_TOTAL done, $REMAINING remaining."

    pids=()
    for i in $(seq 0 $((CONCURRENCY - 1))); do
        SKIP=$((i * TRIALS_PER))
        if [ "$SKIP" -ge "$REMAINING" ]; then break; fi
        run_one "noiseonly" "$SKIP" "0 1 2 3 4" "10" &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    NEW_COMPLETED=$(count_clean "noiseonly")
    if [ "$NEW_COMPLETED" -le "$COMPLETED" ]; then
        STALL=$((STALL + 1))
        echo "  [WARN] No progress (stall $STALL/15)"
        if [ "$STALL" -ge 15 ]; then
            echo "  [ERROR] 15 stalls, moving on."
            break
        fi
    else
        GAINED=$((NEW_COMPLETED - COMPLETED))
        echo "  +$GAINED clean configs"
        STALL=0
    fi
done

echo "=== Phase B Complete ==="
echo "emptyframe clean: $(count_clean 'emptyframe')/$((LIMIT * 5))"
echo "noiseonly clean: $(count_clean 'noiseonly')/$((LIMIT * 5))"
