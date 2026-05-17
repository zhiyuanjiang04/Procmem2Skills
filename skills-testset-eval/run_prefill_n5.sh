#!/usr/bin/env bash
# Phase P0: Prefill N=5 — add seeds 1-4 to existing seed=0 prefill data
# Resolves pseudoreplication: enables true variance estimation for PF vs GT comparison
# Target: 4 seeds x 3 noise modes x 4 pool sizes x 20 tasks = 960 configs
# Expected clean: ~672 (14 tasks), skip: ~288 (6 heavy Dockerfile tasks)
set -euo pipefail
cd "$(dirname "$0")"

LIMIT="${1:-20}"
OUT="results/sb_prefill_n5.jsonl"
CONCURRENCY="${2:-4}"
TRIALS_PER="${3:-3}"
touch "$OUT"

echo "=== Phase P0: Prefill N=5 (seeds 1-4), limit=$LIMIT, conc=$CONCURRENCY ==="

run_one() {
    local MODE=$1
    local SKIP=$2
    local SEEDS=$3
    local SIZES=$4
    sg docker -c "docker run --rm \
      -v $HOME/.claude:/home/evaluser/.claude \
      -v $HOME/.claude.json:/home/evaluser/.claude.json \
      -v $PWD/results:/workspace/results \
      -w /workspace/procmem2skills \
      -e PYTHONPATH=/workspace/procmem2skills/testsets \
      -e SKILL_CORPUS_PATH=/workspace/procmem2skills/data/processed/skill_corpus.jsonl \
      -e SKILL_INDEX_PATH=/workspace/procmem2skills/data/embeddings/index/index.faiss \
      -e SKILL_META_PATH=/workspace/procmem2skills/data/embeddings/skill_metadata.jsonl \
      -e SKILL_EMBEDDINGS_PATH=/workspace/procmem2skills/data/embeddings/skill_embeddings.npy \
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

count_total() {
    python3 -c "
import json, re
rl_pat = re.compile(r\"hit your limit|you've hit|resets \d+:\d+\")
seen = set()
for line in open('$OUT'):
    try:
        r = json.loads(line)
        stdout = (r.get('agent_stdout_tail') or '').lower()
        if rl_pat.search(stdout): continue
        if r.get('skipped'): continue
        seen.add((r.get('task_id'), r.get('pool_size'), r.get('noise_mode'), r.get('seed')))
    except: pass
print(len(seen))
" 2>/dev/null || echo 0
}

# Run each noise mode sequentially, all 4 seeds in each
# Target per mode: 4 seeds x 4 pools x 20 tasks = 320 configs (clean ~224)
for MODE in random hard easy; do
    echo "=== Mode: $MODE (seeds 1-4, pools 5/10/20/50) ==="
    MODE_TARGET=$((LIMIT * 4 * 4))  # 20 tasks x 4 seeds x 4 pools = 320
    STALL=0
    while true; do
        COMPLETED=$(count_clean "$MODE")
        # Achievable clean is ~224 (14 tasks x 4 seeds x 4 pools)
        ACHIEVABLE=$((14 * 4 * 4))
        if [ "$COMPLETED" -ge "$ACHIEVABLE" ]; then
            echo "[$(date '+%H:%M:%S')] $MODE done! ($COMPLETED/$ACHIEVABLE achievable)"
            break
        fi
        REMAINING=$((ACHIEVABLE - COMPLETED))
        echo "[$(date '+%H:%M:%S')] $MODE: $COMPLETED/$ACHIEVABLE done, $REMAINING remaining."

        pids=()
        for i in $(seq 0 $((CONCURRENCY - 1))); do
            SKIP=$((i * TRIALS_PER))
            if [ "$SKIP" -ge "$REMAINING" ]; then break; fi
            run_one "$MODE" "$SKIP" "1 2 3 4" "5 10 20 50" &
            pids+=($!)
        done

        for pid in "${pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done

        NEW_COMPLETED=$(count_clean "$MODE")
        if [ "$NEW_COMPLETED" -le "$COMPLETED" ]; then
            STALL=$((STALL + 1))
            echo "  [WARN] No progress (stall $STALL/15)"
            if [ "$STALL" -ge 15 ]; then
                echo "  [ERROR] 15 stalls on $MODE, moving on."
                break
            fi
        else
            GAINED=$((NEW_COMPLETED - COMPLETED))
            echo "  +$GAINED clean configs"
            STALL=0
        fi
    done
done

echo "=== Phase P0 Complete ==="
TOTAL=$(count_total)
echo "Total clean configs: $TOTAL"
echo "random clean: $(count_clean 'random')"
echo "hard clean: $(count_clean 'hard')"
echo "easy clean: $(count_clean 'easy')"
