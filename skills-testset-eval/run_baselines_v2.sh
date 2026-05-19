#!/usr/bin/env bash
# V2 baselines: noskill + gtonly, one trial per container.
set -euo pipefail
cd "$(dirname "$0")"

LIMIT="${1:-5}"
OUT="results/sb_baselines.jsonl"
touch "$OUT"

run_one() {
    local MODE=$1
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
        --out /workspace/results/sb_baselines.jsonl \
        --work-root /tmp/exec_prefill \
        --model claude-sonnet-4-6 \
        --pool-sizes 0 \
        --noise-modes $MODE \
        --seeds 0 \
        --concurrency 1 \
        --python python \
        --limit $LIMIT \
        --max-trials 1 \
        --resume 2>&1'" || echo "  [WARN] container non-zero"
}

for MODE in noskill gtonly; do
    echo "=== Baseline: $MODE ==="
    for i in $(seq 1 "$LIMIT"); do
        COMPLETED=$(grep -c "\"noise_mode\": \"$MODE\"" "$OUT" 2>/dev/null || true)
        COMPLETED=${COMPLETED:-0}
        if [ "$COMPLETED" -ge "$LIMIT" ]; then
            echo "  $MODE done ($COMPLETED/$LIMIT)"
            break
        fi
        echo "[$(date '+%H:%M:%S')] $MODE trial $((COMPLETED+1))/$LIMIT..."
        run_one "$MODE"
    done
done
echo "=== V2 Baselines Complete ==="
