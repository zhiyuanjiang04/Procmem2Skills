#!/usr/bin/env bash
# Run baseline experiments: noskill (no SKILL.md) and gtonly (GT skills only).
# Sequential fresh-container approach to avoid asyncio hang.
set -euo pipefail
cd "$(dirname "$0")"

LIMIT="${1:-5}"
OUT="results/sb_baselines.jsonl"
touch "$OUT"

for MODE in noskill gtonly; do
    COMPLETED=$(grep -c "\"noise_mode\": \"$MODE\"" "$OUT" 2>/dev/null || echo 0)
    TOTAL=$LIMIT
    echo "=== Baseline: $MODE (completed=$COMPLETED/$TOTAL) ==="

    while true; do
        COMPLETED=$(grep -c "\"noise_mode\": \"$MODE\"" "$OUT" 2>/dev/null || echo 0)
        if [ "$COMPLETED" -ge "$TOTAL" ]; then
            echo "  $MODE done ($COMPLETED/$TOTAL)"
            break
        fi
        echo "[$(date '+%H:%M:%S')] Running $MODE trial ($COMPLETED/$TOTAL done)..."

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
            --resume 2>&1'" || echo "  [WARN] container exited non-zero, continuing..."

        NEW_COMPLETED=$(grep -c "\"noise_mode\": \"$MODE\"" "$OUT" 2>/dev/null || echo 0)
        if [ "$NEW_COMPLETED" -eq "$COMPLETED" ]; then
            echo "  [WARN] No progress, retrying..."
        fi
    done
done

echo "=== Aggregating baselines ==="
sg docker -c "docker run --rm \
  -v $PWD/results:/workspace/results \
  -e PYTHONPATH=/workspace/procmem2skills/testsets \
  -w /workspace/procmem2skills \
  skills-testset-eval-eval-pipeline \
  -c 'python -m exec_eval_prefill.aggregate \
    --results /workspace/results/sb_baselines.jsonl \
    --json /workspace/results/baselines_table.json \
    --md /workspace/results/baselines_table.md && \
  cat /workspace/results/baselines_table.md'"

echo "=== Done ==="
