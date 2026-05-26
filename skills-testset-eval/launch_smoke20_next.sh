#!/usr/bin/env bash
# Launch next 20-task batch at concurrency=12.
# Triggered after smoke10_main container exits.
set -euo pipefail
cd "$(dirname "$0")"

CONTAINER_NAME="smoke20_next"
OUT_FILE="results/smoke20/smoke20_n5.jsonl"
mkdir -p results/smoke20
touch "$OUT_FILE"

# Refuse if container is already running
if sg docker -c "docker ps --format '{{.Names}}'" | grep -q "^${CONTAINER_NAME}$"; then
    echo "ERROR: ${CONTAINER_NAME} already running"
    exit 1
fi

sg docker -c "docker run -d --name ${CONTAINER_NAME} --rm \
  --env-file /tmp/anthropic_key.env \
  -v $PWD/results:/workspace/results \
  -v $PWD/testsets/data/smoke20_next.jsonl:/workspace/procmem2skills/testsets/data/smoke20_next.jsonl \
  -v $PWD/skillsbench_repo:/workspace/skillsbench_repo \
  -v $PWD/logs:/workspace/logs \
  -w /workspace/procmem2skills \
  -e PYTHONPATH=/workspace/procmem2skills/testsets \
  -e SKILL_CORPUS_PATH=/workspace/procmem2skills/data/processed/skill_corpus.jsonl \
  -e SKILL_INDEX_PATH=/workspace/procmem2skills/data/embeddings/index/index.faiss \
  -e SKILL_META_PATH=/workspace/procmem2skills/data/embeddings/skill_metadata.jsonl \
  -e SKILL_EMBEDDINGS_PATH=/workspace/procmem2skills/data/embeddings/skill_embeddings.npy \
  skills-eval-smoke10:latest \
  -c 'python -u -m exec_eval_prefill.run_trial \
    --dataset sb \
    --tasks /workspace/procmem2skills/testsets/data/smoke20_next.jsonl \
    --out /workspace/${OUT_FILE} \
    --work-root /tmp/exec_prefill \
    --model claude-sonnet-4-6 \
    --pool-sizes 5 10 20 50 100 \
    --noise-modes random hard easy \
    --seeds 0 1 2 3 4 \
    --concurrency 12 \
    --python python \
    --resume 2>&1 | tee -a /workspace/logs/smoke20.log'"

echo "Launched ${CONTAINER_NAME}"
sleep 5
sg docker -c "docker ps --filter name=${CONTAINER_NAME} --format '{{.Status}}'"
