#!/usr/bin/env bash
# Run the skills eval pipeline in Docker using Max plan credentials.
#
# Usage:
#   ./run.sh              # smoke test (5 tasks × 12 conditions)
#   ./run.sh --full       # full run (84 tasks × 12 conditions)
#   ./run.sh --shell      # interactive shell inside container
#   CONCURRENCY=4 ./run.sh  # override concurrency

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:---smoke}"

echo "Building Docker image..."
docker compose build

case "$MODE" in
  --shell)
    echo "Starting interactive shell..."
    docker compose run --rm eval-pipeline
    ;;
  --full)
    echo "Starting FULL eval (84 tasks × 12 conditions)..."
    docker compose run --rm eval-pipeline -c "\
      python -m exec_eval_prefill.run_trial \
        --dataset sb \
        --tasks /workspace/procmem2skills/testsets/data/skillsbench_tasks.jsonl \
        --out /workspace/results/sb_exec_full.jsonl \
        --work-root /tmp/exec_prefill \
        --model claude-sonnet-4-6 \
        --pool-sizes 5 10 20 50 \
        --noise-modes random hard easy \
        --seeds 0 \
        --concurrency \${CONCURRENCY:-2} \
        --python python \
        --resume \
      && python -m exec_eval_prefill.aggregate \
        --results /workspace/results/sb_exec_full.jsonl \
        --json /workspace/results/exec_full_table.json \
        --md /workspace/results/exec_full_table.md \
      && echo '=== Results ===' && cat /workspace/results/exec_full_table.md"
    ;;
  *)
    echo "Starting SMOKE eval (5 tasks × 12 conditions)..."
    docker compose up --build
    ;;
esac
