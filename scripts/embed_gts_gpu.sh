#!/bin/bash
#SBATCH --job-name=embed_gts
#SBATCH --account=cis260386-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/embed_gts_%j.out
#SBATCH --error=logs/embed_gts_%j.err
set -eo pipefail

cd /anvil/projects/x-cis260386/william/procmem2skills/procmem2skills
mkdir -p logs

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Start: $(date)"

PYTHONPATH="skills retrieval/src" \
  /anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/python3.10 \
  -m skills_retrieval.scripts.embed_tasks_and_gts \
  --out "skills retrieval/pools/tasks_gt_embeddings_full_gpu.npz"

echo "End: $(date)"
