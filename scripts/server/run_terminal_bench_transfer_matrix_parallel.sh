#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/common.sh"
prepare_server_env "$ROOT"

EXPERIMENT_PREFIX=""
STRONG_MODEL=""
PARALLEL_JOBS=0
DATASET_SPEC="terminal-bench@2.0"
WEAK_MODELS=()
FORWARDED=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-prefix)
      EXPERIMENT_PREFIX="${2:-}"
      shift 2
      ;;
    --strong-model)
      STRONG_MODEL="${2:-}"
      shift 2
      ;;
    --weak-model)
      WEAK_MODELS+=("${2:-}")
      shift 2
      ;;
    --parallel-jobs)
      PARALLEL_JOBS="${2:-0}"
      shift 2
      ;;
    --dataset)
      DATASET_SPEC="${2:-}"
      FORWARDED+=("$1" "$DATASET_SPEC")
      shift 2
      ;;
    --)
      shift
      FORWARDED+=("$@")
      break
      ;;
    *)
      FORWARDED+=("$1")
      shift
      ;;
  esac
done

# If --dataset is provided after `--`, use that value for local run/log directory layout.
for ((idx = 0; idx < ${#FORWARDED[@]}; idx++)); do
  if [[ "${FORWARDED[$idx]}" == "--dataset" && $((idx + 1)) -lt ${#FORWARDED[@]} ]]; then
    DATASET_SPEC="${FORWARDED[$((idx + 1))]}"
  fi
done

if [[ -z "$EXPERIMENT_PREFIX" ]]; then
  echo "error: --experiment-prefix is required" >&2
  exit 2
fi
if [[ -z "$STRONG_MODEL" ]]; then
  echo "error: --strong-model is required" >&2
  exit 2
fi
if [[ ${#WEAK_MODELS[@]} -eq 0 ]]; then
  echo "error: at least one --weak-model is required" >&2
  exit 2
fi

if [[ "$PARALLEL_JOBS" -le 0 ]]; then
  PARALLEL_JOBS=${#WEAK_MODELS[@]}
fi
if [[ "$PARALLEL_JOBS" -le 0 ]]; then
  PARALLEL_JOBS=1
fi

DATASET_DIR="$(truncate_slug "$(normalize_slug "$DATASET_SPEC")" 32)"
mkdir -p "experiments/${DATASET_DIR}"

failed=0
pids=()
CANONICAL_PREFIX="$(truncate_slug "$(normalize_slug "$EXPERIMENT_PREFIX")" 32)"

launch_one() {
  local weak_model="$1"
  local weak_slug
  weak_slug="$(truncate_slug "$(normalize_slug "$weak_model")" 20)"
  local exp_id_raw="${CANONICAL_PREFIX}-to-${weak_slug}"
  local exp_id
  exp_id="$(truncate_slug "$(normalize_slug "$exp_id_raw")" 56)"
  local run_dir="$ROOT/experiments/${DATASET_DIR}/${exp_id}"
  local log_path="${run_dir}/run.log"
  mkdir -p "$run_dir"

  (
    bash "$ROOT/scripts/server/run_terminal_bench_transfer_study.sh" \
      --experiment-id "$exp_id" \
      --strong-model "$STRONG_MODEL" \
      --weak-model "$weak_model" \
      "${FORWARDED[@]}"
  ) 2>&1 | tee "$log_path"
}

for weak_model in "${WEAK_MODELS[@]}"; do
  echo "[launch] strong=${STRONG_MODEL} weak=${weak_model} prefix=${CANONICAL_PREFIX}"
  launch_one "$weak_model" &
  pids+=("$!")
  if [[ "${#pids[@]}" -ge "$PARALLEL_JOBS" ]]; then
    if ! wait "${pids[0]}"; then
      failed=1
    fi
    pids=("${pids[@]:1}")
  fi
done

if [[ "${#pids[@]}" -gt 0 ]]; then
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
fi

exit "$failed"
