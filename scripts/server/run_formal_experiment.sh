#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/common.sh"
prepare_server_env "$ROOT"

BENCHMARK=""
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--benchmark" && $((i + 1)) -lt ${#ARGS[@]} ]]; then
    BENCHMARK="${ARGS[$((i + 1))]}"
    break
  fi
done

if [[ -n "${PROCMEM_EXPERIMENT_PYTHON:-}" ]]; then
  PYTHON_BIN="$PROCMEM_EXPERIMENT_PYTHON"
elif [[ "$BENCHMARK" == "webarena" && -x "$ROOT/.venv-py312/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv-py312/bin/python"
else
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi

exec "$PYTHON_BIN" "$ROOT/scripts/server/run_formal_experiment.py" "$@"
