#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/common.sh"
prepare_server_env "$ROOT"

PYTHON_BIN="$(resolve_benchmark_python "$ROOT")"

exec "$PYTHON_BIN" "$ROOT/scripts/server/run_skillsbench_harbor_experiment.py" "$@"
