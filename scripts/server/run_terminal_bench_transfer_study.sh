#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/common.sh"
prepare_server_env "$ROOT"

# Normalize OpenRouter credentials to the OpenAI-compatible vars consumed by Harbor codex agent.
if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${OPENAI_API_KEY:-$OPENROUTER_API_KEY}"
  export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
  export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$OPENROUTER_BASE_URL}"
fi

PYTHON_BIN="$(resolve_benchmark_python "$ROOT")"

exec "$PYTHON_BIN" "$ROOT/scripts/server/run_terminal_bench_transfer_study.py" "$@"
