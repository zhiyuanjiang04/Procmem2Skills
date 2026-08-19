#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DEFAULTS_FILE="${DEFAULTS_FILE:-${PROJECT_ROOT}/configs/defaults.env}"
PY_SCRIPT="${PROJECT_ROOT}/scripts/run_context_comparison.py"

load_defaults_if_unset() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      local k="${BASH_REMATCH[1]}"
      local v="${BASH_REMATCH[2]}"
      if [[ -z "${!k+x}" ]]; then
        export "${k}=${v}"
      fi
    fi
  done < "$file"
}

print_help() {
  cat <<'USAGE'
Usage: run_eval.sh [options] [extra-args]

Options:
  --trace-root PATH
  --skills-root PATH
  --compact-procedure-root PATH
  --benchmark-config PATH
  --task-source-root PATH
  --procmem2skills-root PATH
  --output-root PATH
  --provider NAME
  --api-key-env NAME
  --base-url URL
  --agent NAME
  --model NAME
  --m-success INT
  --n-failure INT
  --n-attempts INT
  --n-concurrent INT
  --max-steps INT
  --command-timeout-sec INT
  --docker-cleanup
  --no-docker-cleanup
  --docker-cleanup-timeout-sec INT
  --docker-cleanup-strict
  --task-name NAME (repeatable)
  --run-id ID
  --workflow-injection-mode MODE (instruction|memory)
  --workflow-hint-mode MODE (with-status|no-hint)
  --workflow-memory-max-attempts INT
  --workflow-memory-max-workflows-per-attempt INT
  --workflow-memory-max-steps-per-workflow INT
  --dry-run
  --help
USAGE
}

load_defaults_if_unset "$DEFAULTS_FILE"

TRACE_ROOT="${TRACE_ROOT:-}"
SKILLS_ROOT="${SKILLS_ROOT:-}"
COMPACT_PROCEDURE_ROOT="${COMPACT_PROCEDURE_ROOT:-}"
BENCHMARK_CONFIG="${CONFIG_BENCHMARKS:-${PROJECT_ROOT}/configs/benchmarks.json}"
PROCMEM2SKILLS_ROOT="${PROCMEM2SKILLS_ROOT:-/raid/zhiyuan/procmem2skills}"
TASK_SOURCE_ROOT="${TASK_SOURCE_ROOT:-${PROCMEM2SKILLS_ROOT}/benchmarks/harbor-datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
PROVIDER="${PROVIDER:-openrouter}"
API_KEY_ENV="${API_KEY_ENV:-}"
BASE_URL="${BASE_URL:-}"
AGENT="${AGENT:-codex}"
MODEL="${MODEL:-gpt-5.3-codex}"
M_SUCCESS="${EVAL_M:-4}"
N_FAILURE="${EVAL_N:-1}"
N_ATTEMPTS="${EVAL_N_ATTEMPTS:-5}"
N_CONCURRENT="${EVAL_N_CONCURRENT:-20}"
MAX_STEPS="${MAX_STEPS:-20}"
COMMAND_TIMEOUT_SEC="${COMMAND_TIMEOUT_SEC:-1200}"
DOCKER_CLEANUP="${DOCKER_CLEANUP:-1}"
DOCKER_CLEANUP_TIMEOUT_SEC="${DOCKER_CLEANUP_TIMEOUT_SEC:-180}"
DOCKER_CLEANUP_STRICT="${DOCKER_CLEANUP_STRICT:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-}"
WORKFLOW_INJECTION_MODE="${WORKFLOW_INJECTION_MODE:-instruction}"
WORKFLOW_HINT_MODE="${WORKFLOW_HINT_MODE:-with-status}"
WORKFLOW_MEMORY_MAX_ATTEMPTS="${WORKFLOW_MEMORY_MAX_ATTEMPTS:-0}"
WORKFLOW_MEMORY_MAX_WORKFLOWS_PER_ATTEMPT="${WORKFLOW_MEMORY_MAX_WORKFLOWS_PER_ATTEMPT:-0}"
WORKFLOW_MEMORY_MAX_STEPS_PER_WORKFLOW="${WORKFLOW_MEMORY_MAX_STEPS_PER_WORKFLOW:-0}"

TASK_NAME_ARGS=()
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --trace-root) TRACE_ROOT="$2"; shift 2 ;;
    --skills-root) SKILLS_ROOT="$2"; shift 2 ;;
    --compact-procedure-root) COMPACT_PROCEDURE_ROOT="$2"; shift 2 ;;
    --benchmark-config) BENCHMARK_CONFIG="$2"; shift 2 ;;
    --task-source-root) TASK_SOURCE_ROOT="$2"; shift 2 ;;
    --procmem2skills-root) PROCMEM2SKILLS_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --api-key-env) API_KEY_ENV="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --agent) AGENT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --m-success) M_SUCCESS="$2"; shift 2 ;;
    --n-failure) N_FAILURE="$2"; shift 2 ;;
    --n-attempts) N_ATTEMPTS="$2"; shift 2 ;;
    --n-concurrent) N_CONCURRENT="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --command-timeout-sec) COMMAND_TIMEOUT_SEC="$2"; shift 2 ;;
    --docker-cleanup) DOCKER_CLEANUP=1; shift ;;
    --no-docker-cleanup) DOCKER_CLEANUP=0; shift ;;
    --docker-cleanup-timeout-sec) DOCKER_CLEANUP_TIMEOUT_SEC="$2"; shift 2 ;;
    --docker-cleanup-strict) DOCKER_CLEANUP_STRICT=1; shift ;;
    --task-name) TASK_NAME_ARGS+=("$2"); shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --workflow-injection-mode) WORKFLOW_INJECTION_MODE="$2"; shift 2 ;;
    --workflow-hint-mode) WORKFLOW_HINT_MODE="$2"; shift 2 ;;
    --workflow-memory-max-attempts) WORKFLOW_MEMORY_MAX_ATTEMPTS="$2"; shift 2 ;;
    --workflow-memory-max-workflows-per-attempt) WORKFLOW_MEMORY_MAX_WORKFLOWS_PER_ATTEMPT="$2"; shift 2 ;;
    --workflow-memory-max-steps-per-workflow) WORKFLOW_MEMORY_MAX_STEPS_PER_WORKFLOW="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h)
      print_help
      echo
      python3 "$PY_SCRIPT" --help
      exit 0
      ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$TRACE_ROOT" ]]; then
  TRACE_ROOT="$(find "$OUTPUT_ROOT" -type d -path '*/pipeline-v1/traces' 2>/dev/null | sort | tail -n 1)"
fi
if [[ -z "$SKILLS_ROOT" ]]; then
  # Use condition-specific skill dir root from latest model tree.
  SKILLS_ROOT="$(find "$OUTPUT_ROOT" -type d -path '*/pipeline-v1/skills' 2>/dev/null | sort | tail -n 1)"
fi
if [[ -z "$TRACE_ROOT" || -z "$SKILLS_ROOT" ]]; then
  echo "ERROR: unable to resolve TRACE_ROOT/SKILLS_ROOT. pass explicitly." >&2
  exit 1
fi
if [[ "$WORKFLOW_INJECTION_MODE" != "instruction" && "$WORKFLOW_INJECTION_MODE" != "memory" ]]; then
  echo "ERROR: --workflow-injection-mode must be instruction or memory (got: $WORKFLOW_INJECTION_MODE)" >&2
  exit 2
fi

if [[ "$WORKFLOW_HINT_MODE" != "with-status" && "$WORKFLOW_HINT_MODE" != "no-hint" ]]; then
  echo "ERROR: --workflow-hint-mode must be with-status or no-hint (got: $WORKFLOW_HINT_MODE)" >&2
  exit 2
fi

if [[ -z "$API_KEY_ENV" ]]; then
  case "$PROVIDER" in
    openai) API_KEY_ENV="OPENAI_API_KEY" ;;
    openrouter) API_KEY_ENV="OPENROUTER_API_KEY" ;;
    google) API_KEY_ENV="GOOGLE_API_KEY" ;;
    claude) API_KEY_ENV="ANTHROPIC_API_KEY" ;;
    uniapi) API_KEY_ENV="UNIAPI_API_KEY" ;;
    *) echo "ERROR: unsupported provider $PROVIDER" >&2; exit 2 ;;
  esac
fi

echo "[RUN_EVAL CHECK] provider=${PROVIDER} api_key_env=${API_KEY_ENV} base_url=${BASE_URL:-<unset>}"
echo "[RUN_EVAL CHECK] agent=${AGENT} model=${MODEL} m_success=${M_SUCCESS} n_failure=${N_FAILURE} n_attempts=${N_ATTEMPTS} n_concurrent=${N_CONCURRENT}"
echo "[RUN_EVAL CHECK] trace_root=${TRACE_ROOT}"
echo "[RUN_EVAL CHECK] skills_root=${SKILLS_ROOT}"
echo "[RUN_EVAL CHECK] compact_procedure_root=${COMPACT_PROCEDURE_ROOT:-<unset>}"
echo "[RUN_EVAL CHECK] benchmark_config=${BENCHMARK_CONFIG} output_root=${OUTPUT_ROOT}"

CMD=(
  python3 "$PY_SCRIPT"
  --trace-root "$TRACE_ROOT"
  --skills-root "$SKILLS_ROOT"
  --benchmark-config "$BENCHMARK_CONFIG"
  --task-source-root "$TASK_SOURCE_ROOT"
  --procmem2skills-root "$PROCMEM2SKILLS_ROOT"
  --output-root "$OUTPUT_ROOT"
  --provider "$PROVIDER"
  --api-key-env "$API_KEY_ENV"
  --agent "$AGENT"
  --model "$MODEL"
  --m-success "$M_SUCCESS"
  --n-failure "$N_FAILURE"
  --n-attempts "$N_ATTEMPTS"
  --n-concurrent "$N_CONCURRENT"
  --max-steps "$MAX_STEPS"
  --command-timeout-sec "$COMMAND_TIMEOUT_SEC"
  --docker-cleanup-timeout-sec "$DOCKER_CLEANUP_TIMEOUT_SEC"
)

if [[ -n "$COMPACT_PROCEDURE_ROOT" ]]; then
  CMD+=(--compact-procedure-root "$COMPACT_PROCEDURE_ROOT")
fi
if [[ -n "$BASE_URL" ]]; then
  CMD+=(--base-url "$BASE_URL")
fi
if [[ -n "$RUN_ID" ]]; then
  CMD+=(--run-id "$RUN_ID")
fi
CMD+=(--workflow-injection-mode "$WORKFLOW_INJECTION_MODE")
CMD+=(--workflow-hint-mode "$WORKFLOW_HINT_MODE")
if [[ "$WORKFLOW_MEMORY_MAX_ATTEMPTS" =~ ^-?[0-9]+$ ]] && (( WORKFLOW_MEMORY_MAX_ATTEMPTS > 0 )); then
  CMD+=(--workflow-memory-max-attempts "$WORKFLOW_MEMORY_MAX_ATTEMPTS")
fi
if [[ "$WORKFLOW_MEMORY_MAX_WORKFLOWS_PER_ATTEMPT" =~ ^-?[0-9]+$ ]] && (( WORKFLOW_MEMORY_MAX_WORKFLOWS_PER_ATTEMPT > 0 )); then
  CMD+=(--workflow-memory-max-workflows-per-attempt "$WORKFLOW_MEMORY_MAX_WORKFLOWS_PER_ATTEMPT")
fi
if [[ "$WORKFLOW_MEMORY_MAX_STEPS_PER_WORKFLOW" =~ ^-?[0-9]+$ ]] && (( WORKFLOW_MEMORY_MAX_STEPS_PER_WORKFLOW > 0 )); then
  CMD+=(--workflow-memory-max-steps-per-workflow "$WORKFLOW_MEMORY_MAX_STEPS_PER_WORKFLOW")
fi
if [[ "$DOCKER_CLEANUP" == "0" || "$DOCKER_CLEANUP" == "false" || "$DOCKER_CLEANUP" == "FALSE" ]]; then
  CMD+=(--no-docker-cleanup)
else
  CMD+=(--docker-cleanup)
fi
if [[ "$DOCKER_CLEANUP_STRICT" == "1" || "$DOCKER_CLEANUP_STRICT" == "true" || "$DOCKER_CLEANUP_STRICT" == "TRUE" ]]; then
  CMD+=(--docker-cleanup-strict)
fi
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
  CMD+=(--dry-run)
fi
if (( ${#TASK_NAME_ARGS[@]} > 0 )); then
  for t in "${TASK_NAME_ARGS[@]}"; do
    CMD+=(--task-name "$t")
  done
fi
if (( ${#EXTRA_ARGS[@]} > 0 )); then
  CMD+=("${EXTRA_ARGS[@]}")
fi

exec "${CMD[@]}"
