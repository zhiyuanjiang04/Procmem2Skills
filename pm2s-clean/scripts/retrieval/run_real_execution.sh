#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLEAN_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT="${ROOT:-${PM2S_RETRIEVAL_ROOT:-${CLEAN_ROOT}/retrieval-data}}"
PM2S_ROOT="${PM2S_ROOT:-${CLEAN_ROOT}}"
PROCMEM2SKILLS_ROOT="${PROCMEM2SKILLS_ROOT:-/raid/zhiyuan/procmem2skills}"

CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:-}"
if [[ -z "${CANDIDATE_MANIFEST}" ]]; then
  echo "ERROR: set CANDIDATE_MANIFEST to execution_manifests/.../seed-*.json" >&2
  exit 2
fi
if [[ ! -f "${CANDIDATE_MANIFEST}" ]]; then
  echo "ERROR: CANDIDATE_MANIFEST missing: ${CANDIDATE_MANIFEST}" >&2
  exit 2
fi

BENCHMARK="${BENCHMARK:-skillsbench}"
TRACE_ROOT="${TRACE_ROOT:-${ROOT}/pools/trace_stubs/${BENCHMARK}}"
SKILLS_ROOT="${SKILLS_ROOT:-${ROOT}/pools/gt_primary/${BENCHMARK}}"
AGENT="${AGENT:-claude-code}"
MODEL="${MODEL:-claude-sonnet-4-6}"
PROVIDER="${PROVIDER:-claude}"
API_KEY_ENV="${API_KEY_ENV:-ANTHROPIC_API_KEY}"
BASE_URL="${BASE_URL:-}"
USE_CLAUDE_CODE_OAUTH="${USE_CLAUDE_CODE_OAUTH:-0}"
CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-}"
CLAUDE_CODE_CREDENTIALS_FILE="${CLAUDE_CODE_CREDENTIALS_FILE:-}"
if [[ -z "${CLAUDE_CODE_CREDENTIALS_FILE}" && -n "${CLAUDE_CONFIG_DIR}" ]]; then
  CLAUDE_CODE_CREDENTIALS_FILE="${CLAUDE_CONFIG_DIR}/.credentials.json"
fi
CLAUDE_CLI="${CLAUDE_CLI:-$(command -v claude || true)}"
if [[ -z "${CLAUDE_CLI}" && -x "/root/.local/bin/claude" ]]; then
  CLAUDE_CLI="/root/.local/bin/claude"
fi
N_ATTEMPTS="${N_ATTEMPTS:-5}"
N_CONCURRENT="${N_CONCURRENT:-10}"
MAX_STEPS="${MAX_STEPS:-20}"
COMMAND_TIMEOUT_SEC="${COMMAND_TIMEOUT_SEC:-1200}"
RUN_ID="${RUN_ID:-skill-retrieval-$(date +%Y%m%d-%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

NOISE_MODE="${NOISE_MODE:-}"
POOL_SIZE="${POOL_SIZE:-}"
SEED="${SEED:-}"
if [[ "${CANDIDATE_MANIFEST}" =~ /execution_manifests/([^/]+)/([^/]+)/k([0-9]+)/seed-([0-9]+)\.json$ ]]; then
  BENCHMARK="${BENCHMARK:-${BASH_REMATCH[1]}}"
  NOISE_MODE="${NOISE_MODE:-${BASH_REMATCH[2]}}"
  POOL_SIZE="${POOL_SIZE:-${BASH_REMATCH[3]}}"
  SEED="${SEED:-${BASH_REMATCH[4]}}"
else
  if [[ "${CANDIDATE_MANIFEST}" =~ /([^/]+)/k([0-9]+)/seed-([0-9]+)\.json$ ]]; then
    NOISE_MODE="${NOISE_MODE:-${BASH_REMATCH[1]}}"
    POOL_SIZE="${POOL_SIZE:-${BASH_REMATCH[2]}}"
    SEED="${SEED:-${BASH_REMATCH[3]}}"
  fi
fi
NOISE_MODE="${NOISE_MODE:-unknown}"
POOL_SIZE="${POOL_SIZE:-0}"
SEED="${SEED:-0}"
MODEL_LEAF="${MODEL##*/}"
AGENT_MODEL="$(printf '%s-%s' "${AGENT}" "${MODEL_LEAF}" | sed -E 's/[^A-Za-z0-9._-]+/-/g; s/^[-._]+//; s/[-._]+$//')"
SETTING_ROOT="${ROOT}/outputs/${BENCHMARK}/${AGENT_MODEL}/real_execution/${NOISE_MODE}/k${POOL_SIZE}/seed-${SEED}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SETTING_ROOT}/raw}"

if [[ "${AGENT}" == "claude-code" && ( "${USE_CLAUDE_CODE_OAUTH}" == "1" || "${USE_CLAUDE_CODE_OAUTH}" == "true" || "${USE_CLAUDE_CODE_OAUTH}" == "TRUE" ) ]]; then
  if [[ ! -f "${CLAUDE_CODE_CREDENTIALS_FILE}" ]]; then
    echo "ERROR: USE_CLAUDE_CODE_OAUTH=1 but credentials file is missing: ${CLAUDE_CODE_CREDENTIALS_FILE}" >&2
    exit 2
  fi
  export CLAUDE_CODE_CREDENTIALS_FILE
  export CLAUDE_CODE_CREDENTIALS_PATH="${CLAUDE_CODE_CREDENTIALS_PATH:-${CLAUDE_CODE_CREDENTIALS_FILE}}"
  export CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$(dirname "${CLAUDE_CODE_CREDENTIALS_FILE}")}"
  if [[ -z "${CLAUDE_CLI}" ]]; then
    echo "ERROR: claude CLI not found in PATH and /root/.local/bin/claude missing" >&2
    exit 3
  fi
  unset CLAUDE_CODE_OAUTH_TOKEN
  if ! timeout 60 "${CLAUDE_CLI}" -p "Reply OK only" --model "${MODEL}" --no-session-persistence >/dev/null 2>&1; then
    echo "ERROR: host Claude Code OAuth is not usable; run claude auth login before launching Harbor jobs." >&2
    exit 3
  fi
fi

CMD=(
  python3 "${PM2S_ROOT}/scripts/run_context_comparison.py"
  --trace-root "${TRACE_ROOT}"
  --skills-root "${SKILLS_ROOT}"
  --skills-manifest "${CANDIDATE_MANIFEST}"
  --benchmark-config "${PM2S_ROOT}/configs/benchmarks.json"
  --task-source-root "${PROCMEM2SKILLS_ROOT}/benchmarks/harbor-datasets"
  --procmem2skills-root "${PROCMEM2SKILLS_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --output-layout normal
  --benchmark-output "${BENCHMARK}"
  --provider "${PROVIDER}"
  --api-key-env "${API_KEY_ENV}"
  --agent "${AGENT}"
  --model "${MODEL}"
  --m-success 0
  --n-failure 0
  --n-attempts "${N_ATTEMPTS}"
  --n-concurrent "${N_CONCURRENT}"
  --max-steps "${MAX_STEPS}"
  --command-timeout-sec "${COMMAND_TIMEOUT_SEC}"
  --docker-cleanup
  --arms skill
  --run-id "${RUN_ID}"
)
if [[ -n "${BASE_URL}" ]]; then
  CMD+=(--base-url "${BASE_URL}")
fi
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "TRUE" ]]; then
  CMD+=(--dry-run)
fi

echo "[SKILL_RETRIEVAL_EXEC CHECK] candidate_manifest=${CANDIDATE_MANIFEST}"
echo "[SKILL_RETRIEVAL_EXEC CHECK] trace_root=${TRACE_ROOT}"
echo "[SKILL_RETRIEVAL_EXEC CHECK] skills_root=${SKILLS_ROOT}"
echo "[SKILL_RETRIEVAL_EXEC CHECK] output_root=${OUTPUT_ROOT}"
echo "[SKILL_RETRIEVAL_EXEC CHECK] standard_summary_root=${SETTING_ROOT}/${RUN_ID}"
echo "[SKILL_RETRIEVAL_EXEC CHECK] agent=${AGENT} model=${MODEL} provider=${PROVIDER} api_key_env=${API_KEY_ENV}"
if [[ "${AGENT}" == "claude-code" && ( "${USE_CLAUDE_CODE_OAUTH}" == "1" || "${USE_CLAUDE_CODE_OAUTH}" == "true" || "${USE_CLAUDE_CODE_OAUTH}" == "TRUE" ) ]]; then
  echo "[SKILL_RETRIEVAL_EXEC CHECK] claude_code_oauth=enabled source=${CLAUDE_CODE_CREDENTIALS_FILE}"
  echo "[SKILL_RETRIEVAL_EXEC CHECK] claude_config_dir=${CLAUDE_CONFIG_DIR:-<unset>} claude_cli=${CLAUDE_CLI:-<unset>}"
fi
echo "[SKILL_RETRIEVAL_EXEC CHECK] after run: python3 ${ROOT}/scripts/summarize_execution_retrieval.py --candidate-manifest ${CANDIDATE_MANIFEST} --run-root ${OUTPUT_ROOT}/${BENCHMARK}/0s0f/runs/${RUN_ID} --benchmark ${BENCHMARK} --agent ${AGENT} --model ${MODEL}"
exec "${CMD[@]}"
