#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<USAGE
Usage: run_harbor_terminal_bench.sh [harbor extra args]

Primary mode:
  Run one harbor job with dataset/agent/model/provider from env.

Mixed-collect mode:
  Set MIXED_COLLECT_RAW_PATH and it will iteratively run pending tasks
  until TARGET_SUCCESS and TARGET_FAILURE are reached or ROUND_LIMIT hits.

Common env knobs:
  DATASET DATASET_PATH AGENT MODEL N_ATTEMPTS N_CONCURRENT ENV_TYPE
  TASK_NAME EXCLUDE_TASK_NAME N_TASKS
  PROVIDER(auto|openai|openrouter|uniapi|google) API_KEY_ENV BASE_URL
  PREPULL_IMAGES PREPULL_RETRIES PREPULL_SLEEP_SEC
  MIXED_COLLECT_RAW_PATH TARGET_SUCCESS TARGET_FAILURE ROUND_LIMIT ROUND_PAUSE_SEC

Notes:
  API key must exist in env variable named by API_KEY_ENV.
  Extra CLI args are passed through to harbor run.
USAGE
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PM2S_RESULTS_ROOT:-${CLEAN_ROOT}/outputs}}"
SKILLSBENCH_ROOT="${SKILLSBENCH_ROOT:-}"

DATASET="${DATASET:-terminal-bench@2.0}"
DATASET_PATH="${DATASET_PATH:-}"
AGENT="${AGENT:-codex}"
MODEL="${MODEL:-gpt-5.3-codex}"
N_ATTEMPTS="${N_ATTEMPTS:-5}"
N_CONCURRENT="${N_CONCURRENT:-20}"
ENV_TYPE="${ENV_TYPE:-docker}"

TASK_NAME="${TASK_NAME:-}"
EXCLUDE_TASK_NAME="${EXCLUDE_TASK_NAME:-}"
N_TASKS="${N_TASKS:-}"

JOBS_DIR="${JOBS_DIR:-${CLEAN_ROOT}/jobs}"
JOB_NAME="${JOB_NAME:-}"
QUIET="${QUIET:-0}"
DEBUG="${DEBUG:-0}"

PROVIDER="${PROVIDER:-auto}"
API_KEY_ENV="${API_KEY_ENV:-}"
BASE_URL="${BASE_URL:-}"
PREPULL_IMAGES="${PREPULL_IMAGES:-}"
PREPULL_RETRIES="${PREPULL_RETRIES:-5}"
PREPULL_SLEEP_SEC="${PREPULL_SLEEP_SEC:-15}"

MIXED_COLLECT_RAW_PATH="${MIXED_COLLECT_RAW_PATH:-}"
TARGET_SUCCESS="${TARGET_SUCCESS:-5}"
TARGET_FAILURE="${TARGET_FAILURE:-5}"
ROUND_LIMIT="${ROUND_LIMIT:-50}"
ROUND_PAUSE_SEC="${ROUND_PAUSE_SEC:-0}"

HARBOR_BIN="${HARBOR_BIN:-}"
if [[ -z "${HARBOR_BIN}" ]]; then
  if command -v harbor >/dev/null 2>&1; then
    HARBOR_BIN="$(command -v harbor)"
  elif [[ -x "/raid/zhiyuan/procmem2skills/.venv-py312/bin/harbor" ]]; then
    HARBOR_BIN="/raid/zhiyuan/procmem2skills/.venv-py312/bin/harbor"
  else
    echo "[ERROR] harbor not found in PATH and fallback binary missing." >&2
    exit 1
  fi
fi

dataset_result_slug_from_spec() {
  local spec="${1:-}"
  local text
  text="$(echo "$spec" | tr '[:upper:]' '[:lower:]')"
  case "$text" in
    skillsbench|skillsbench@*|skills-bench|skills-bench@*) echo "skillsbench" ;;
    terminal-bench@2*|terminal-bench) echo "terminal-bench-2" ;;
    terminal-bench-pro@*|terminal-bench-pro) echo "terminal-bench-pro" ;;
    *)
      local base="${text%%@*}"
      base="$(echo "$base" | tr -cs 'a-z0-9._-' '-')"
      base="${base##-}"
      base="${base%%-}"
      [[ -z "$base" ]] && base="dataset"
      echo "$base"
      ;;
  esac
}

dataset_job_prefix_from_spec() {
  local spec="${1:-}"
  local text
  text="$(echo "$spec" | tr '[:upper:]' '[:lower:]')"
  case "$text" in
    skillsbench|skillsbench@*|skills-bench|skills-bench@*) echo "sb" ;;
    terminal-bench@2*|terminal-bench) echo "tb2" ;;
    terminal-bench-pro@*|terminal-bench-pro) echo "tbp" ;;
    *)
      local base="$(dataset_result_slug_from_spec "$spec")"
      echo "${base:0:12}"
      ;;
  esac
}

DATASET_RESULT_SLUG="$(dataset_result_slug_from_spec "$DATASET")"
DATASET_JOB_PREFIX="$(dataset_job_prefix_from_spec "$DATASET")"

resolve_dataset_results_root() {
  local slug="$1"
  local base="$RESULTS_ROOT"
  case "$slug" in
    skillsbench)
      if [[ -d "$base/skillsbench" ]]; then
        echo "$base/skillsbench"
      elif [[ -d "$base/skills-bench" ]]; then
        echo "$base/skills-bench"
      else
        echo "$base/skillsbench"
      fi
      ;;
    *)
      echo "$base/$slug"
      ;;
  esac
}

DATASET_RESULTS_ROOT="$(resolve_dataset_results_root "$DATASET_RESULT_SLUG")"

resolve_dataset_path_mode() {
  if [[ -n "$DATASET_PATH" ]]; then
    [[ -d "$DATASET_PATH" ]] || { echo "[ERROR] DATASET_PATH not found: $DATASET_PATH" >&2; exit 1; }
    DATASET_PATH="$(python3 - "$DATASET_PATH" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
    return 0
  fi

  local text
  text="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"
  case "$text" in
    skillsbench|skillsbench@*|skills-bench|skills-bench@*)
      if [[ -d "$SKILLSBENCH_ROOT/tasks" ]]; then
        DATASET_PATH="$SKILLSBENCH_ROOT/tasks"
      elif [[ -d "$SKILLSBENCH_ROOT" ]]; then
        DATASET_PATH="$SKILLSBENCH_ROOT"
      fi
      ;;
  esac
}

append_dataset_selector_args() {
  local -n _cmd_ref="$1"
  if [[ -n "$DATASET_PATH" ]]; then
    _cmd_ref+=( -p "$DATASET_PATH" )
  else
    _cmd_ref+=( -d "$DATASET" )
  fi
}

resolve_dataset_path_mode

if [[ -z "$JOB_NAME" ]]; then
  JOB_NAME="${DATASET_JOB_PREFIX}-$(date +%Y%m%d-%H%M%S)"
fi

choose_provider() {
  local p="${PROVIDER}"
  if [[ "$p" == "auto" ]]; then
    if [[ -n "${UNIAPI_API_KEY:-}" ]]; then
      p="uniapi"
    elif [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
      p="openrouter"
    else
      p="openrouter"
    fi
  fi

  case "$p" in
    openai)
      : "${API_KEY_ENV:=OPENAI_API_KEY}"
      : "${BASE_URL:=}"
      ;;
    openrouter)
      : "${API_KEY_ENV:=OPENROUTER_API_KEY}"
      : "${BASE_URL:=https://openrouter.ai/api/v1}"
      ;;
    uniapi)
      : "${API_KEY_ENV:=UNIAPI_API_KEY}"
      : "${BASE_URL:=https://api.uniapi.io/v1}"
      ;;
    google)
      : "${API_KEY_ENV:=GOOGLE_API_KEY}"
      : "${BASE_URL:=https://generativelanguage.googleapis.com/v1beta/openai}"
      ;;
    *)
      echo "[ERROR] Unsupported PROVIDER=${p}. Use auto|openai|openrouter|uniapi|google" >&2
      exit 1
      ;;
  esac
  PROVIDER="$p"
}

choose_provider

prepull_images() {
  [[ -n "${PREPULL_IMAGES}" ]] || return 0
  local img
  local attempt
  local ok
  for img in ${PREPULL_IMAGES}; do
    echo "[INFO] Pre-pulling image: ${img}"
    ok=0
    for ((attempt = 1; attempt <= PREPULL_RETRIES; attempt++)); do
      if docker pull "${img}"; then
        ok=1
        break
      fi
      echo "[WARN] docker pull failed attempt=${attempt}/${PREPULL_RETRIES} image=${img}" >&2
      if (( attempt < PREPULL_RETRIES )); then
        sleep "${PREPULL_SLEEP_SEC}"
      fi
    done
    if [[ "${ok}" != "1" ]]; then
      echo "[ERROR] Failed to pre-pull image after ${PREPULL_RETRIES} attempts: ${img}" >&2
      exit 3
    fi
  done
}

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "[ERROR] Missing API key env: ${API_KEY_ENV}" >&2
  echo "Try one of: export OPENROUTER_API_KEY=... or export UNIAPI_API_KEY=..." >&2
  exit 1
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-${!API_KEY_ENV}}"
if [[ -n "$BASE_URL" ]]; then
  export OPENAI_BASE_URL="$BASE_URL"
fi
if [[ "$PROVIDER" == "google" ]]; then
  export GOOGLE_API_KEY="${!API_KEY_ENV}"
  export GEMINI_API_KEY="${!API_KEY_ENV}"
  export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-$BASE_URL}"
fi

prepull_images

EXTRA_ARGS=("$@")

build_mixed_seed_tasks() {
  local raw_path="$1"
  local include_csv="$2"
  local exclude_csv="$3"
  local output_file="$4"

  python3 - "$raw_path" "$include_csv" "$exclude_csv" "$output_file" <<'PY'
import fnmatch
import json
import sys
from pathlib import Path

raw_path = Path(sys.argv[1]).resolve()
include_csv = sys.argv[2]
exclude_csv = sys.argv[3]
out_path = Path(sys.argv[4]).resolve()

include_patterns = [s.strip() for s in include_csv.split(",") if s.strip()]
exclude_patterns = [s.strip() for s in exclude_csv.split(",") if s.strip()]

def reward_of(obj: dict):
    vr = obj.get("verifier_result") or {}
    rw = (vr.get("rewards") or {}).get("reward")
    if isinstance(rw, (int, float)):
        return float(rw)
    rw2 = obj.get("reward")
    if isinstance(rw2, (int, float)):
        return float(rw2)
    return None

def has_trajectory(result_path: Path, obj: dict) -> bool:
    trial_uri = obj.get("trial_uri")
    if isinstance(trial_uri, str) and trial_uri.startswith("file://"):
        trial_dir = Path(trial_uri[7:])
        if (trial_dir / "agent" / "trajectory.json").is_file():
            return True
    return (result_path.parent / "agent" / "trajectory.json").is_file()

counts = {}
for rp in raw_path.rglob("result.json"):
    try:
        obj = json.loads(rp.read_text())
    except Exception:
        continue
    if not isinstance(obj, dict):
        continue
    task = obj.get("task_name")
    trial = obj.get("trial_name")
    if not isinstance(task, str) or not isinstance(trial, str):
        continue

    if include_patterns and not any(fnmatch.fnmatch(task, p) for p in include_patterns):
        continue
    if exclude_patterns and any(fnmatch.fnmatch(task, p) for p in exclude_patterns):
        continue
    if not has_trajectory(rp, obj):
        continue

    rw = reward_of(obj)
    success = (rw is not None and rw >= 1.0 and not obj.get("exception_info"))

    c = counts.setdefault(task, {"success": 0, "failure": 0})
    if success:
        c["success"] += 1
    else:
        c["failure"] += 1

mixed = sorted([t for t, c in counts.items() if c["success"] > 0 and c["failure"] > 0])
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(mixed) + ("\n" if mixed else ""), encoding="utf-8")
print(f"mixed_seed_tasks={len(mixed)}")
PY
}

compute_active_tasks() {
  local raw_path="$1"
  local collect_jobs_dir="$2"
  local seed_tasks_file="$3"
  local target_success="$4"
  local target_failure="$5"
  local active_out="$6"

  python3 - "$raw_path" "$collect_jobs_dir" "$seed_tasks_file" "$target_success" "$target_failure" "$active_out" <<'PY'
import json
import sys
from pathlib import Path

raw_path = Path(sys.argv[1]).resolve()
collect_jobs_dir = Path(sys.argv[2]).resolve()
seed_tasks_file = Path(sys.argv[3]).resolve()
target_success = int(sys.argv[4])
target_failure = int(sys.argv[5])
active_out = Path(sys.argv[6]).resolve()

seed_tasks = [ln.strip() for ln in seed_tasks_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
seed_set = set(seed_tasks)

if not seed_tasks:
    active_out.write_text("", encoding="utf-8")
    print("active_tasks=0")
    sys.exit(0)

def reward_of(obj: dict):
    vr = obj.get("verifier_result") or {}
    rw = (vr.get("rewards") or {}).get("reward")
    if isinstance(rw, (int, float)):
        return float(rw)
    rw2 = obj.get("reward")
    if isinstance(rw2, (int, float)):
        return float(rw2)
    return None

def has_trajectory(result_path: Path, obj: dict) -> bool:
    trial_uri = obj.get("trial_uri")
    if isinstance(trial_uri, str) and trial_uri.startswith("file://"):
        trial_dir = Path(trial_uri[7:])
        if (trial_dir / "agent" / "trajectory.json").is_file():
            return True
    return (result_path.parent / "agent" / "trajectory.json").is_file()

def fold_counts(root: Path, counts: dict):
    if not root.exists():
        return
    for rp in root.rglob("result.json"):
        try:
            obj = json.loads(rp.read_text())
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        task = obj.get("task_name")
        trial = obj.get("trial_name")
        if not isinstance(task, str) or not isinstance(trial, str):
            continue
        if task not in seed_set:
            continue
        if not has_trajectory(rp, obj):
            continue
        rw = reward_of(obj)
        success = (rw is not None and rw >= 1.0 and not obj.get("exception_info"))
        c = counts.setdefault(task, {"success": 0, "failure": 0})
        if success:
            c["success"] += 1
        else:
            c["failure"] += 1

counts = {t: {"success": 0, "failure": 0} for t in seed_tasks}
fold_counts(raw_path, counts)
fold_counts(collect_jobs_dir, counts)

active = []
for t in seed_tasks:
    s = counts[t]["success"]
    f = counts[t]["failure"]
    need_s = max(target_success - s, 0)
    need_f = max(target_failure - f, 0)
    if need_s > 0 or need_f > 0:
        active.append(t)
    print(f"[STATUS] task={t} success={s} failure={f} need_success={need_s} need_failure={need_f}")

active_out.parent.mkdir(parents=True, exist_ok=True)
active_out.write_text("\n".join(active) + ("\n" if active else ""), encoding="utf-8")
print(f"active_tasks={len(active)}")
PY
}

run_harbor_once() {
  local jobs_dir="$1"
  local job_name="$2"
  shift 2
  local -a task_list=("$@")

  local -a cmd=("${HARBOR_BIN}" run)
  append_dataset_selector_args cmd
  cmd+=(
    -a "${AGENT}"
    -m "${MODEL}"
    -k "${N_ATTEMPTS}"
    -n "${N_CONCURRENT}"
    -e "${ENV_TYPE}"
    --jobs-dir "${jobs_dir}"
    --job-name "${job_name}"
  )

  [[ "${QUIET}" == "1" ]] && cmd+=(-q)
  [[ "${DEBUG}" == "1" ]] && cmd+=(--debug)

  local t
  for t in "${task_list[@]}"; do
    cmd+=(--task-name "${t}")
  done
  [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && cmd+=("${EXTRA_ARGS[@]}")

  echo "[INFO] Running: ${cmd[*]}"
  "${cmd[@]}"
}

run_single_mode() {
  mkdir -p "${JOBS_DIR}"

  local -a cmd=("${HARBOR_BIN}" run)
  append_dataset_selector_args cmd
  cmd+=(
    -a "${AGENT}"
    -m "${MODEL}"
    -k "${N_ATTEMPTS}"
    -n "${N_CONCURRENT}"
    -e "${ENV_TYPE}"
    --jobs-dir "${JOBS_DIR}"
    --job-name "${JOB_NAME}"
  )

  [[ -n "${TASK_NAME}" ]] && cmd+=(--task-name "${TASK_NAME}")
  [[ -n "${EXCLUDE_TASK_NAME}" ]] && cmd+=(--exclude-task-name "${EXCLUDE_TASK_NAME}")
  [[ -n "${N_TASKS}" ]] && cmd+=(--n-tasks "${N_TASKS}")
  [[ "${QUIET}" == "1" ]] && cmd+=(-q)
  [[ "${DEBUG}" == "1" ]] && cmd+=(--debug)
  [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && cmd+=("${EXTRA_ARGS[@]}")

  echo "[INFO] PROVIDER=${PROVIDER} API_KEY_ENV=${API_KEY_ENV} OPENAI_BASE_URL=${OPENAI_BASE_URL:-<unset>}"
  echo "[INFO] DATASET=${DATASET} DATASET_PATH=${DATASET_PATH:-<registry>} DATASET_RESULT_SLUG=${DATASET_RESULT_SLUG} JOB_NAME=${JOB_NAME}"
  echo "[INFO] DATASET_RESULTS_ROOT=${DATASET_RESULTS_ROOT}"
  echo "[RUN_HARBOR CHECK] jobs_dir=${JOBS_DIR} n_attempts=${N_ATTEMPTS} n_concurrent=${N_CONCURRENT} env_type=${ENV_TYPE} provider=${PROVIDER}"
  echo "[INFO] Running: ${cmd[*]}"
  exec "${cmd[@]}"
}

run_mixed_collect_mode() {
  local raw_path="${MIXED_COLLECT_RAW_PATH}"
  [[ -d "${raw_path}" ]] || { echo "[ERROR] MIXED_COLLECT_RAW_PATH not found: ${raw_path}" >&2; exit 1; }

  local collect_jobs_dir="${JOBS_DIR%/}/${JOB_NAME}-mixed-collect"
  mkdir -p "${collect_jobs_dir}"

  local seed_tasks_file="${collect_jobs_dir}/seed_mixed_tasks.txt"
  local active_tasks_file="${collect_jobs_dir}/active_tasks.txt"

  echo "[INFO] Building mixed seed tasks from: ${raw_path}"
  build_mixed_seed_tasks "${raw_path}" "${TASK_NAME}" "${EXCLUDE_TASK_NAME}" "${seed_tasks_file}"

  local seed_count
  seed_count="$(wc -l < "${seed_tasks_file}" | tr -d '[:space:]')"
  echo "[INFO] mixed_seed_tasks=${seed_count}"
  [[ "${seed_count}" == "0" ]] && { echo "[INFO] No mixed tasks found, nothing to collect."; return 0; }

  local round=1
  while (( round <= ROUND_LIMIT )); do
    echo "[INFO] ===== Round ${round}/${ROUND_LIMIT} ====="
    compute_active_tasks "${raw_path}" "${collect_jobs_dir}" "${seed_tasks_file}" "${TARGET_SUCCESS}" "${TARGET_FAILURE}" "${active_tasks_file}"

    mapfile -t active_tasks < "${active_tasks_file}"
    if [[ ${#active_tasks[@]} -eq 0 ]]; then
      echo "[INFO] All mixed tasks reached success>=${TARGET_SUCCESS} and failure>=${TARGET_FAILURE}."
      break
    fi

    if [[ -n "${N_TASKS}" ]] && [[ "${N_TASKS}" =~ ^[0-9]+$ ]] && (( N_TASKS < ${#active_tasks[@]} )); then
      active_tasks=("${active_tasks[@]:0:N_TASKS}")
    fi

    echo "[INFO] pending_tasks_this_round=${#active_tasks[@]}"
    run_harbor_once "${collect_jobs_dir}" "${JOB_NAME}-r$(printf "%04d" "${round}")" "${active_tasks[@]}"

    if [[ "${ROUND_PAUSE_SEC}" =~ ^[0-9]+$ ]] && (( ROUND_PAUSE_SEC > 0 )); then
      sleep "${ROUND_PAUSE_SEC}"
    fi
    ((round++))
  done

  if (( round > ROUND_LIMIT )); then
    echo "[WARN] Reached ROUND_LIMIT=${ROUND_LIMIT}. Remaining tasks (if any):"
    compute_active_tasks "${raw_path}" "${collect_jobs_dir}" "${seed_tasks_file}" "${TARGET_SUCCESS}" "${TARGET_FAILURE}" "${active_tasks_file}"
  fi

  echo "[INFO] Mixed collection outputs: ${collect_jobs_dir}"
}

echo "[INFO] PROVIDER=${PROVIDER} API_KEY_ENV=${API_KEY_ENV} OPENAI_BASE_URL=${OPENAI_BASE_URL:-<unset>}"

if [[ -n "${MIXED_COLLECT_RAW_PATH}" ]]; then
  run_mixed_collect_mode
else
  run_single_mode
fi
