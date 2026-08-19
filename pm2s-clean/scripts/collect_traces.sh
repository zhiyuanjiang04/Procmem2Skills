#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<USAGE
Usage: collect_mixed_tasks.sh [harbor extra args]

Iteratively runs TRIALS_PER_ROUND per pending task, keeping only tasks with
both success and failure evidence, until each task reaches TARGET_SUCCESS and
TARGET_FAILURE, or MAX_ATTEMPT_BLOCKS rounds are exhausted.

Primary env knobs:
  RAW_RESULTS_PATH OUTPUT_PATH AGENT MODEL MAX_ATTEMPT_BLOCKS

Common env knobs:
  DATASET DATASET_PATH TRIALS_PER_ROUND N_CONCURRENT ENV_TYPE
  TARGET_SUCCESS TARGET_FAILURE ROUND_PAUSE_SEC
  PROVIDER(auto|openai|openrouter|uniapi|google|gemini) API_KEY_ENV BASE_URL
  CONTINUE_FROM_EXISTING CLEANUP_AFTER_ROUND CLEANUP_PRUNE
USAGE
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PM2S_RESULTS_ROOT:-${CLEAN_ROOT}/outputs}}"
SKILLSBENCH_ROOT="${SKILLSBENCH_ROOT:-}"

DATASET="${DATASET:-terminal-bench@2.0}"
DATASET_PATH="${DATASET_PATH:-}"


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

DATASET_RESULT_SLUG="$(dataset_result_slug_from_spec "$DATASET")"

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

RAW_RESULTS_PATH="${RAW_RESULTS_PATH:-${DATASET_RESULTS_ROOT}/results/raw}"
OUTPUT_PATH="${OUTPUT_PATH:-${DATASET_RESULTS_ROOT}/mixed-collect-$(date +%Y%m%d-%H%M%S)}"
AGENT="${AGENT:-codex}"
MODEL="${MODEL:-gpt-5.3-codex}"
MAX_ATTEMPT_BLOCKS="${MAX_ATTEMPT_BLOCKS:-10}"
CONTINUE_FROM_EXISTING="${CONTINUE_FROM_EXISTING:-0}"

TRIALS_PER_ROUND="${TRIALS_PER_ROUND:-5}"
N_CONCURRENT="${N_CONCURRENT:-20}"
ENV_TYPE="${ENV_TYPE:-docker}"
TARGET_SUCCESS="${TARGET_SUCCESS:-5}"
TARGET_FAILURE="${TARGET_FAILURE:-5}"
ROUND_PAUSE_SEC="${ROUND_PAUSE_SEC:-0}"
CLEANUP_AFTER_ROUND="${CLEANUP_AFTER_ROUND:-1}"
CLEANUP_PRUNE="${CLEANUP_PRUNE:-1}"
# Optional Docker build mode for compose build steps.
# Set to 0 to force classic builder when BuildKit metadata requests are flaky.
DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

TASK_NAME="${TASK_NAME:-}"
EXCLUDE_TASK_NAME="${EXCLUDE_TASK_NAME:-}"
N_TASKS="${N_TASKS:-}"

PROVIDER="${PROVIDER:-auto}"
API_KEY_ENV="${API_KEY_ENV:-}"
BASE_URL="${BASE_URL:-}"
GEMINI_CLI_VERSION="${GEMINI_CLI_VERSION:-0.34.0}"

QUIET="${QUIET:-0}"
DEBUG="${DEBUG:-0}"

HARBOR_BIN="${HARBOR_BIN:-}"
if [[ -z "${HARBOR_BIN}" ]]; then
  # IMPORTANT: prefer py312 harbor from procmem2skills first.
  # That build includes Codex provider config compatibility needed for UniAPI/OpenRouter routing.
  if [[ -x "/raid/zhiyuan/procmem2skills/.venv-py312/bin/harbor" ]]; then
    HARBOR_BIN="/raid/zhiyuan/procmem2skills/.venv-py312/bin/harbor"
  elif [[ -x "/root/.local/bin/harbor" ]]; then
    HARBOR_BIN="/root/.local/bin/harbor"
  elif [[ -x "/root/.local/share/uv/tools/harbor/bin/harbor" ]]; then
    HARBOR_BIN="/root/.local/share/uv/tools/harbor/bin/harbor"
  elif command -v harbor >/dev/null 2>&1; then
    HARBOR_BIN="$(command -v harbor)"
  else
    echo "[ERROR] harbor not found in PATH and fallback binary missing." >&2
    exit 1
  fi
fi

choose_provider() {
  local p="${PROVIDER}"
  local agent_name
  agent_name="$(echo "${AGENT:-}" | tr '[:upper:]' '[:lower:]')"
  if [[ "$p" == "auto" ]]; then
    if [[ "$agent_name" == "gemini-cli" || "$agent_name" == "gemini" ]]; then
      if [[ -n "${GOOGLE_API_KEY:-}" || -n "${GEMINI_API_KEY:-}" ]]; then
        p="google"
      elif [[ -n "${UNIAPI_API_KEY:-}" ]]; then
        p="uniapi"
      elif [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
        p="openrouter"
      else
        p="google"
      fi
    elif [[ -n "${UNIAPI_API_KEY:-}" ]]; then
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
    google|gemini)
      if [[ -z "$API_KEY_ENV" ]]; then
        if [[ -n "${GOOGLE_API_KEY:-}" ]]; then
          API_KEY_ENV="GOOGLE_API_KEY"
        else
          API_KEY_ENV="GEMINI_API_KEY"
        fi
      fi
      : "${BASE_URL:=https://generativelanguage.googleapis.com/v1beta}"
      p="google"
      ;;
    *)
      echo "[ERROR] Unsupported PROVIDER=${p}. Use auto|openai|openrouter|uniapi|google|gemini" >&2
      exit 1
      ;;
  esac
  PROVIDER="$p"
}

choose_provider

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "[ERROR] Missing API key env: ${API_KEY_ENV}" >&2
  if [[ "$PROVIDER" == "google" ]]; then
    echo "Try one of: export GOOGLE_API_KEY=... or export GEMINI_API_KEY=..." >&2
  else
    echo "Try one of: export OPENROUTER_API_KEY=... or export UNIAPI_API_KEY=..." >&2
  fi
  exit 1
fi

ACTIVE_API_KEY="${!API_KEY_ENV}"
if [[ "$PROVIDER" == "google" ]]; then
  export GOOGLE_API_KEY="${GOOGLE_API_KEY:-$ACTIVE_API_KEY}"
  export GEMINI_API_KEY="${GEMINI_API_KEY:-$ACTIVE_API_KEY}"
  export GEMINI_CLI_TRUST_WORKSPACE="${GEMINI_CLI_TRUST_WORKSPACE:-true}"
else
  export OPENAI_API_KEY="${OPENAI_API_KEY:-$ACTIVE_API_KEY}"
fi

# Codex-style OpenAI-compatible routing only applies to non-Gemini providers.
if [[ -n "$BASE_URL" && "$PROVIDER" != "google" ]]; then
  export CODEX_HOME="${CODEX_HOME:-/logs/agent}"
  mkdir -p "$CODEX_HOME"
  if [[ "$PROVIDER" == "uniapi" ]]; then
    cat > "$CODEX_HOME/config.toml" <<EOF
model_provider = "openrouter"

[model_providers.openrouter]
name = "OpenRouter Responses"
base_url = "$BASE_URL"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
stream_max_retries = 8
stream_idle_timeout_ms = 300000
EOF
  else
    cat > "$CODEX_HOME/config.toml" <<EOF
model_provider = "openai"
openai_base_url = "$BASE_URL"

[model_providers.openai]
name = "OpenAI Compatible"
base_url = "$BASE_URL"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
stream_max_retries = 8
stream_idle_timeout_ms = 300000
EOF
  fi
fi

api_health_check() {
  if [[ "$PROVIDER" == "google" ]]; then
    local endpoint="${BASE_URL%/}/models"
    python3 - "$endpoint" <<'PY'
import os
import sys
import urllib.error
import urllib.request

endpoint = sys.argv[1]
key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
if not key:
    print("[ERROR] GOOGLE_API_KEY/GEMINI_API_KEY is empty after mapping from API_KEY_ENV", file=sys.stderr)
    raise SystemExit(2)

req = urllib.request.Request(endpoint, headers={"x-goog-api-key": key})
try:
    with urllib.request.urlopen(req, timeout=20) as resp:
        code = getattr(resp, "status", 200)
        if code != 200:
            print(f"[ERROR] Gemini API health check failed: status={code} endpoint={endpoint}", file=sys.stderr)
            raise SystemExit(2)
        print(f"[INFO] Gemini API health check passed: status={code} endpoint={endpoint}")
except urllib.error.HTTPError as e:
    body = e.read(300).decode("utf-8", "ignore")
    print(f"[ERROR] Gemini API health check HTTPError: status={e.code} endpoint={endpoint} body={body}", file=sys.stderr)
    raise SystemExit(2)
except Exception as e:
    print(f"[ERROR] Gemini API health check exception: endpoint={endpoint} err={type(e).__name__}: {e}", file=sys.stderr)
    raise SystemExit(2)
PY
    return
  fi

  local endpoint="${BASE_URL:-https://api.openai.com/v1}/models"
  python3 - "$endpoint" <<'PY'
import os
import sys
import urllib.error
import urllib.request

endpoint = sys.argv[1]
key = os.getenv("OPENAI_API_KEY", "")
if not key:
    print("[ERROR] OPENAI_API_KEY is empty after mapping from API_KEY_ENV", file=sys.stderr)
    raise SystemExit(2)

req = urllib.request.Request(endpoint, headers={"Authorization": f"Bearer {key}"})
try:
    with urllib.request.urlopen(req, timeout=20) as resp:
        code = getattr(resp, "status", 200)
        if code != 200:
            print(f"[ERROR] API health check failed: status={code} endpoint={endpoint}", file=sys.stderr)
            raise SystemExit(2)
        print(f"[INFO] API health check passed: status={code} endpoint={endpoint}")
except urllib.error.HTTPError as e:
    body = e.read(300).decode("utf-8", "ignore")
    print(f"[ERROR] API health check HTTPError: status={e.code} endpoint={endpoint} body={body}", file=sys.stderr)
    raise SystemExit(2)
except Exception as e:
    print(f"[ERROR] API health check exception: endpoint={endpoint} err={type(e).__name__}: {e}", file=sys.stderr)
    raise SystemExit(2)
PY
}

api_health_check

export DOCKER_BUILDKIT

RAW_REAL="$(python3 - "$RAW_RESULTS_PATH" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
OUT_REAL="$(python3 - "$OUTPUT_PATH" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"

[[ -d "$RAW_REAL" ]] || { echo "[ERROR] RAW_RESULTS_PATH not found: $RAW_REAL" >&2; exit 1; }

if [[ "$OUT_REAL" == "$RAW_REAL" || "$OUT_REAL" == "$RAW_REAL"/* || "$RAW_REAL" == "$OUT_REAL"/* ]]; then
  echo "[ERROR] OUTPUT_PATH overlaps RAW_RESULTS_PATH" >&2
  echo "[ERROR] raw=$RAW_REAL" >&2
  echo "[ERROR] out=$OUT_REAL" >&2
  exit 1
fi

mkdir -p "$OUT_REAL"
SEED_FILE="$OUT_REAL/seed_mixed_tasks.txt"
ACTIVE_FILE="$OUT_REAL/active_tasks.txt"
STATUS_LOG="$OUT_REAL/status.log"
META_LOG="$OUT_REAL/meta.log"
EXTRA_ARGS=("$@")

build_seed() {
  python3 - "$RAW_REAL" "$TASK_NAME" "$EXCLUDE_TASK_NAME" "$SEED_FILE" <<'PY'
import fnmatch
import json
import sys
from pathlib import Path

raw = Path(sys.argv[1])
inc = [s.strip() for s in sys.argv[2].split(",") if s.strip()]
exc = [s.strip() for s in sys.argv[3].split(",") if s.strip()]
out = Path(sys.argv[4])

def reward_of(o):
    vr = o.get("verifier_result") or {}
    rw = (vr.get("rewards") or {}).get("reward")
    if isinstance(rw, (int, float)):
        return float(rw)
    rw2 = o.get("reward")
    if isinstance(rw2, (int, float)):
        return float(rw2)
    return None

def has_traj(rp, o):
    def has_agent_evidence(td):
        agent_dir = td / "agent"
        return any((agent_dir / name).is_file() for name in (
            "trajectory.json",
            "gemini-cli.txt",
            "codex.txt",
            "claude.txt",
        ))

    tu = o.get("trial_uri")
    if isinstance(tu, str) and tu.startswith("file://"):
        td = Path(tu[7:])
        if has_agent_evidence(td):
            return True
    return has_agent_evidence(rp.parent)

cnt = {}
for rp in raw.rglob("result.json"):
    try:
        o = json.loads(rp.read_text())
    except Exception:
        continue
    if not isinstance(o, dict):
        continue
    t = o.get("task_name")
    tr = o.get("trial_name")
    if not isinstance(t, str) or not isinstance(tr, str):
        continue
    if inc and not any(fnmatch.fnmatch(t, p) for p in inc):
        continue
    if exc and any(fnmatch.fnmatch(t, p) for p in exc):
        continue
    if not has_traj(rp, o):
        continue
    rw = reward_of(o)
    succ = (rw is not None and rw >= 1.0 and not o.get("exception_info"))
    c = cnt.setdefault(t, {"s": 0, "f": 0})
    if succ:
        c["s"] += 1
    else:
        c["f"] += 1

mixed = sorted([t for t, c in cnt.items() if c["s"] > 0 and c["f"] > 0])
out.write_text("\n".join(mixed) + ("\n" if mixed else ""), encoding="utf-8")
print(f"mixed_seed_tasks={len(mixed)}")
PY
}

compute_active() {
  python3 - "$RAW_REAL" "$OUT_REAL" "$SEED_FILE" "$TARGET_SUCCESS" "$TARGET_FAILURE" "$ACTIVE_FILE" "$STATUS_LOG" <<'PY'
import json
import sys
from pathlib import Path

raw = Path(sys.argv[1])
out = Path(sys.argv[2])
seed_file = Path(sys.argv[3])
ts = int(sys.argv[4])
tf = int(sys.argv[5])
active_file = Path(sys.argv[6])
status_log = Path(sys.argv[7])

seed = [x.strip() for x in seed_file.read_text(encoding="utf-8").splitlines() if x.strip()]
seed_set = set(seed)

def reward_of(o):
    vr = o.get("verifier_result") or {}
    rw = (vr.get("rewards") or {}).get("reward")
    if isinstance(rw, (int, float)):
        return float(rw)
    rw2 = o.get("reward")
    if isinstance(rw2, (int, float)):
        return float(rw2)
    return None

def has_traj(rp, o):
    def has_agent_evidence(td):
        agent_dir = td / "agent"
        return any((agent_dir / name).is_file() for name in (
            "trajectory.json",
            "gemini-cli.txt",
            "codex.txt",
            "claude.txt",
        ))

    tu = o.get("trial_uri")
    if isinstance(tu, str) and tu.startswith("file://"):
        td = Path(tu[7:])
        if has_agent_evidence(td):
            return True
    return has_agent_evidence(rp.parent)

def fold(root, cnt):
    if not root.exists():
        return
    for rp in root.rglob("result.json"):
        try:
            o = json.loads(rp.read_text())
        except Exception:
            continue
        if not isinstance(o, dict):
            continue
        t = o.get("task_name")
        tr = o.get("trial_name")
        if not isinstance(t, str) or not isinstance(tr, str):
            continue
        if t not in seed_set:
            continue
        if not has_traj(rp, o):
            continue
        rw = reward_of(o)
        succ = (rw is not None and rw >= 1.0 and not o.get("exception_info"))
        c = cnt.setdefault(t, {"s": 0, "f": 0})
        if succ:
            c["s"] += 1
        else:
            c["f"] += 1

cnt = {t: {"s": 0, "f": 0} for t in seed}
fold(raw, cnt)
fold(out, cnt)

active = []
lines = []
for t in seed:
    s = cnt[t]["s"]
    f = cnt[t]["f"]
    ns = max(ts - s, 0)
    nf = max(tf - f, 0)
    if ns > 0 or nf > 0:
        active.append(t)
    lines.append(f"task={t} success={s} failure={f} need_success={ns} need_failure={nf}")

active_file.write_text("\n".join(active) + ("\n" if active else ""), encoding="utf-8")
status_log.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
print(f"active_tasks={len(active)}")
PY
}

cleanup_after_round() {
  local round="$1"

  [[ "$CLEANUP_AFTER_ROUND" == "1" ]] || return 0
  [[ "$ENV_TYPE" == "docker" ]] || return 0
  command -v docker >/dev/null 2>&1 || { echo "[WARN] docker not found; skip cleanup_after_round"; return 0; }

  local round_dir="$OUT_REAL/mixed-r$(printf "%04d" "$round")"
  [[ -d "$round_dir" ]] || { echo "[WARN] round dir missing for cleanup: $round_dir"; return 0; }

  mapfile -t _projects < <(find "$round_dir" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | awk '/__/ {print tolower($0)}' | sort -u)
  echo "[INFO] cleanup round $round: projects=${#_projects[@]}"

  local p
  for p in "${_projects[@]}"; do
    docker ps -aq --filter "name=^${p}-" | xargs -r docker rm -f >/dev/null 2>&1 || true
    docker network rm "${p}_default" >/dev/null 2>&1 || true
  done

  if [[ "$CLEANUP_PRUNE" == "1" ]]; then
    docker container prune -f >/dev/null 2>&1 || true
    docker network prune -f >/dev/null 2>&1 || true
  fi

  local left
  left="$(docker network ls --format '{{.Name}}' | grep -c '__' || true)"
  echo "[INFO] cleanup round $round done. harbor_like_networks=$left"
}

run_round() {
  local round="$1"
  mapfile -t _active < "$ACTIVE_FILE"
  [[ ${#_active[@]} -eq 0 ]] && return 0

  if [[ -n "$N_TASKS" && "$N_TASKS" =~ ^[0-9]+$ ]] && (( N_TASKS < ${#_active[@]} )); then
    _active=("${_active[@]:0:N_TASKS}")
  fi

  local -a cmd=("$HARBOR_BIN" run)
  append_dataset_selector_args cmd
  cmd+=(
    -a "$AGENT"
    -m "$MODEL"
    -k "$TRIALS_PER_ROUND"
    -n "$N_CONCURRENT"
    -e "$ENV_TYPE"
    --jobs-dir "$OUT_REAL"
    --job-name "mixed-r$(printf "%04d" "$round")"
  )

  if [[ "$PROVIDER" == "google" ]]; then
    local gemini_key="${GOOGLE_API_KEY:-${GEMINI_API_KEY:-}}"
    [[ -n "$gemini_key" ]] || { echo "[ERROR] missing Gemini key at run time" >&2; exit 1; }
    cmd+=(--ak "version=${GEMINI_CLI_VERSION}")
    cmd+=(--ae "GOOGLE_API_KEY=$gemini_key")
    cmd+=(--ae "GEMINI_API_KEY=$gemini_key")
    cmd+=(--ae "GEMINI_CLI_TRUST_WORKSPACE=${GEMINI_CLI_TRUST_WORKSPACE:-true}")
  else
    cmd+=(--ae "OPENAI_API_KEY=$OPENAI_API_KEY")
    if [[ -n "$BASE_URL" ]]; then
      cmd+=(--ae "OPENAI_BASE_URL=$BASE_URL")
    fi
    if [[ -n "${CODEX_HOME:-}" ]]; then
      cmd+=(--ae "CODEX_HOME=$CODEX_HOME")
    fi
  fi

  local t
  for t in "${_active[@]}"; do
    cmd+=(--task-name "$t")
  done

  [[ "$QUIET" == "1" ]] && cmd+=(-q)
  [[ "$DEBUG" == "1" ]] && cmd+=(--debug)
  [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && cmd+=("${EXTRA_ARGS[@]}")

  echo "[INFO] Round ${round}: running ${#_active[@]} pending tasks"
  local display_cmd="${cmd[*]}"
  if [[ -n "${GOOGLE_API_KEY:-}" ]]; then
    display_cmd="${display_cmd//${GOOGLE_API_KEY}/***REDACTED***}"
  fi
  if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    display_cmd="${display_cmd//${GEMINI_API_KEY}/***REDACTED***}"
  fi
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    display_cmd="${display_cmd//${OPENAI_API_KEY}/***REDACTED***}"
  fi
  echo "[INFO] Running: ${display_cmd}"
  "${cmd[@]}"
}

{
  echo "provider=$PROVIDER"
  echo "api_key_env=$API_KEY_ENV"
  echo "raw_results_path=$RAW_REAL"
  echo "output_path=$OUT_REAL"
  echo "dataset=$DATASET"
  echo "dataset_path=${DATASET_PATH:-<registry>}"
  echo "dataset_result_slug=$DATASET_RESULT_SLUG"
  echo "dataset_results_root=$DATASET_RESULTS_ROOT"
  echo "agent=$AGENT"
  echo "model=$MODEL"
  echo "trials_per_round=$TRIALS_PER_ROUND"
  echo "max_attempt_blocks=$MAX_ATTEMPT_BLOCKS"
  echo "target_success=$TARGET_SUCCESS"
  echo "target_failure=$TARGET_FAILURE"
  echo "n_concurrent=$N_CONCURRENT"
  echo "cleanup_after_round=$CLEANUP_AFTER_ROUND"
  echo "cleanup_prune=$CLEANUP_PRUNE"
} > "$META_LOG"

echo "[INFO] RAW_RESULTS_PATH=$RAW_REAL"
echo "[INFO] OUTPUT_PATH=$OUT_REAL"
echo "[INFO] AGENT=$AGENT MODEL=$MODEL"
echo "[INFO] DATASET=$DATASET DATASET_PATH=${DATASET_PATH:-<registry>} DATASET_RESULT_SLUG=$DATASET_RESULT_SLUG"
echo "[INFO] DATASET_RESULTS_ROOT=$DATASET_RESULTS_ROOT"
echo "[COLLECT_MIXED CHECK] provider=${PROVIDER} api_key_env=${API_KEY_ENV} base_url=${BASE_URL:-<unset>} trials_per_round=${TRIALS_PER_ROUND} n_concurrent=${N_CONCURRENT} max_blocks=${MAX_ATTEMPT_BLOCKS}"
echo "[INFO] HARBOR_BIN=$HARBOR_BIN"
echo "[INFO] DOCKER_BUILDKIT=$DOCKER_BUILDKIT"
echo "[INFO] MAX_ATTEMPT_BLOCKS=$MAX_ATTEMPT_BLOCKS (each block = ${TRIALS_PER_ROUND} trials)"

build_seed
seed_n="$(wc -l < "$SEED_FILE" | tr -d '[:space:]')"
echo "[INFO] seed mixed tasks: $seed_n"
if [[ "$seed_n" == "0" ]]; then
  echo "[INFO] no mixed tasks found, stop."
  exit 0
fi

round=1
if [[ "$CONTINUE_FROM_EXISTING" == "1" ]] && compgen -G "$OUT_REAL/mixed-r[0-9][0-9][0-9][0-9]" >/dev/null; then
  last_round="$(ls -1d "$OUT_REAL"/mixed-r[0-9][0-9][0-9][0-9] | sed -E 's#.*/mixed-r([0-9]{4})#\1#' | sort | tail -n1)"
  [[ -n "$last_round" ]] && round=$((10#$last_round + 1))
fi

end_round=$((round + MAX_ATTEMPT_BLOCKS - 1))
while (( round <= end_round )); do
  echo "[INFO] ===== round $round/$end_round ====="
  compute_active
  mapfile -t active < "$ACTIVE_FILE"
  if [[ ${#active[@]} -eq 0 ]]; then
    echo "[INFO] all tasks reached success>=$TARGET_SUCCESS and failure>=$TARGET_FAILURE"
    break
  fi

  round_rc=0
  if run_round "$round"; then
    round_rc=0
  else
    round_rc=$?
  fi

  cleanup_after_round "$round"
  if (( round_rc != 0 )); then
    echo "[ERROR] round $round failed with rc=$round_rc" >&2
    exit "$round_rc"
  fi

  if [[ "$ROUND_PAUSE_SEC" =~ ^[0-9]+$ ]] && (( ROUND_PAUSE_SEC > 0 )); then
    sleep "$ROUND_PAUSE_SEC"
  fi
  ((round++))
done

if (( round > end_round )); then
  echo "[WARN] reached MAX_ATTEMPT_BLOCKS=$MAX_ATTEMPT_BLOCKS (effective end_round=$end_round)"
  compute_active
fi

echo "[INFO] done. outputs in: $OUT_REAL"
echo "[INFO] status log: $STATUS_LOG"
