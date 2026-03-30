#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/common.sh"

prepare_common_env() {
  prepare_server_env "$ROOT"
  mkdir -p "$ROOT/.uv-python" "$ROOT/.uv-tool" "$ROOT/.pip-cache"
  export UV_PYTHON_INSTALL_DIR="$ROOT/.uv-python"
}

ensure_uv_tool() {
  local python_bin="${PYTHON_BIN:-python3}"
  if [[ ! -x "$ROOT/.uv-tool/bin/uv" ]]; then
    TMPDIR="$ROOT/.tmp" PIP_CACHE_DIR="$ROOT/.pip-cache" "$python_bin" -m pip install --no-cache-dir --target "$ROOT/.uv-tool" uv
  fi
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/server/setup_benchmark.sh <target>

Targets:
  core             Prepare the project-local uv envs (.venv and .venv-py312).
  mind2web         Prepare the offline Mind2Web layout.
  alfworld         Install text-only ALFWorld into .venv.
  terminal-bench   Install Harbor into .venv-py312.
  skillsbench      Clone/update skillsbench benchmark repo under benchmarks/skillsbench.
  webarena         Install BrowserGym/WebArena into .venv-py312.
  all              Run core + all benchmark-specific setup steps.
EOF
}

require_arg() {
  if [[ $# -lt 1 ]]; then
    usage >&2
    exit 1
  fi
}

run_step() {
  local title="$1"
  shift
  echo
  echo "==> $title"
  "$@"
}

setup_core() {
  prepare_common_env
  ensure_uv_tool
  local python_bin="${PYTHON_BIN:-python3}"
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    "$ROOT/.uv-tool/bin/uv" venv "$ROOT/.venv" --python "$python_bin"
  fi
  "$ROOT/.uv-tool/bin/uv" pip install -e "$ROOT" --python "$ROOT/.venv/bin/python"
  "$ROOT/.uv-tool/bin/uv" python install 3.12
  if [[ ! -x "$ROOT/.venv-py312/bin/python" ]]; then
    "$ROOT/.uv-tool/bin/uv" venv "$ROOT/.venv-py312" --python 3.12
  fi
  cat <<EOF
Local env setup completed:
  - $ROOT/.venv
  - $ROOT/.venv-py312

Use Python 3.12 for Harbor/WebArena when needed:
  export PROCMEM_BENCHMARK_PYTHON="$ROOT/.venv-py312/bin/python"
EOF
}

setup_mind2web() {
  prepare_common_env
  mkdir -p data/mind2web/train data/mind2web/test_task data/mind2web/test_website data/mind2web/test_domain
  cat <<EOF
Mind2Web directories are ready:
  $ROOT/data/mind2web/train
  $ROOT/data/mind2web/test_task
  $ROOT/data/mind2web/test_website
  $ROOT/data/mind2web/test_domain
EOF
}

setup_alfworld() {
  prepare_common_env
  ensure_uv_tool
  mkdir -p .cache/alfworld
  export ALFWORLD_DATA="$ROOT/.cache/alfworld"
  "$ROOT/.uv-tool/bin/uv" pip install "alfworld" --python "$ROOT/.venv/bin/python"
  cat <<EOF
Installed text-only ALFWorld in:
  $ROOT/.venv

Environment:
  export ALFWORLD_DATA="$ALFWORLD_DATA"
EOF
}

setup_terminal_bench() {
  prepare_common_env
  ensure_uv_tool
  mkdir -p benchmarks
  local benchmark_python
  benchmark_python="$(resolve_benchmark_python "$ROOT")"
  "$ROOT/.uv-tool/bin/uv" pip install harbor --python "$benchmark_python"
  cat <<EOF
Installed Harbor using:
  $benchmark_python

Docker is required for live Terminal-Bench and SkillsBench execution.
EOF
}

setup_skillsbench() {
  prepare_common_env
  mkdir -p benchmarks
  local benchmark_dir="$ROOT/benchmarks/skillsbench"
  local repo_url="${SKILLSBENCH_REPO_URL:-https://github.com/benchflow-ai/skillsbench.git}"

  if [[ -d "$benchmark_dir/.git" ]]; then
    git -C "$benchmark_dir" pull --ff-only
    echo "Updated existing SkillsBench repo: $benchmark_dir"
  elif [[ -d "$benchmark_dir" ]]; then
    cat <<EOF
Found existing directory without git metadata:
  $benchmark_dir

Skipping clone. You can pass --skillsbench-path explicitly when running experiment scripts.
EOF
  else
    git clone --depth 1 "$repo_url" "$benchmark_dir"
    echo "Cloned SkillsBench repo: $benchmark_dir"
  fi

  cat <<EOF
SkillsBench expected task path:
  $benchmark_dir/tasks
EOF
}

setup_webarena() {
  prepare_common_env
  ensure_uv_tool
  mkdir -p .playwright
  export PLAYWRIGHT_BROWSERS_PATH="$ROOT/.playwright"
  local benchmark_python
  benchmark_python="$(resolve_benchmark_python "$ROOT")"
  "$ROOT/.uv-tool/bin/uv" pip install browsergym browsergym-webarena playwright --python "$benchmark_python"
  "$(dirname "$benchmark_python")/playwright" install chromium
  cat <<EOF
Installed BrowserGym/WebArena using:
  $benchmark_python

Chromium path:
  $PLAYWRIGHT_BROWSERS_PATH
EOF
}

require_arg "$@"
target="$1"
case "$target" in
  core)
    run_step "Core uv environments" setup_core
    ;;
  mind2web)
    run_step "Core uv environments" setup_core
    run_step "Mind2Web directories" setup_mind2web
    ;;
  alfworld)
    run_step "Core uv environments" setup_core
    run_step "ALFWorld setup" setup_alfworld
    ;;
  terminal-bench)
    run_step "Core uv environments" setup_core
    run_step "Terminal-Bench setup" setup_terminal_bench
    ;;
  skillsbench)
    run_step "Core uv environments" setup_core
    run_step "Terminal-Bench setup" setup_terminal_bench
    run_step "SkillsBench repo setup" setup_skillsbench
    ;;
  webarena)
    run_step "Core uv environments" setup_core
    run_step "WebArena setup" setup_webarena
    ;;
  all)
    run_step "Core uv environments" setup_core
    run_step "Mind2Web directories" setup_mind2web
    run_step "ALFWorld setup" setup_alfworld
    run_step "Terminal-Bench setup" setup_terminal_bench
    run_step "SkillsBench repo setup" setup_skillsbench
    run_step "WebArena setup" setup_webarena
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown target: $target" >&2
    usage >&2
    exit 1
    ;;
esac
