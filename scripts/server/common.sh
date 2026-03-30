#!/usr/bin/env bash

prepare_server_env() {
  local root="$1"
  mkdir -p "$root/.tmp" "$root/.uv-cache" "$root/.cache/xdg"
  export TMPDIR="$root/.tmp"
  export UV_CACHE_DIR="$root/.uv-cache"
  export XDG_CACHE_HOME="$root/.cache/xdg"
  export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
}

resolve_benchmark_python() {
  local root="$1"
  local benchmark_python="${PROCMEM_BENCHMARK_PYTHON:-}"
  if [[ -z "$benchmark_python" && -x "$root/.venv-py312/bin/python" ]]; then
    benchmark_python="$root/.venv-py312/bin/python"
  fi
  if [[ -z "$benchmark_python" ]]; then
    benchmark_python="$root/.venv/bin/python"
  fi
  printf "%s\n" "$benchmark_python"
}

normalize_slug() {
  local raw="$1"
  local lowered
  lowered="$(echo "$raw" | tr '[:upper:]' '[:lower:]')"
  local normalized
  normalized="$(echo "$lowered" | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-{2,}/-/g')"
  if [[ -z "$normalized" ]]; then
    normalized="experiment"
  fi
  echo "$normalized"
}

truncate_slug() {
  local value="$1"
  local max_len="$2"
  if [[ ${#value} -le $max_len ]]; then
    echo "$value"
    return
  fi
  local trimmed="${value:0:$max_len}"
  trimmed="$(echo "$trimmed" | sed -E 's/-+$//')"
  if [[ -z "$trimmed" ]]; then
    trimmed="experiment"
  fi
  echo "$trimmed"
}
