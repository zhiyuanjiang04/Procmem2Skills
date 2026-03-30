#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/server/run_benchmark_smoke.sh <target>

Targets:
  mind2web               Offline Mind2Web import/distill smoke.
  alfworld               Text-only ALFWorld smoke.
  terminal-bench         Offline Terminal-Bench import/distill smoke.
  terminal-bench-harbor  Harbor package/dataset smoke.
  skillsbench-harbor     SkillsBench Harbor dry-run smoke (dataset first, path fallback).
  webarena               WebArena offline + BrowserGym package smoke.
  all                    Run all benchmark smoke checks.
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

smoke_mind2web() {
  prepare_server_env "$ROOT"
  mkdir -p .tmp/imported .tmp/skills
  local input_path="${MIND2WEB_INPUT:-}"
  if [[ -z "$input_path" ]]; then
    for candidate in \
      "$ROOT/data/mind2web/train/train_0.json" \
      "$ROOT/data/mind2web/test_task/test_task_0.json" \
      "$ROOT/examples/raw-mind2web.json"; do
      if [[ -f "$candidate" ]]; then
        input_path="$candidate"
        break
      fi
    done
  fi
  if [[ -z "$input_path" ]]; then
    echo "No Mind2Web input found. Set MIND2WEB_INPUT or place official files under $ROOT/data/mind2web/." >&2
    exit 1
  fi
  "$ROOT/.venv/bin/procmem2skills" import-benchmark mind2web "$input_path" "$ROOT/.tmp/imported/mind2web.jsonl"
  "$ROOT/.venv/bin/procmem2skills" distill-offline "$ROOT/.tmp/imported/mind2web.jsonl" "$ROOT/.tmp/skills/mind2web-smoke"
  echo "Mind2Web offline smoke completed using: $input_path"
}

smoke_alfworld() {
  prepare_server_env "$ROOT"
  mkdir -p .cache/alfworld
  export ALFWORLD_DATA="$ROOT/.cache/alfworld"
  local raw_output="$ROOT/.tmp/alfworld-smoke/raw.json"
  local imported_output="$ROOT/.tmp/alfworld-smoke/imported.jsonl"
  local skill_dir="$ROOT/.tmp/alfworld-smoke/skills"
  if [[ ! -d "$ALFWORLD_DATA/json_2.1.1" ]]; then
    "$ROOT/.venv/bin/alfworld-download" --data-dir "$ALFWORLD_DATA"
  fi
  "$ROOT/.venv/bin/python" "$ROOT/scripts/server/alfworld_collect_smoke.py" \
    --root "$ROOT" \
    --output "$raw_output" \
    --split train \
    --task-types 1 \
    --max-steps 20 \
    --num-games 1
  "$ROOT/.venv/bin/procmem2skills" import-benchmark alfworld "$raw_output" "$imported_output"
  "$ROOT/.venv/bin/procmem2skills" distill-offline "$imported_output" "$skill_dir" --min-support 1
  echo "ALFWorld smoke outputs:"
  echo "  raw:      $raw_output"
  echo "  imported: $imported_output"
  echo "  skills:   $skill_dir"
}

smoke_terminal_bench_offline() {
  prepare_server_env "$ROOT"
  local legacy_root="${TERMINAL_BENCH_LEGACY_ROOT:-/raid/zhiyuan/procmem2skills-legacy-20260315/wm2s/jobs}"
  local run_id
  run_id="$(date +%Y%m%d-%H%M%S)"
  local staging_root="$ROOT/.tmp/terminal-bench-smoke/$run_id/source"
  local imported_output="$ROOT/.tmp/terminal-bench-smoke/$run_id/imported.jsonl"
  local skill_dir="$ROOT/.tmp/terminal-bench-smoke/$run_id/skills"
  mkdir -p "$staging_root"
  local count=0
  while IFS= read -r trajectory_path; do
    local trial_dir
    local trial_name
    trial_dir="$(dirname "$(dirname "$trajectory_path")")"
    trial_name="$(basename "$trial_dir")"
    mkdir -p "$staging_root/$trial_name/agent"
    cp "$trajectory_path" "$staging_root/$trial_name/agent/trajectory.json"
    cp "$trial_dir/config.json" "$staging_root/$trial_name/config.json"
    cp "$trial_dir/result.json" "$staging_root/$trial_name/result.json"
    count=$((count + 1))
    if [[ "$count" -ge 3 ]]; then
      break
    fi
  done < <(find "$legacy_root" -path '*/agent/trajectory.json' | sort)
  if [[ "$count" -eq 0 ]]; then
    echo "No legacy Terminal-Bench trajectories found under: $legacy_root" >&2
    exit 1
  fi
  "$ROOT/.venv/bin/procmem2skills" import-benchmark terminal-bench "$staging_root" "$imported_output"
  "$ROOT/.venv/bin/procmem2skills" distill-offline "$imported_output" "$skill_dir" --min-support 1
  echo "Terminal-Bench offline smoke imported $count legacy trajectories."
  echo "  staged:   $staging_root"
  echo "  imported: $imported_output"
  echo "  skills:   $skill_dir"
}

smoke_terminal_bench_harbor() {
  prepare_server_env "$ROOT"
  local benchmark_python
  benchmark_python="$(resolve_benchmark_python "$ROOT")"
  "$(dirname "$benchmark_python")/python" --version
  "$(dirname "$benchmark_python")/harbor" datasets list | grep -i 'terminal-bench' | head -n 20
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$benchmark_python" - <<'PY'
from procmem2skills.integrations.harbor_terminal_agent import SkillAwareTerminalAgent

print(SkillAwareTerminalAgent.import_path())
PY
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$benchmark_python" \
    "$ROOT/scripts/server/run_terminal_bench_harbor_experiment.py" \
    --experiment-id tb-harbor-plan \
    --bootstrap-input "$ROOT/examples/raw-terminal-bench.json" \
    --model anthropic/claude-sonnet-4 \
    --n-tasks 1 \
    --n-concurrent 8 \
    --dry-run
}

smoke_skillsbench_harbor() {
  prepare_server_env "$ROOT"
  local benchmark_python
  benchmark_python="$(resolve_benchmark_python "$ROOT")"
  local harbor_bin
  harbor_bin="$(dirname "$benchmark_python")/harbor"
  local skillsbench_path="$ROOT/benchmarks/skillsbench/tasks"
  local dataset_name="${SKILLSBENCH_DATASET:-skillsbench}"

  local source_mode="dataset"
  if ! "$harbor_bin" datasets list | grep -qi 'skillsbench'; then
    source_mode="path"
  fi

  if [[ "$source_mode" == "path" && ! -d "$skillsbench_path" ]]; then
    echo "SkillsBench dataset not found in Harbor registry, and local path missing: $skillsbench_path" >&2
    echo "Run: bash scripts/server/setup_benchmark.sh skillsbench" >&2
    exit 1
  fi

  local cmd=(
    "$benchmark_python"
    "$ROOT/scripts/server/run_skillsbench_harbor_experiment.py"
    --experiment-id sb-harbor-plan
    --model openai/gpt-5.3-codex
    --n-tasks 1
    --n-concurrent 8
    --dry-run
  )

  if [[ "$source_mode" == "dataset" ]]; then
    cmd+=(--source-mode dataset --dataset "$dataset_name")
  else
    cmd+=(--source-mode path --skillsbench-path "$skillsbench_path")
  fi

  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "${cmd[@]}"
}

smoke_webarena_offline() {
  prepare_server_env "$ROOT"
  mkdir -p .tmp/imported .tmp/skills
  "$ROOT/.venv/bin/procmem2skills" import-benchmark webarena "$ROOT/examples/raw-webarena.json" "$ROOT/.tmp/imported/webarena.jsonl"
  "$ROOT/.venv/bin/procmem2skills" distill-offline "$ROOT/.tmp/imported/webarena.jsonl" "$ROOT/.tmp/skills/webarena-smoke"
  echo "WebArena offline smoke completed."
}

smoke_webarena_package() {
  prepare_server_env "$ROOT"
  mkdir -p .playwright
  export PLAYWRIGHT_BROWSERS_PATH="$ROOT/.playwright"
  local benchmark_python
  benchmark_python="$(resolve_benchmark_python "$ROOT")"
  "$benchmark_python" - <<'PY'
import gymnasium as gym
import browsergym.webarena  # noqa: F401

ids = sorted(env_id for env_id in gym.envs.registry.keys() if "webarena" in env_id.lower())
print(f"registered_webarena_envs={len(ids)}")
print("\n".join(ids[:20]))

env = gym.make("browsergym/webarena.0")
try:
    env.reset()
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
finally:
    env.close()
PY
}

require_arg "$@"
target="$1"
case "$target" in
  mind2web)
    run_step "Mind2Web smoke" smoke_mind2web
    ;;
  alfworld)
    run_step "ALFWorld smoke" smoke_alfworld
    ;;
  terminal-bench)
    run_step "Terminal-Bench offline smoke" smoke_terminal_bench_offline
    ;;
  terminal-bench-harbor)
    run_step "Terminal-Bench Harbor smoke" smoke_terminal_bench_harbor
    ;;
  skillsbench-harbor)
    run_step "SkillsBench Harbor smoke" smoke_skillsbench_harbor
    ;;
  webarena)
    run_step "WebArena offline smoke" smoke_webarena_offline
    run_step "WebArena package smoke" smoke_webarena_package
    ;;
  all)
    run_step "Mind2Web smoke" smoke_mind2web
    run_step "ALFWorld smoke" smoke_alfworld
    run_step "Terminal-Bench offline smoke" smoke_terminal_bench_offline
    run_step "Terminal-Bench Harbor smoke" smoke_terminal_bench_harbor
    run_step "SkillsBench Harbor smoke" smoke_skillsbench_harbor
    run_step "WebArena offline smoke" smoke_webarena_offline
    run_step "WebArena package smoke" smoke_webarena_package
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
