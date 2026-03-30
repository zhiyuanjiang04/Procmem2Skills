from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from procmem2skills.analysis.failure import build_failure_analysis_from_trajectories
from procmem2skills.analysis.taxonomy import build_taxonomy_report
from procmem2skills.evaluation.pipeline import SkillDistillationPipeline
from procmem2skills.importers import import_alfworld, import_mind2web, import_terminal_bench, import_webarena
from procmem2skills.packager.materialize import materialize_skill_repository_standard_llm
from procmem2skills.recorder.jsonl import write_trajectories
from procmem2skills.runtime.update import OnlineSkillUpdater, OnlineUpdateConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a unified formal procmem2skills experiment for one benchmark.")
    parser.add_argument("--benchmark", required=True, choices=["mind2web", "webarena", "alfworld", "terminal-bench"])
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--input-path", type=Path, help="Raw trajectory input path, result directory, or benchmark corpus root.")
    parser.add_argument("--memory-mode", default="offline", choices=["offline", "online"])
    parser.add_argument("--agent", default="agent")
    parser.add_argument("--harness", default=None)
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--similarity-threshold", type=float, default=0.34)
    parser.add_argument("--structure-threshold", type=float, default=0.5)
    parser.add_argument("--cluster-backend", default="auto", choices=["auto", "lexical", "embedding-dbscan"])
    parser.add_argument("--cluster-embedding-model")
    parser.add_argument("--cluster-embedding-base-url")
    parser.add_argument("--cluster-dbscan-eps", type=float, default=0.35)
    parser.add_argument("--cluster-dbscan-min-samples", type=int, default=2)
    parser.add_argument("--cluster-embedding-strict", action="store_true")
    parser.add_argument(
        "--workflow-aggregation-mode",
        default="global",
        choices=["global", "per-task", "global-dbscan-qwen"],
    )
    parser.add_argument(
        "--no-per-task-skill-namespace",
        action="store_true",
        help="Disable task prefix for skill IDs in per-task aggregation mode.",
    )
    parser.add_argument("--skill-creator-model")
    parser.add_argument("--skill-creator-base-url")
    parser.add_argument(
        "--skill-creator-agent-style",
        default="codex",
        choices=["codex", "claude-code", "cc", "opencode"],
    )
    parser.add_argument("--skill-creator-system-prompt")
    parser.add_argument("--skip-taxonomy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--collect-live", action="store_true", help="Only supported for ALFWorld.")
    parser.add_argument("--alfworld-split", default="train", choices=["train", "eval_in_distribution", "eval_out_of_distribution"])
    parser.add_argument("--alfworld-task-types", default="1")
    parser.add_argument("--alfworld-max-steps", type=int, default=40)
    parser.add_argument("--alfworld-num-games", type=int, default=1)
    return parser.parse_args()


def normalize_experiment_name(value: str, *, max_length: int = 56) -> str:
    tokens = re.findall(r"[a-z0-9]+", (value or "").strip().lower())
    if not tokens:
        return "experiment"
    slug = "-".join(tokens)
    if len(slug) <= max_length:
        return slug
    trimmed = slug[:max_length].rstrip("-")
    return trimmed or "experiment"


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    requested_experiment_id = args.experiment_id
    normalized_experiment_id = normalize_experiment_name(requested_experiment_id, max_length=56)
    run_dir = (project_root / args.output_root / args.benchmark / normalized_experiment_id).resolve()
    raw_dir = run_dir / "raw"
    imported_dir = run_dir / "imported"
    analysis_dir = run_dir / "analysis"
    memory_dir = run_dir / "memory"
    skill_dir = run_dir / "skills"
    imported_path = imported_dir / "trajectories.jsonl"
    taxonomy_path = analysis_dir / "taxonomy.json"
    archive_path = memory_dir / "archive.jsonl"
    manifest_path = run_dir / "manifest.json"
    runbook_path = run_dir / "RUNBOOK.md"

    resolved_input_path = _resolve_input_path(args, project_root, raw_dir)
    plan = {
        "benchmark": args.benchmark,
        "experiment_id": normalized_experiment_id,
        "requested_experiment_id": requested_experiment_id,
        "memory_mode": args.memory_mode,
        "input_path": str(resolved_input_path) if resolved_input_path else None,
        "run_dir": str(run_dir),
        "outputs": {
            "imported_path": str(imported_path),
            "skill_dir": str(skill_dir),
            "archive_path": str(archive_path),
            "taxonomy_path": str(taxonomy_path),
            "manifest_path": str(manifest_path),
            "runbook_path": str(runbook_path),
        },
    }

    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    imported_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)

    trajectories = _import_trajectories(args, resolved_input_path)
    if not trajectories:
        raise RuntimeError(f"no trajectories imported for benchmark={args.benchmark} from {resolved_input_path}")
    write_trajectories(imported_path, trajectories)

    skill_count, cluster_count, generation_meta = _materialize_memory(
        args=args,
        trajectories=trajectories,
        skill_dir=skill_dir,
        archive_path=archive_path,
    )

    if args.skip_taxonomy:
        taxonomy_count = None
    else:
        report = build_taxonomy_report(trajectories)
        taxonomy_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        taxonomy_count = report.total_tasks

    manifest = {
        **plan,
        "trajectory_count": len(trajectories),
        "skill_count": skill_count,
        "workflow_cluster_count": cluster_count,
        "taxonomy_task_count": taxonomy_count,
        "skill_generation": generation_meta,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    runbook_path.write_text(_render_runbook(args, manifest), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def _resolve_input_path(args: argparse.Namespace, project_root: Path, raw_dir: Path) -> Path | None:
    if args.collect_live:
        if args.benchmark != "alfworld":
            raise ValueError("--collect-live is only supported for benchmark=alfworld")
        output_path = raw_dir / "alfworld-live.json"
        _collect_alfworld_live(args, project_root, output_path)
        return output_path

    if args.input_path is None:
        raise ValueError("--input-path is required unless benchmark=alfworld with --collect-live")
    return args.input_path.resolve()


def _collect_alfworld_live(args: argparse.Namespace, project_root: Path, output_path: Path) -> None:
    python_bin = project_root / ".venv" / "bin" / "python"
    collector = project_root / "scripts" / "server" / "alfworld_collect_smoke.py"
    command = [
        str(python_bin),
        str(collector),
        "--root",
        str(project_root),
        "--output",
        str(output_path),
        "--split",
        args.alfworld_split,
        "--task-types",
        args.alfworld_task_types,
        "--max-steps",
        str(args.alfworld_max_steps),
        "--num-games",
        str(args.alfworld_num_games),
    ]
    subprocess.run(command, check=True)


def _import_trajectories(args: argparse.Namespace, input_path: Path) -> list:
    harness = args.harness
    if args.benchmark == "mind2web":
        trajectories = []
        for path in _recursive_json_inputs(input_path):
            trajectories.extend(import_mind2web(path, agent=args.agent, harness=harness or "mind2web/replay"))
        return trajectories
    if args.benchmark == "webarena":
        if input_path.is_dir():
            browsergym_result = import_webarena(input_path, agent=args.agent, harness=harness or "browsergym/webarena")
            if browsergym_result:
                return browsergym_result
            trajectories = []
            for path in _recursive_json_inputs(input_path):
                trajectories.extend(import_webarena(path, agent=args.agent, harness=harness or "browsergym/webarena"))
            return trajectories
        return import_webarena(input_path, agent=args.agent, harness=harness or "browsergym/webarena")
    if args.benchmark == "alfworld":
        trajectories = []
        for path in _recursive_json_inputs(input_path):
            trajectories.extend(import_alfworld(path, agent=args.agent, harness=harness or "alfworld/textworld"))
        return trajectories
    if args.benchmark == "terminal-bench":
        return import_terminal_bench(input_path, agent=args.agent, harness=harness or "terminal-bench/harness")
    raise ValueError(f"unsupported benchmark: {args.benchmark}")


def _recursive_json_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    candidates = sorted(child for child in path.rglob("*") if child.is_file() and child.suffix in {".json", ".jsonl"})
    return candidates


def _materialize_memory(
    *,
    args: argparse.Namespace,
    trajectories: list,
    skill_dir: Path,
    archive_path: Path,
) -> tuple[int, int | None, dict]:
    if args.memory_mode == "offline":
        failure_analysis = build_failure_analysis_from_trajectories(trajectories)
        result = SkillDistillationPipeline(
            min_support=args.min_support,
            similarity_threshold=args.similarity_threshold,
            structure_threshold=args.structure_threshold,
            cluster_backend=args.cluster_backend,
            cluster_embedding_model=args.cluster_embedding_model,
            cluster_embedding_base_url=args.cluster_embedding_base_url,
            cluster_dbscan_eps=args.cluster_dbscan_eps,
            cluster_dbscan_min_samples=args.cluster_dbscan_min_samples,
            cluster_embedding_strict=args.cluster_embedding_strict,
            workflow_aggregation_mode=args.workflow_aggregation_mode,
            per_task_skill_namespace=not args.no_per_task_skill_namespace,
        ).distill(trajectories)
        _, generation_meta = materialize_skill_repository_standard_llm(
            skills=result.skills,
            output_dir=skill_dir,
            model=args.skill_creator_model,
            base_url=args.skill_creator_base_url,
            skill_creator_agent_style=args.skill_creator_agent_style,
            skill_creator_system_prompt=args.skill_creator_system_prompt,
            failure_analysis=failure_analysis.get("global"),
            failure_analysis_by_task=failure_analysis.get("by_task"),
        )
        generation_meta["workflow_cluster_count"] = len(result.clusters)
        generation_meta["failure_analysis"] = failure_analysis.get("global")
        return len(result.skills), len(result.clusters), generation_meta

    updater = OnlineSkillUpdater(
        OnlineUpdateConfig(
            archive_path=archive_path,
            repository_dir=skill_dir,
            min_support=args.min_support,
            min_score=args.min_score,
            similarity_threshold=args.similarity_threshold,
            structure_threshold=args.structure_threshold,
            cluster_backend=args.cluster_backend,
            cluster_embedding_model=args.cluster_embedding_model,
            cluster_embedding_base_url=args.cluster_embedding_base_url,
            cluster_dbscan_eps=args.cluster_dbscan_eps,
            cluster_dbscan_min_samples=args.cluster_dbscan_min_samples,
            cluster_embedding_strict=args.cluster_embedding_strict,
            workflow_aggregation_mode=args.workflow_aggregation_mode,
            per_task_skill_namespace=not args.no_per_task_skill_namespace,
            skill_creator_model=args.skill_creator_model,
            skill_creator_base_url=args.skill_creator_base_url,
            skill_creator_agent_style=args.skill_creator_agent_style,
            skill_creator_system_prompt=args.skill_creator_system_prompt,
        )
    )
    written = []
    for trajectory in trajectories:
        written = updater.ingest(trajectory)
    skill_count = len([path for path in skill_dir.iterdir() if path.is_dir()]) if skill_dir.exists() else len(written)
    generation_meta = {
        "requested_mode": "llm-agent",
        "effective_mode": "llm-agent",
        "llm_model": args.skill_creator_model,
        "written_skill_dirs": skill_count,
        "standardized_llm_flow": "llm-agent-strict",
    }
    return skill_count, None, generation_meta


def _render_runbook(args: argparse.Namespace, manifest: dict) -> str:
    command = " ".join(
        shlex.quote(part)
        for part in [
            "bash",
            "scripts/server/run_formal_experiment.sh",
            "--benchmark",
            args.benchmark,
            "--experiment-id",
            str(manifest.get("requested_experiment_id") or manifest["experiment_id"]),
            "--memory-mode",
            args.memory_mode,
            *([] if args.input_path is None else ["--input-path", str(args.input_path)]),
            *([] if not args.collect_live else ["--collect-live"]),
        ]
    )
    lines = [
        "# Formal Experiment Runbook",
        "",
        f"- Benchmark: {manifest['benchmark']}",
        f"- Experiment ID: {manifest['experiment_id']}",
        f"- Memory Mode: {manifest['memory_mode']}",
        f"- Input Path: {manifest['input_path']}",
        f"- Run Dir: {manifest['run_dir']}",
        f"- Imported Trajectories: {manifest['outputs']['imported_path']}",
        f"- Skill Repository: {manifest['outputs']['skill_dir']}",
        "",
        "## Re-run Command",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Notes",
        "",
        "- `offline` means batch distillation from imported trajectories.",
        "- `online` means incremental `update-online` style memory growth over the same imported trajectory stream.",
        "- For WebArena and Terminal-Bench live collection, follow the benchmark-specific instructions in `docs/formal-experiment-interface.md` first, then point `--input-path` to the recorded output directory.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - entrypoint guard
        print(f"error: {exc}", file=sys.stderr)
        raise
