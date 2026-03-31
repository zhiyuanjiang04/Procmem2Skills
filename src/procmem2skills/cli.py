from __future__ import annotations

import json
from pathlib import Path

try:
    import typer
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    class _FallbackTyperApp:
        def __init__(self, **_kwargs) -> None:
            pass

        def command(self, *_args, **_kwargs):
            def _decorator(func):
                return func

            return _decorator

        def __call__(self, *_args, **_kwargs) -> None:
            print("Usage: procmem2skills [COMMAND] [OPTIONS]")

    class _FallbackTyper:
        BadParameter = ValueError

        @staticmethod
        def Typer(**kwargs):
            return _FallbackTyperApp(**kwargs)

        @staticmethod
        def Argument(default=None, **_kwargs):
            return default

        @staticmethod
        def Option(default=None, **_kwargs):
            return default

        @staticmethod
        def echo(message: str) -> None:
            print(message)

    typer = _FallbackTyper()

from procmem2skills.adapters import BENCHMARK_PROFILES
from procmem2skills.adapters.mock import MockTerminalAdapter
from procmem2skills.analysis.failure import build_failure_analysis_from_trajectories
from procmem2skills.analysis.taxonomy import build_taxonomy_report
from procmem2skills.evaluation.pipeline import SkillDistillationPipeline
from procmem2skills.evaluation.policies import SkillFirstTerminalPolicy
from procmem2skills.evaluation.replay_transfer import evaluate_replay_transfer
from procmem2skills.evaluation.runner import LiveRunner
from procmem2skills.importers import import_alfworld, import_mind2web, import_terminal_bench, import_webarena
from procmem2skills.packager.materialize import materialize_skill_repository_standard_llm
from procmem2skills.packager.skill_writer import SkillWriter
from procmem2skills.research.skill_failure_study import (
    InjectionStrategy,
    RetrievalMethod,
    SelfGeneratedMode,
    SkillFailureStudyConfig,
    SplitMode,
    analyze_stored_traces,
    run_skill_failure_study,
)
from procmem2skills.recorder.jsonl import load_trajectories, write_trajectories
from procmem2skills.runtime.update import OnlineSkillUpdater, OnlineUpdateConfig

app = typer.Typer(no_args_is_help=True, help="Trajectory-to-skill pipeline for cross-harness agent memory research.")


@app.command("list-benchmarks")
def list_benchmarks() -> None:
    for profile in BENCHMARK_PROFILES:
        typer.echo(
            f"{profile.benchmark.value}\t{profile.mode.value}\t{profile.harness}\t{profile.evaluation_style}"
        )


@app.command("distill-offline")
def distill_offline(
    trajectory_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    skill_repository: Path = typer.Argument(...),
    min_support: int = typer.Option(1, min=1),
    similarity_threshold: float = typer.Option(0.34),
    structure_threshold: float = typer.Option(0.5),
    cluster_backend: str = typer.Option("auto", help="auto | lexical | embedding-dbscan"),
    cluster_embedding_model: str | None = typer.Option(None),
    cluster_embedding_base_url: str | None = typer.Option(None),
    cluster_dbscan_eps: float = typer.Option(0.35),
    cluster_dbscan_min_samples: int = typer.Option(2, min=1),
    cluster_embedding_strict: bool = typer.Option(False),
    workflow_aggregation_mode: str = typer.Option("global", help="global | per-task | global-dbscan-qwen"),
    per_task_skill_namespace: bool = typer.Option(True, help="Namespace skill ids in per-task mode."),
    skill_creator_model: str | None = typer.Option(None),
    skill_creator_base_url: str | None = typer.Option(None),
    skill_creator_agent_style: str = typer.Option("codex", help="codex | claude-code | opencode"),
    skill_creator_system_prompt: str | None = typer.Option(None),
) -> None:
    trajectories = load_trajectories(trajectory_path)
    result = SkillDistillationPipeline(
        min_support=min_support,
        similarity_threshold=similarity_threshold,
        structure_threshold=structure_threshold,
        cluster_backend=cluster_backend,
        cluster_embedding_model=cluster_embedding_model,
        cluster_embedding_base_url=cluster_embedding_base_url,
        cluster_dbscan_eps=cluster_dbscan_eps,
        cluster_dbscan_min_samples=cluster_dbscan_min_samples,
        cluster_embedding_strict=cluster_embedding_strict,
        workflow_aggregation_mode=workflow_aggregation_mode,
        per_task_skill_namespace=per_task_skill_namespace,
    ).distill(trajectories)
    failure_analysis = build_failure_analysis_from_trajectories(trajectories)
    written, generation_meta = materialize_skill_repository_standard_llm(
        skills=result.skills,
        output_dir=skill_repository,
        writer=SkillWriter(),
        model=skill_creator_model,
        base_url=skill_creator_base_url,
        skill_creator_agent_style=skill_creator_agent_style,
        skill_creator_system_prompt=skill_creator_system_prompt,
        failure_analysis=failure_analysis.get("global"),
        failure_analysis_by_task=failure_analysis.get("by_task"),
    )
    typer.echo(
        f"distilled {len(result.skills)} skills from {len(result.trajectories)} trajectories into {skill_repository}"
    )
    typer.echo(f"formed {len(result.clusters)} workflow clusters and wrote {len(written)} skill directories")
    typer.echo(f"skill generation mode={generation_meta['effective_mode']} llm_generated={generation_meta['llm_generated']}")


@app.command("update-online")
def update_online(
    trajectory_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    archive_path: Path = typer.Argument(...),
    skill_repository: Path = typer.Argument(...),
    min_support: int = typer.Option(1, min=1),
    min_score: float = typer.Option(0.0),
    similarity_threshold: float = typer.Option(0.34),
    structure_threshold: float = typer.Option(0.5),
    cluster_backend: str = typer.Option("auto", help="auto | lexical | embedding-dbscan"),
    cluster_embedding_model: str | None = typer.Option(None),
    cluster_embedding_base_url: str | None = typer.Option(None),
    cluster_dbscan_eps: float = typer.Option(0.35),
    cluster_dbscan_min_samples: int = typer.Option(2, min=1),
    cluster_embedding_strict: bool = typer.Option(False),
    workflow_aggregation_mode: str = typer.Option("global", help="global | per-task | global-dbscan-qwen"),
    per_task_skill_namespace: bool = typer.Option(True, help="Namespace skill ids in per-task mode."),
    skill_creator_model: str | None = typer.Option(None),
    skill_creator_base_url: str | None = typer.Option(None),
    skill_creator_agent_style: str = typer.Option("codex", help="codex | claude-code | opencode"),
    skill_creator_system_prompt: str | None = typer.Option(None),
) -> None:
    updater = OnlineSkillUpdater(
        OnlineUpdateConfig(
            archive_path=archive_path,
            repository_dir=skill_repository,
            min_support=min_support,
            min_score=min_score,
            similarity_threshold=similarity_threshold,
            structure_threshold=structure_threshold,
            cluster_backend=cluster_backend,
            cluster_embedding_model=cluster_embedding_model,
            cluster_embedding_base_url=cluster_embedding_base_url,
            cluster_dbscan_eps=cluster_dbscan_eps,
            cluster_dbscan_min_samples=cluster_dbscan_min_samples,
            cluster_embedding_strict=cluster_embedding_strict,
            workflow_aggregation_mode=workflow_aggregation_mode,
            per_task_skill_namespace=per_task_skill_namespace,
            skill_creator_model=skill_creator_model,
            skill_creator_base_url=skill_creator_base_url,
            skill_creator_agent_style=skill_creator_agent_style,
            skill_creator_system_prompt=skill_creator_system_prompt,
        )
    )
    trajectories = load_trajectories(trajectory_path)
    written = []
    for trajectory in trajectories:
        written = updater.ingest(trajectory)
    typer.echo(
        f"processed {len(trajectories)} trajectories and materialized {len(written)} skills in {skill_repository}"
    )


@app.command("import-benchmark")
def import_benchmark(
    benchmark: str = typer.Argument(..., help="One of: webarena, mind2web, alfworld, terminal-bench"),
    input_path: Path = typer.Argument(..., exists=True),
    output_path: Path = typer.Argument(...),
    agent: str = typer.Option("agent"),
    harness: str | None = typer.Option(None),
) -> None:
    normalized = benchmark.strip().lower()
    if normalized == "webarena":
        trajectories = import_webarena(input_path, agent=agent, harness=harness or "browsergym/webarena")
    elif normalized == "mind2web":
        trajectories = import_mind2web(input_path, agent=agent, harness=harness or "mind2web/replay")
    elif normalized == "alfworld":
        trajectories = import_alfworld(input_path, agent=agent, harness=harness or "alfworld/textworld")
    elif normalized == "terminal-bench":
        trajectories = import_terminal_bench(input_path, agent=agent, harness=harness or "terminal-bench/harness")
    else:
        raise typer.BadParameter(f"unsupported benchmark: {benchmark}")
    write_trajectories(output_path, trajectories)
    typer.echo(f"imported {len(trajectories)} trajectories from {benchmark} into {output_path}")


@app.command("run-mock-live")
def run_mock_live(
    skill_repository: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    output_path: Path = typer.Argument(...),
    top_k_skills: int = typer.Option(3, min=1),
    max_steps: int = typer.Option(3, min=1),
) -> None:
    runner = LiveRunner(skill_repository=skill_repository, top_k_skills=top_k_skills, max_steps=max_steps)
    result = runner.run(MockTerminalAdapter(), SkillFirstTerminalPolicy(), episode_id="mock-live-episode")
    write_trajectories(output_path, [result.trajectory])
    typer.echo(
        f"completed mock live run with {len(result.trajectory.events)} events and wrote trajectory to {output_path}"
    )


@app.command("summarize-taxonomy")
def summarize_taxonomy(
    trajectory_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    output_path: Path | None = typer.Argument(None),
) -> None:
    trajectories = load_trajectories(trajectory_path)
    report = build_taxonomy_report(trajectories)
    if output_path is None:
        typer.echo(report.model_dump_json(indent=2))
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"summarized taxonomy for {len(trajectories)} tasks into {output_path}")


@app.command("evaluate-replay-transfer")
def evaluate_replay_transfer_cmd(
    skill_repository: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    reference_trajectory_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    target_trajectory_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    output_path: Path | None = typer.Argument(None),
    top_k: int = typer.Option(5, min=1),
) -> None:
    references = load_trajectories(reference_trajectory_path)
    targets = load_trajectories(target_trajectory_path)
    if not references:
        raise typer.BadParameter(f"no trajectories found in {reference_trajectory_path}")
    if not targets:
        raise typer.BadParameter(f"no trajectories found in {target_trajectory_path}")
    report = evaluate_replay_transfer(
        skill_repository=skill_repository,
        reference=references[0],
        target=targets[0],
        top_k=top_k,
    )
    if output_path is None:
        typer.echo(report.model_dump_json(indent=2))
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"wrote replay transfer report to {output_path}")


@app.command("run-skill-failure-study")
def run_skill_failure_study_cmd(
    trajectory_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    skill_repository: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    output_path: Path = typer.Argument(...),
    retrieval_methods: str = typer.Option(
        "page-index,qwen3-embedding",
        help="Comma-separated: page-index, context-injection, embedding-based, qwen3-embedding",
    ),
    injection_strategies: str = typer.Option(
        "no-skill,direct-inline,claude-style-progressive",
        help="Comma-separated: no-skill, direct-inline, claude-style-progressive",
    ),
    pool_sizes: str = typer.Option("50,500,5000", help="Comma-separated pool sizes"),
    self_generated_modes: str = typer.Option(
        "all-procedural-memories,success-only-memories,skills-plus-procedural-memory",
        help="Comma-separated self memory settings",
    ),
    split_modes: str = typer.Option(
        "in-task,cross-task-holdout",
        help="Comma-separated split modes",
    ),
    benchmarks: str = typer.Option("", help="Optional comma-separated benchmark filter"),
    agents: str = typer.Option("", help="Optional comma-separated agent filter"),
    terminal_bench_datasets: str = typer.Option(
        "",
        help="Optional comma-separated terminal-bench dataset filter, e.g. terminal-bench@2.0,terminal-bench-sample@2.0",
    ),
    terminal_bench_parameter_keys: str = typer.Option(
        "",
        help="Optional comma-separated terminal-bench parameter keys included in study query/context.",
    ),
    top_k: int = typer.Option(5, min=1),
    cross_task_holdout_ratio: float = typer.Option(0.3, min=0.0, max=0.9),
    random_seed: int = typer.Option(0),
    qwen3_embedding_model: str | None = typer.Option(None, help="Qwen3 embedding model name."),
    qwen3_embedding_local_base_url: str | None = typer.Option(
        None,
        help="Local embeddings endpoint (preferred), e.g. http://127.0.0.1:8000/v1",
    ),
    qwen3_embedding_base_url: str | None = typer.Option(
        None,
        help="Remote embeddings endpoint fallback.",
    ),
    qwen3_embedding_api_key: str | None = typer.Option(None, help="Optional embeddings API key."),
    qwen3_embedding_timeout_sec: int = typer.Option(30, min=1),
    qwen3_embedding_batch_size: int = typer.Option(16, min=1),
    qwen3_embedding_strict: bool = typer.Option(
        False,
        help="Strict mode: fail immediately when qwen3 embedding retrieval is unavailable.",
    ),
    procedural_dbscan_eps: float = typer.Option(0.35),
    procedural_dbscan_min_samples: int = typer.Option(2, min=1),
    batch_rollout_required: bool = typer.Option(True, help="Require multi-task batch rollout before distillation."),
    minimum_task_count_for_distillation: int = typer.Option(2, min=1),
) -> None:
    config = SkillFailureStudyConfig(
        retrieval_methods=_parse_csv_enum(retrieval_methods, RetrievalMethod),
        pool_sizes=[int(item) for item in _parse_csv(pool_sizes)],
        injection_strategies=_parse_csv_enum(injection_strategies, InjectionStrategy),
        self_generated_modes=_parse_csv_enum(self_generated_modes, SelfGeneratedMode),
        split_modes=_parse_csv_enum(split_modes, SplitMode),
        supported_benchmarks=_parse_csv(benchmarks) or None,
        supported_agents=_parse_csv(agents) or None,
        terminal_bench_dataset_filters=_parse_csv(terminal_bench_datasets) or None,
        terminal_bench_parameter_keys=_parse_csv(terminal_bench_parameter_keys)
        or SkillFailureStudyConfig.model_fields["terminal_bench_parameter_keys"].default_factory(),
        top_k=top_k,
        cross_task_holdout_ratio=cross_task_holdout_ratio,
        random_seed=random_seed,
        qwen3_embedding_model=qwen3_embedding_model,
        qwen3_embedding_local_base_url=qwen3_embedding_local_base_url,
        qwen3_embedding_base_url=qwen3_embedding_base_url,
        qwen3_embedding_api_key=qwen3_embedding_api_key,
        qwen3_embedding_timeout_sec=qwen3_embedding_timeout_sec,
        qwen3_embedding_batch_size=qwen3_embedding_batch_size,
        qwen3_embedding_strict=qwen3_embedding_strict,
        procedural_dbscan_eps=procedural_dbscan_eps,
        procedural_dbscan_min_samples=procedural_dbscan_min_samples,
        batch_rollout_required=batch_rollout_required,
        minimum_task_count_for_distillation=minimum_task_count_for_distillation,
    )
    report = run_skill_failure_study(
        trajectory_path=trajectory_path,
        skill_repository=skill_repository,
        output_path=output_path,
        config=config,
    )
    typer.echo(
        json.dumps(
            {
                "output_path": str(output_path),
                "trace_artifacts": report.get("trace_artifacts", {}),
                "filtered_trajectory_count": report.get("filtered_trajectory_count", 0),
                "cell_count": (report.get("summary") or {}).get("cell_count", 0),
                "mean_hit_at_k": (report.get("summary") or {}).get("mean_hit_at_k", 0.0),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("analyze-skill-failure-traces")
def analyze_skill_failure_traces_cmd(
    trace_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    output_path: Path | None = typer.Argument(None),
) -> None:
    report = analyze_stored_traces(trace_dir)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        typer.echo(f"wrote trace analysis report to {output_path}")
        return
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def _parse_csv_enum(raw: str, enum_cls):
    values = _parse_csv(raw)
    parsed = []
    for value in values:
        try:
            parsed.append(enum_cls(value))
        except ValueError as exc:
            allowed = ", ".join(member.value for member in enum_cls)
            raise typer.BadParameter(f"invalid value '{value}', expected one of: {allowed}") from exc
    if not parsed:
        allowed = ", ".join(member.value for member in enum_cls)
        raise typer.BadParameter(f"at least one value is required, expected one of: {allowed}")
    return parsed


def main() -> None:
    app()


if __name__ == "__main__":
    main()
