from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from procmem2skills.analysis.failure import build_failure_analysis_from_trajectories
from procmem2skills.evaluation.pipeline import SkillDistillationPipeline
from procmem2skills.inducer.workflow import induce_workflow
from procmem2skills.miner.clustering import WorkflowClusterer
from procmem2skills.models import BoundaryReason, Event, Segment, Trajectory, WorkflowCandidate
from procmem2skills.recorder.jsonl import load_trajectories
from procmem2skills.runtime.retrieval import SkillIndex, SkillRecord
from procmem2skills.segmenter.heuristics import segment_trajectory


class RetrievalMethod(str, Enum):
    PAGE_INDEX = "page-index"
    CONTEXT_INJECTION = "context-injection"
    EMBEDDING_BASED = "embedding-based"
    QWEN3_EMBEDDING = "qwen3-embedding"


class InjectionStrategy(str, Enum):
    NO_SKILL = "no-skill"
    DIRECT_INLINE = "direct-inline"
    CLAUDE_STYLE_PROGRESSIVE = "claude-style-progressive"


class SelfGeneratedMode(str, Enum):
    ALL_PROCEDURAL_MEMORIES = "all-procedural-memories"
    SUCCESS_ONLY_MEMORIES = "success-only-memories"
    SKILLS_PLUS_PROCEDURAL_MEMORY = "skills-plus-procedural-memory"


class SplitMode(str, Enum):
    IN_TASK = "in-task"
    CROSS_TASK_HOLDOUT = "cross-task-holdout"


class SkillFailureCategory(str, Enum):
    SUCCESS = "success"
    UNABLE_TO_RETRIEVE_RELATED_SKILLS = "unable-to-retrieve-related-skills"
    PICK_WRONG_SKILLS = "pick-wrong-skills"
    PICK_RELATED_BUT_FAIL_TO_USE = "pick-related-skills-but-fail-to-use"
    AGENT_MISUSE_OF_RELATED_SKILLS = "agent-misuse-of-related-skills"
    ERROR_INSIDE_SKILLS_THEMSELVES = "error-inside-skills-themselves"
    MISLED_BY_NOISY_SKILLS = "misled-by-noisy-skills"


class SkillFailureStudyConfig(BaseModel):
    retrieval_methods: list[RetrievalMethod] = Field(default_factory=lambda: [RetrievalMethod.PAGE_INDEX])
    pool_sizes: list[int] = Field(default_factory=lambda: [50])
    injection_strategies: list[InjectionStrategy] = Field(default_factory=lambda: [InjectionStrategy.DIRECT_INLINE])
    self_generated_modes: list[SelfGeneratedMode] = Field(default_factory=lambda: [SelfGeneratedMode.ALL_PROCEDURAL_MEMORIES])
    split_modes: list[SplitMode] = Field(default_factory=lambda: [SplitMode.IN_TASK])
    supported_benchmarks: list[str] | None = None
    supported_agents: list[str] | None = None
    terminal_bench_dataset_filters: list[str] | None = None
    terminal_bench_parameter_keys: list[str] = Field(
        default_factory=lambda: [
            "dataset_spec",
            "dataset_name",
            "dataset_version",
            "task_path",
            "task_category",
            "task_difficulty",
            "task_tags",
            "task_checksum",
        ]
    )
    cross_task_holdout_ratio: float = 0.3
    top_k: int = 5
    random_seed: int = 0
    batch_rollout_required: bool = True
    minimum_task_count_for_distillation: int = 2
    qwen3_embedding_model: str | None = None
    qwen3_embedding_local_base_url: str | None = None
    qwen3_embedding_base_url: str | None = None
    qwen3_embedding_api_key: str | None = None
    qwen3_embedding_timeout_sec: int = 30
    qwen3_embedding_batch_size: int = 16
    qwen3_embedding_strict: bool = False
    procedural_dbscan_eps: float = 0.35
    procedural_dbscan_min_samples: int = 2


class ExperimentCell(BaseModel):
    retrieval_method: RetrievalMethod
    pool_size: int
    injection_strategy: InjectionStrategy
    self_generated_mode: SelfGeneratedMode
    split_mode: SplitMode
    top_k: int


def build_experiment_cells(config: SkillFailureStudyConfig) -> list[ExperimentCell]:
    methods = list(dict.fromkeys(config.retrieval_methods))
    pool_sizes = [size for size in config.pool_sizes if size > 0]
    injection_strategies = list(dict.fromkeys(config.injection_strategies))
    self_modes = list(dict.fromkeys(config.self_generated_modes))
    split_modes = list(dict.fromkeys(config.split_modes))
    cells: list[ExperimentCell] = []
    for method in methods:
        for pool_size in pool_sizes:
            for injection_strategy in injection_strategies:
                for self_mode in self_modes:
                    for split_mode in split_modes:
                        cells.append(
                            ExperimentCell(
                                retrieval_method=method,
                                pool_size=pool_size,
                                injection_strategy=injection_strategy,
                                self_generated_mode=self_mode,
                                split_mode=split_mode,
                                top_k=config.top_k,
                            )
                        )
    return cells


def split_tasks_for_cross_task_generalization(task_ids: list[str], *, holdout_ratio: float) -> tuple[set[str], set[str]]:
    unique_tasks = sorted({task_id for task_id in task_ids if task_id})
    if not unique_tasks:
        return set(), set()
    if len(unique_tasks) == 1:
        return set(unique_tasks), set()

    bounded_ratio = min(max(float(holdout_ratio), 0.0), 0.9)
    holdout_count = max(1, int(len(unique_tasks) * bounded_ratio))
    holdout_count = min(holdout_count, len(unique_tasks) - 1)
    holdout = set(unique_tasks[-holdout_count:])
    seen = set(unique_tasks) - holdout
    return seen, holdout


def expand_index_to_pool_size(index: SkillIndex, pool_size: int, seed: int = 0) -> SkillIndex:
    records = dict(index.records)
    if pool_size <= 0:
        return SkillIndex({}, repo_dir=index.repo_dir)
    if len(records) >= pool_size:
        kept_ids = sorted(records)[:pool_size]
        return SkillIndex({skill_id: records[skill_id] for skill_id in kept_ids}, repo_dir=index.repo_dir)

    noise_cursor = 0
    while len(records) < pool_size:
        noise_cursor += 1
        skill_id = f"noise-skill-{seed}-{noise_cursor:04d}"
        text = f"Noise skill {noise_cursor} unrelated retrieval filler"
        records[skill_id] = SkillRecord(
            skill_id=skill_id,
            body=text,
            name=skill_id,
            description="Noise skill to simulate large skill pools.",
            metadata_vector=_vectorize_text(text),
            fulltext_vector=_vectorize_text(text),
        )
    return SkillIndex(records, repo_dir=index.repo_dir)


class Qwen3EmbeddingRuntime:
    def __init__(
        self,
        *,
        model: str | None = None,
        local_base_url: str | None = None,
        remote_base_url: str | None = None,
        api_key: str | None = None,
        timeout_sec: int = 30,
        batch_size: int = 16,
        strict: bool = False,
    ) -> None:
        self.model = (
            model
            or os.environ.get("PROCMEM_QWEN3_EMBED_MODEL")
            or os.environ.get("OPENROUTER_EMBED_MODEL")
            or "Qwen/Qwen3-Embedding-0.6B"
        )
        self.endpoints = _resolve_qwen3_embedding_endpoints(
            local_base_url=local_base_url,
            remote_base_url=remote_base_url,
        )
        self.api_key = (
            api_key
            or os.environ.get("PROCMEM_QWEN3_EMBED_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or ""
        )
        self.timeout_sec = max(1, int(timeout_sec))
        self.batch_size = max(1, int(batch_size))
        self.strict = bool(strict)
        self.active_endpoint: str | None = None
        self.used_fallback = False
        self._query_vectors: dict[str, list[float]] = {}
        self._record_vectors: dict[str, list[float]] = {}

    def search(self, index: SkillIndex, *, query: str, top_k: int) -> list[str]:
        if top_k <= 0:
            return []
        query_text = query.strip()
        if not query_text or not index.records:
            return []
        try:
            query_vector = self._embed_query(query_text)
            self._ensure_record_vectors(index)
            scored = []
            for skill_id, record_vector in self._record_vectors.items():
                score = _cosine_dense(query_vector, record_vector)
                if score > 0:
                    scored.append((score, skill_id))
            scored.sort(key=lambda item: (-item[0], item[1]))
            return [skill_id for _, skill_id in scored[:top_k]]
        except Exception:
            if self.strict:
                raise
            self.used_fallback = True
            return [hit.skill_id for hit in index.search(query_text, top_k=top_k, scope="fulltext")]

    def _embed_query(self, text: str) -> list[float]:
        cached = self._query_vectors.get(text)
        if cached is not None:
            return cached
        vectors = self._embed_texts([text])
        if not vectors:
            raise RuntimeError("qwen3 embedding returned empty query vector")
        self._query_vectors[text] = vectors[0]
        return vectors[0]

    def _ensure_record_vectors(self, index: SkillIndex) -> None:
        missing_ids = [skill_id for skill_id in index.records if skill_id not in self._record_vectors]
        if not missing_ids:
            return
        texts = [_record_text_for_embedding(index.records[skill_id]) for skill_id in missing_ids]
        vectors = self._embed_texts(texts)
        if len(vectors) != len(missing_ids):
            raise RuntimeError("qwen3 embedding record vector size mismatch")
        for skill_id, vector in zip(missing_ids, vectors):
            self._record_vectors[skill_id] = vector

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            vectors.extend(self._embed_chunk(chunk))
        return vectors

    def _embed_chunk(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for endpoint in self.endpoints:
            if not self.api_key and not _is_local_embedding_endpoint(endpoint):
                continue
            try:
                vectors = _request_embeddings(
                    endpoint=endpoint,
                    model=self.model,
                    texts=texts,
                    api_key=self.api_key,
                    timeout_sec=self.timeout_sec,
                )
                self.active_endpoint = endpoint
                return vectors
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("no qwen3 embedding endpoint configured")


def retrieve_skills(
    index: SkillIndex,
    *,
    query: str,
    method: RetrievalMethod,
    top_k: int,
    qwen3_runtime: Qwen3EmbeddingRuntime | None = None,
) -> list[str]:
    if top_k <= 0:
        return []
    if method == RetrievalMethod.PAGE_INDEX:
        return [hit.skill_id for hit in index.search(query, top_k=top_k, scope="metadata")]
    if method == RetrievalMethod.EMBEDDING_BASED:
        return [hit.skill_id for hit in index.search(query, top_k=top_k, scope="fulltext")]
    if method == RetrievalMethod.QWEN3_EMBEDDING:
        runtime = qwen3_runtime or Qwen3EmbeddingRuntime()
        return runtime.search(index, query=query, top_k=top_k)
    if method == RetrievalMethod.CONTEXT_INJECTION:
        return sorted(index.records.keys())[:top_k]
    raise ValueError(f"unsupported retrieval method: {method}")


def classify_skill_failure_case(
    *,
    oracle_skill_ids: list[str],
    retrieved_skill_ids: list[str],
    trajectory_failed: bool,
    failure_signals: list[str] | None = None,
    noise_skill_ids: set[str] | None = None,
    executed_commands: int | None = None,
) -> SkillFailureCategory:
    oracle_set = set(oracle_skill_ids)
    retrieved = list(retrieved_skill_ids)
    overlap = oracle_set & set(retrieved)
    noise_ids = set(noise_skill_ids or set())
    normalized_signals = [signal.lower() for signal in (failure_signals or [])]

    if overlap:
        if not trajectory_failed:
            return SkillFailureCategory.SUCCESS
        if executed_commands is not None and executed_commands <= 0:
            return SkillFailureCategory.AGENT_MISUSE_OF_RELATED_SKILLS
        if any("skill" in signal and ("invalid" in signal or "broken" in signal) for signal in normalized_signals):
            return SkillFailureCategory.ERROR_INSIDE_SKILLS_THEMSELVES
        return SkillFailureCategory.PICK_RELATED_BUT_FAIL_TO_USE

    if not retrieved:
        return SkillFailureCategory.UNABLE_TO_RETRIEVE_RELATED_SKILLS

    if noise_ids and all(skill_id in noise_ids for skill_id in retrieved):
        return SkillFailureCategory.MISLED_BY_NOISY_SKILLS

    return SkillFailureCategory.PICK_WRONG_SKILLS


def build_benchmark_analysis(trajectories: list[Trajectory]) -> dict[str, dict]:
    by_benchmark: dict[str, list[Trajectory]] = {}
    for trajectory in trajectories:
        by_benchmark.setdefault(trajectory.benchmark.value, []).append(trajectory)

    report: dict[str, dict] = {}
    for benchmark, items in sorted(by_benchmark.items()):
        total = len(items)
        success = len([trajectory for trajectory in items if not _trajectory_failed(trajectory)])
        avg_steps = (sum(len(trajectory.events) for trajectory in items) / total) if total else 0.0
        tools = sorted(
            {
                event.action.tool
                for trajectory in items
                for event in trajectory.events
                if event.action is not None and event.action.tool
            }
        )
        success_rate = (success / total) if total else 0.0
        terminal_profile_summary = _build_terminal_bench_profile_summary(items) if benchmark == "terminal-bench" else None
        recommended = _recommended_analysis_for_benchmark(
            benchmark=benchmark,
            avg_steps=avg_steps,
            success_rate=success_rate,
            tool_count=len(tools),
        )
        if terminal_profile_summary is not None:
            dataset_profile_count = len(terminal_profile_summary["dataset_profiles"])
            if dataset_profile_count > 1 and "multi-version-slice-comparison" not in recommended:
                recommended.append("multi-version-slice-comparison")
            parameter_cardinality = terminal_profile_summary["parameter_cardinality"]
            if (
                int(parameter_cardinality.get("task_category", 0)) > 1
                or int(parameter_cardinality.get("task_difficulty", 0)) > 1
            ) and "task-parameter-sensitivity" not in recommended:
                recommended.append("task-parameter-sensitivity")
        report[benchmark] = {
            "trajectory_count": total,
            "task_count": len({trajectory.task_id for trajectory in items}),
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "tool_modalities": tools,
            "recommended_analysis": recommended,
        }
        if terminal_profile_summary is not None:
            report[benchmark]["dataset_profiles"] = terminal_profile_summary["dataset_profiles"]
            report[benchmark]["parameter_coverage"] = terminal_profile_summary["parameter_coverage"]
            report[benchmark]["parameter_cardinality"] = terminal_profile_summary["parameter_cardinality"]
    return report


def run_skill_failure_study(
    *,
    trajectory_path: Path,
    skill_repository: Path,
    output_path: Path | None,
    config: SkillFailureStudyConfig,
) -> dict:
    resolved_output_path = output_path or Path("skill-failure-study-report.json")
    trajectories = load_trajectories(trajectory_path)
    filtered_trajectories = _filter_trajectories(
        trajectories,
        supported_benchmarks=config.supported_benchmarks,
        supported_agents=config.supported_agents,
        terminal_bench_dataset_filters=config.terminal_bench_dataset_filters,
    )
    repository_skill_index = SkillIndex.from_repository(skill_repository)
    failure_analysis = build_failure_analysis_from_trajectories(filtered_trajectories)
    failure_signals_by_task = _failure_signals_by_task(failure_analysis)

    cells = build_experiment_cells(config)
    cell_reports = []
    for cell in cells:
        qwen3_runtime = _build_qwen3_runtime(config) if cell.retrieval_method == RetrievalMethod.QWEN3_EMBEDDING else None
        train_set, eval_set, seen_tasks, holdout_tasks = _split_train_eval_trajectories(
            filtered_trajectories,
            split_mode=cell.split_mode,
            holdout_ratio=config.cross_task_holdout_ratio,
        )
        min_tasks = config.minimum_task_count_for_distillation if config.batch_rollout_required else 1
        mode_context = _prepare_mode_context(
            mode=cell.self_generated_mode,
            train_set=train_set,
            repository_skill_index=repository_skill_index,
            minimum_task_count=min_tasks,
            pool_size=cell.pool_size,
            seed=config.random_seed,
            config=config,
        )
        expanded_index = mode_context["expanded_index"]
        noise_ids = mode_context["noise_ids"]
        bucket_counts: Counter[str] = Counter()
        case_reports = []
        hit_count = 0
        retrieved_skill_count = 0
        progressive_expand_count = 0
        progressive_catalog_count = 0
        terminal_dataset_counts: Counter[str] = Counter()

        for trajectory in eval_set:
            query = _build_query_from_trajectory(trajectory)
            oracle = sorted(mode_context["oracle_by_task"].get(trajectory.task_id, set()))
            candidate_pool_k = max(cell.top_k, min(cell.pool_size, cell.top_k * 3))
            base_retrieved = _retrieve_for_mode(
                mode=cell.self_generated_mode,
                task_id=trajectory.task_id,
                query=query,
                top_k=cell.top_k,
                retrieval_method=cell.retrieval_method,
                procedural_by_task=mode_context["procedural_by_task"],
                expanded_index=expanded_index,
                qwen3_runtime=qwen3_runtime,
            )
            progressive_candidates = _retrieve_for_mode(
                mode=cell.self_generated_mode,
                task_id=trajectory.task_id,
                query=query,
                top_k=candidate_pool_k,
                retrieval_method=cell.retrieval_method,
                procedural_by_task=mode_context["procedural_by_task"],
                expanded_index=expanded_index,
                qwen3_runtime=qwen3_runtime,
            )
            retrieved, injection_trace = _apply_injection_strategy(
                strategy=cell.injection_strategy,
                query=query,
                top_k=cell.top_k,
                base_retrieved=base_retrieved,
                progressive_candidates=progressive_candidates,
                expanded_index=expanded_index,
            )
            failed = _trajectory_failed(trajectory)
            executed_commands = len([event for event in trajectory.events if event.action is not None])
            failure_signals = failure_signals_by_task.get(trajectory.task_id, [])
            terminal_profile = _extract_terminal_bench_profile(
                trajectory,
                parameter_keys=config.terminal_bench_parameter_keys,
            )
            if terminal_profile is not None:
                terminal_dataset_counts[terminal_profile["dataset_spec"]] += 1
            category = classify_skill_failure_case(
                oracle_skill_ids=oracle,
                retrieved_skill_ids=retrieved,
                trajectory_failed=failed,
                failure_signals=failure_signals,
                noise_skill_ids=noise_ids,
                executed_commands=executed_commands,
            )
            if set(oracle) & set(retrieved):
                hit_count += 1
            retrieved_skill_count += len(retrieved)
            progressive_expand_count += int(injection_trace.get("expanded_count", 0))
            progressive_catalog_count += int(injection_trace.get("catalog_count", 0))
            bucket_counts[category.value] += 1
            case_reports.append(
                {
                    "episode_id": trajectory.episode_id,
                    "task_id": trajectory.task_id,
                    "benchmark": trajectory.benchmark.value,
                    "query": query,
                    "injection_strategy": cell.injection_strategy.value,
                    "terminal_bench_dataset": terminal_profile["dataset_spec"] if terminal_profile else None,
                    "terminal_bench_parameters": terminal_profile["parameters"] if terminal_profile else {},
                    "oracle_skill_ids": oracle,
                    "retrieved_skill_ids": retrieved,
                    "injection_trace": injection_trace,
                    "failed": failed,
                    "category": category.value,
                    "failure_signals": failure_signals,
                }
            )

        total_cases = len(case_reports)
        hit_at_k = (hit_count / total_cases) if total_cases else 0.0
        cell_reports.append(
            {
                "retrieval_method": cell.retrieval_method.value,
                "pool_size": cell.pool_size,
                "injection_strategy": cell.injection_strategy.value,
                "self_generated_mode": cell.self_generated_mode.value,
                "split_mode": cell.split_mode.value,
                "memory_strategy": mode_context["memory_strategy"],
                "top_k": cell.top_k,
                "train_task_count": len(seen_tasks),
                "eval_task_count": len(holdout_tasks or seen_tasks),
                "oracle_task_count": len(mode_context["oracle_by_task"]),
                "procedural_memory_count": mode_context["procedural_memory_count"],
                "atomic_skill_count": mode_context["atomic_skill_count"],
                "procedural_memory_store": mode_context["procedural_memory_store"],
                "total_cases": total_cases,
                "hit_at_k": hit_at_k,
                "retrieved_skill_count": retrieved_skill_count,
                "progressive_catalog_count": progressive_catalog_count,
                "progressive_expand_count": progressive_expand_count,
                "qwen3_embedding_endpoint": qwen3_runtime.active_endpoint if qwen3_runtime else None,
                "qwen3_used_fallback": bool(qwen3_runtime.used_fallback) if qwen3_runtime else False,
                "terminal_bench_dataset_counts": dict(sorted(terminal_dataset_counts.items())),
                "bucket_counts": dict(sorted(bucket_counts.items())),
                "cases": case_reports,
            }
        )

    mean_hit = 0.0
    if cell_reports:
        mean_hit = sum(float(item.get("hit_at_k", 0.0)) for item in cell_reports) / len(cell_reports)
    trace_artifacts = _persist_trace_artifacts(
        output_path=resolved_output_path,
        cell_reports=cell_reports,
    )

    payload = {
        "study_version": "0.1",
        "rollout_scope": "batch-rollout-first",
        "trajectory_count": len(trajectories),
        "filtered_trajectory_count": len(filtered_trajectories),
        "benchmark_analysis": build_benchmark_analysis(filtered_trajectories),
        "trace_artifacts": trace_artifacts,
        "config": config.model_dump(),
        "cells": cell_reports,
        "summary": {
            "cell_count": len(cell_reports),
            "mean_hit_at_k": mean_hit,
        },
    }
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def analyze_stored_traces(trace_dir: Path) -> dict:
    trace_files = sorted(trace_dir.glob("cell-*.jsonl"))
    by_category: Counter[str] = Counter()
    by_benchmark: Counter[str] = Counter()
    by_retrieval_method: Counter[str] = Counter()
    by_injection_strategy: Counter[str] = Counter()
    by_self_mode: Counter[str] = Counter()
    by_split_mode: Counter[str] = Counter()
    by_terminal_dataset: Counter[str] = Counter()
    total = 0

    for trace_file in trace_files:
        with trace_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = line.strip()
                if not row:
                    continue
                payload = json.loads(row)
                total += 1
                by_category[str(payload.get("category", "unknown"))] += 1
                by_benchmark[str(payload.get("benchmark", "unknown"))] += 1
                by_retrieval_method[str(payload.get("retrieval_method", "unknown"))] += 1
                by_injection_strategy[str(payload.get("injection_strategy", "unknown"))] += 1
                by_self_mode[str(payload.get("self_generated_mode", "unknown"))] += 1
                by_split_mode[str(payload.get("split_mode", "unknown"))] += 1
                dataset_spec = str(payload.get("terminal_bench_dataset") or "").strip().lower()
                if dataset_spec:
                    by_terminal_dataset[dataset_spec] += 1

    return {
        "total_traces": total,
        "trace_file_count": len(trace_files),
        "by_category": dict(sorted(by_category.items())),
        "by_benchmark": dict(sorted(by_benchmark.items())),
        "by_retrieval_method": dict(sorted(by_retrieval_method.items())),
        "by_injection_strategy": dict(sorted(by_injection_strategy.items())),
        "by_self_generated_mode": dict(sorted(by_self_mode.items())),
        "by_split_mode": dict(sorted(by_split_mode.items())),
        "by_terminal_bench_dataset": dict(sorted(by_terminal_dataset.items())),
    }


def _persist_trace_artifacts(*, output_path: Path, cell_reports: list[dict]) -> dict:
    trace_dir = output_path.parent / f"{output_path.stem}.traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    manifest_cells = []

    for index, cell in enumerate(cell_reports, start=1):
        cell_id = f"cell-{index:03d}"
        trace_file = trace_dir / f"{cell_id}.jsonl"
        with trace_file.open("w", encoding="utf-8") as handle:
            for case in cell.get("cases", []):
                row = {
                    "cell_id": cell_id,
                    "retrieval_method": cell.get("retrieval_method"),
                    "pool_size": cell.get("pool_size"),
                    "injection_strategy": cell.get("injection_strategy"),
                    "self_generated_mode": cell.get("self_generated_mode"),
                    "split_mode": cell.get("split_mode"),
                    "memory_strategy": cell.get("memory_strategy"),
                    **case,
                }
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        manifest_cells.append(
            {
                "cell_id": cell_id,
                "trace_file": str(trace_file),
                "case_count": int(cell.get("total_cases", 0)),
                "retrieval_method": cell.get("retrieval_method"),
                "pool_size": cell.get("pool_size"),
                "injection_strategy": cell.get("injection_strategy"),
                "self_generated_mode": cell.get("self_generated_mode"),
                "split_mode": cell.get("split_mode"),
                "memory_strategy": cell.get("memory_strategy"),
            }
        )

    manifest = {
        "trace_dir": str(trace_dir),
        "cells": manifest_cells,
    }
    manifest_path = trace_dir / "trace_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    analysis = analyze_stored_traces(trace_dir)
    analysis_path = trace_dir / "trace_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "trace_dir": str(trace_dir),
        "manifest_path": str(manifest_path),
        "analysis_path": str(analysis_path),
        "trace_file_count": len(manifest_cells),
    }


def _prepare_mode_context(
    *,
    mode: SelfGeneratedMode,
    train_set: list[Trajectory],
    repository_skill_index: SkillIndex,
    minimum_task_count: int,
    pool_size: int,
    seed: int,
    config: SkillFailureStudyConfig,
) -> dict:
    if mode == SelfGeneratedMode.ALL_PROCEDURAL_MEMORIES:
        procedural_bank = _build_procedural_memory_bank(
            train_set,
            include_failed=True,
            minimum_task_count=minimum_task_count,
            config=config,
        )
        expanded_index = expand_index_to_pool_size(procedural_bank["index"], pool_size=pool_size, seed=seed)
        return {
            "memory_strategy": "task-procedural-direct",
            "oracle_by_task": procedural_bank["by_task"],
            "procedural_by_task": procedural_bank["by_task"],
            "expanded_index": expanded_index,
            "noise_ids": {skill_id for skill_id in expanded_index.records if skill_id.startswith("noise-skill-")},
            "procedural_memory_count": procedural_bank["entry_count"],
            "procedural_memory_store": procedural_bank["store_summary"],
            "atomic_skill_count": 0,
        }

    if mode == SelfGeneratedMode.SUCCESS_ONLY_MEMORIES:
        procedural_bank = _build_procedural_memory_bank(
            train_set,
            include_failed=False,
            minimum_task_count=minimum_task_count,
            config=config,
        )
        expanded_index = expand_index_to_pool_size(procedural_bank["index"], pool_size=pool_size, seed=seed)
        return {
            "memory_strategy": "task-procedural-direct",
            "oracle_by_task": procedural_bank["by_task"],
            "procedural_by_task": procedural_bank["by_task"],
            "expanded_index": expanded_index,
            "noise_ids": {skill_id for skill_id in expanded_index.records if skill_id.startswith("noise-skill-")},
            "procedural_memory_count": procedural_bank["entry_count"],
            "procedural_memory_store": procedural_bank["store_summary"],
            "atomic_skill_count": 0,
        }

    procedural_bank = _build_procedural_memory_bank(
        train_set,
        include_failed=False,
        minimum_task_count=minimum_task_count,
        config=config,
    )
    atomic_bank = _build_atomic_skill_bank(
        train_set,
        minimum_task_count=minimum_task_count,
    )
    combined_index = _merge_skill_indexes([atomic_bank["index"], procedural_bank["index"], repository_skill_index])
    expanded_index = expand_index_to_pool_size(combined_index, pool_size=pool_size, seed=seed)
    return {
        "memory_strategy": "procedural-plus-atomic-skills",
        "oracle_by_task": atomic_bank["by_task"],
        "procedural_by_task": procedural_bank["by_task"],
        "expanded_index": expanded_index,
        "noise_ids": {skill_id for skill_id in expanded_index.records if skill_id.startswith("noise-skill-")},
        "procedural_memory_count": procedural_bank["entry_count"],
        "procedural_memory_store": procedural_bank["store_summary"],
        "atomic_skill_count": atomic_bank["entry_count"],
    }


def _apply_injection_strategy(
    *,
    strategy: InjectionStrategy,
    query: str,
    top_k: int,
    base_retrieved: list[str],
    progressive_candidates: list[str],
    expanded_index: SkillIndex,
) -> tuple[list[str], dict]:
    if strategy == InjectionStrategy.NO_SKILL:
        return [], {"strategy": strategy.value, "catalog_count": 0, "expanded_count": 0}

    if strategy == InjectionStrategy.DIRECT_INLINE:
        selected = list(base_retrieved[:top_k])
        return selected, {
            "strategy": strategy.value,
            "catalog_count": len(selected),
            "expanded_count": len(selected),
        }

    catalog: list[str] = []
    seen = set()
    for skill_id in progressive_candidates + base_retrieved:
        if skill_id in seen:
            continue
        catalog.append(skill_id)
        seen.add(skill_id)
    if not catalog:
        return [], {"strategy": strategy.value, "catalog_count": 0, "expanded_count": 0}

    metadata_hits = index_search_subset(
        index=expanded_index,
        query=query,
        scope="metadata",
        skill_ids=set(catalog),
        top_k=max(top_k, len(catalog)),
    )
    selected: list[str] = []
    for skill_id in metadata_hits + catalog:
        if skill_id in selected:
            continue
        selected.append(skill_id)
        if len(selected) >= top_k:
            break
    return selected, {
        "strategy": strategy.value,
        "catalog_count": len(catalog),
        "expanded_count": len(selected),
    }


def index_search_subset(
    *,
    index: SkillIndex,
    query: str,
    scope: Literal["metadata", "fulltext"],
    skill_ids: set[str],
    top_k: int,
) -> list[str]:
    if top_k <= 0 or not skill_ids:
        return []
    hits = index.search(query, top_k=max(top_k, len(skill_ids)), scope=scope)
    ranked = []
    for hit in hits:
        if hit.skill_id in skill_ids:
            ranked.append(hit.skill_id)
        if len(ranked) >= top_k:
            break
    return ranked


def _retrieve_for_mode(
    *,
    mode: SelfGeneratedMode,
    task_id: str,
    query: str,
    top_k: int,
    retrieval_method: RetrievalMethod,
    procedural_by_task: dict[str, set[str]],
    expanded_index: SkillIndex,
    qwen3_runtime: Qwen3EmbeddingRuntime | None = None,
) -> list[str]:
    if mode in {SelfGeneratedMode.ALL_PROCEDURAL_MEMORIES, SelfGeneratedMode.SUCCESS_ONLY_MEMORIES}:
        direct = sorted(procedural_by_task.get(task_id, set()))
        if direct:
            return direct[:top_k]
    if mode == SelfGeneratedMode.SKILLS_PLUS_PROCEDURAL_MEMORY:
        direct = sorted(procedural_by_task.get(task_id, set()))[:top_k]
        retrieved = retrieve_skills(
            expanded_index,
            query=query,
            method=retrieval_method,
            top_k=top_k,
            qwen3_runtime=qwen3_runtime,
        )
        merged = []
        for item in direct + retrieved:
            if item not in merged:
                merged.append(item)
            if len(merged) >= top_k:
                break
        return merged
    return retrieve_skills(
        expanded_index,
        query=query,
        method=retrieval_method,
        top_k=top_k,
        qwen3_runtime=qwen3_runtime,
    )


def _build_procedural_memory_bank(
    trajectories: list[Trajectory],
    *,
    include_failed: bool,
    minimum_task_count: int,
    config: SkillFailureStudyConfig,
) -> dict:
    source = list(trajectories) if include_failed else [trajectory for trajectory in trajectories if not _trajectory_failed(trajectory)]
    task_ids = {trajectory.task_id for trajectory in source if trajectory.task_id}
    if len(task_ids) < max(1, minimum_task_count):
        return {
            "index": SkillIndex({}),
            "by_task": {},
            "entry_count": 0,
            "store_summary": {},
        }
    workflows = _distill_workflows(source, config=config)
    trajectory_by_episode = {trajectory.episode_id: trajectory for trajectory in source}
    records: dict[str, SkillRecord] = {}
    by_task: dict[str, set[str]] = {}
    store: dict[str, dict[str, int | set[str]]] = {}

    covered_episodes: set[str] = set()
    for workflow in workflows:
        episode_id = workflow.source_segment_id.split("-seg-", 1)[0]
        trajectory = trajectory_by_episode.get(episode_id)
        if trajectory is None:
            continue
        covered_episodes.add(episode_id)
        failed = _trajectory_failed(trajectory)
        entry_id = f"pm::{workflow.workflow_id}"
        summary = _workflow_summary(workflow)
        terminal_context_lines = _terminal_context_lines_for_memory(trajectory)
        body = (
            f"benchmark:{trajectory.benchmark.value}\n"
            f"harness:{trajectory.harness}\n"
            f"{''.join(terminal_context_lines)}"
            f"task:{trajectory.task_id}\n"
            f"instruction:{trajectory.instruction}\n"
            f"outcome:{'failure' if failed else 'success'}\n"
            f"{summary}"
        )
        description = f"Procedural memory for task {trajectory.task_id} ({'failure' if failed else 'success'})"
        records[entry_id] = SkillRecord(
            skill_id=entry_id,
            body=body,
            name=entry_id,
            description=description,
            metadata_vector=_vectorize_text(description),
            fulltext_vector=_vectorize_text(body),
        )
        by_task.setdefault(trajectory.task_id, set()).add(entry_id)

        benchmark = trajectory.benchmark.value
        benchmark_bucket = store.setdefault(
            benchmark,
            {
                "entry_count": 0,
                "success_labeled_entries": 0,
                "failure_labeled_entries": 0,
                "task_ids": set(),
            },
        )
        benchmark_bucket["entry_count"] = int(benchmark_bucket["entry_count"]) + 1
        if failed:
            benchmark_bucket["failure_labeled_entries"] = int(benchmark_bucket["failure_labeled_entries"]) + 1
        else:
            benchmark_bucket["success_labeled_entries"] = int(benchmark_bucket["success_labeled_entries"]) + 1
        task_set = benchmark_bucket["task_ids"]
        if isinstance(task_set, set):
            task_set.add(trajectory.task_id)

    # Preserve failed/success procedural memories even when clustering picks only one canonical entry.
    for trajectory in source:
        if trajectory.episode_id in covered_episodes:
            continue
        failed = _trajectory_failed(trajectory)
        entry_id = f"pm::episode::{trajectory.episode_id}"
        terminal_context_lines = _terminal_context_lines_for_memory(trajectory)
        body = (
            f"benchmark:{trajectory.benchmark.value}\n"
            f"harness:{trajectory.harness}\n"
            f"{''.join(terminal_context_lines)}"
            f"task:{trajectory.task_id}\n"
            f"instruction:{trajectory.instruction}\n"
            f"outcome:{'failure' if failed else 'success'}\n"
            f"{_trajectory_summary(trajectory)}"
        )
        description = f"Procedural memory episode {trajectory.episode_id} ({'failure' if failed else 'success'})"
        records[entry_id] = SkillRecord(
            skill_id=entry_id,
            body=body,
            name=entry_id,
            description=description,
            metadata_vector=_vectorize_text(description),
            fulltext_vector=_vectorize_text(body),
        )
        by_task.setdefault(trajectory.task_id, set()).add(entry_id)
        benchmark = trajectory.benchmark.value
        benchmark_bucket = store.setdefault(
            benchmark,
            {
                "entry_count": 0,
                "success_labeled_entries": 0,
                "failure_labeled_entries": 0,
                "task_ids": set(),
            },
        )
        benchmark_bucket["entry_count"] = int(benchmark_bucket["entry_count"]) + 1
        if failed:
            benchmark_bucket["failure_labeled_entries"] = int(benchmark_bucket["failure_labeled_entries"]) + 1
        else:
            benchmark_bucket["success_labeled_entries"] = int(benchmark_bucket["success_labeled_entries"]) + 1
        task_set = benchmark_bucket["task_ids"]
        if isinstance(task_set, set):
            task_set.add(trajectory.task_id)

    store_summary = {}
    for benchmark, values in store.items():
        task_set = values.get("task_ids")
        task_count = len(task_set) if isinstance(task_set, set) else 0
        store_summary[benchmark] = {
            "entry_count": int(values.get("entry_count", 0)),
            "success_labeled_entries": int(values.get("success_labeled_entries", 0)),
            "failure_labeled_entries": int(values.get("failure_labeled_entries", 0)),
            "task_count": task_count,
        }

    return {
        "index": SkillIndex(records),
        "by_task": by_task,
        "entry_count": len(records),
        "store_summary": store_summary,
    }


def _build_atomic_skill_bank(
    trajectories: list[Trajectory],
    *,
    minimum_task_count: int,
) -> dict:
    successful = [trajectory for trajectory in trajectories if not _trajectory_failed(trajectory)]
    source = successful or list(trajectories)
    task_ids = {trajectory.task_id for trajectory in source if trajectory.task_id}
    if len(task_ids) < max(1, minimum_task_count):
        return {"index": SkillIndex({}), "by_task": {}, "entry_count": 0}
    result = SkillDistillationPipeline(min_support=1).distill(source)
    records: dict[str, SkillRecord] = {}
    by_task: dict[str, set[str]] = {}
    for skill in result.skills:
        entry_id = f"skill::{skill.skill_id}"
        body = (
            f"title:{skill.title}\n"
            f"description:{skill.description}\n"
            f"trigger:{skill.trigger}\n"
            f"actions:{' | '.join(step.operation for step in skill.actions)}"
        )
        description = skill.description or skill.title
        records[entry_id] = SkillRecord(
            skill_id=entry_id,
            body=body,
            name=skill.title or skill.skill_id,
            description=description,
            metadata_vector=_vectorize_text(description),
            fulltext_vector=_vectorize_text(body),
        )
        for task_id in skill.task_origins:
            by_task.setdefault(task_id, set()).add(entry_id)
    return {
        "index": SkillIndex(records),
        "by_task": by_task,
        "entry_count": len(records),
    }


def _merge_skill_indexes(indexes: list[SkillIndex]) -> SkillIndex:
    merged: dict[str, SkillRecord] = {}
    for index in indexes:
        for skill_id, record in index.records.items():
            if skill_id not in merged:
                merged[skill_id] = record
    return SkillIndex(merged)


def _distill_workflows(
    trajectories: list[Trajectory],
    *,
    config: SkillFailureStudyConfig,
) -> list[WorkflowCandidate]:
    segments = []
    for trajectory in trajectories:
        segments.extend(_segment_for_procedural_memory(trajectory))
    if not segments:
        return []

    trajectory_by_episode = {trajectory.episode_id: trajectory for trajectory in trajectories}
    workflows = []
    for segment in segments:
        workflow = induce_workflow(segment)
        trajectory = trajectory_by_episode.get(segment.episode_id)
        workflows.append(_adapt_workflow_for_trace_format(workflow, segment=segment, trajectory=trajectory))
    if not workflows:
        return []

    clusterer = _build_procedural_clusterer(config)
    clusters = clusterer.cluster(workflows, trajectories)
    return clusterer.dedupe_workflows(workflows, clusters)


def _build_procedural_clusterer(config: SkillFailureStudyConfig) -> WorkflowClusterer:
    endpoints = _resolve_qwen3_embedding_endpoints(
        local_base_url=config.qwen3_embedding_local_base_url,
        remote_base_url=config.qwen3_embedding_base_url,
    )
    preferred_endpoint = endpoints[0] if endpoints else (config.qwen3_embedding_base_url or "https://openrouter.ai/api/v1")
    model = (
        config.qwen3_embedding_model
        or os.environ.get("PROCMEM_QWEN3_EMBED_MODEL")
        or os.environ.get("OPENROUTER_EMBED_MODEL")
        or "Qwen/Qwen3-Embedding-0.6B"
    )
    return WorkflowClusterer(
        cluster_backend="embedding-dbscan",
        embedding_model=model,
        embedding_base_url=preferred_endpoint,
        dbscan_eps=float(config.procedural_dbscan_eps),
        dbscan_min_samples=max(1, int(config.procedural_dbscan_min_samples)),
    )


def _segment_for_procedural_memory(trajectory: Trajectory) -> list[Segment]:
    if not _is_terminal_or_skills_bench_trace(trajectory):
        return segment_trajectory(trajectory)
    return _segment_terminal_like_trace(trajectory)


def _segment_terminal_like_trace(trajectory: Trajectory) -> list[Segment]:
    if not trajectory.events:
        return []
    segments: list[Segment] = []
    current_events: list[Event] = []
    start_step = trajectory.events[0].step_id
    max_events_per_segment = 6

    for index, event in enumerate(trajectory.events):
        current_events.append(event)
        boundary_reasons: list[BoundaryReason] = []
        next_event = trajectory.events[index + 1] if index + 1 < len(trajectory.events) else None

        if len(current_events) >= max_events_per_segment:
            boundary_reasons.append(BoundaryReason.MAX_EVENTS)
        if event.success_signal:
            boundary_reasons.append(BoundaryReason.SUCCESS_SIGNAL)
        if event.result is not None and not event.result.ok:
            boundary_reasons.append(BoundaryReason.MAX_EVENTS)
        if next_event and event.action and next_event.action and event.action.tool != next_event.action.tool:
            boundary_reasons.append(BoundaryReason.TOOL_SWITCH)

        if boundary_reasons or next_event is None:
            segment_id = f"{trajectory.episode_id}-seg-{len(segments) + 1}"
            segments.append(
                Segment(
                    segment_id=segment_id,
                    episode_id=trajectory.episode_id,
                    start_step=start_step,
                    end_step=event.step_id,
                    reasons=list(dict.fromkeys(boundary_reasons)) or [BoundaryReason.MAX_EVENTS],
                    tool_sequence=[item.action.tool for item in current_events if item.action],
                    summary_hint=_segment_summary_for_trace(current_events),
                    events=list(current_events),
                )
            )
            current_events = []
            start_step = next_event.step_id if next_event else event.step_id
    return segments


def _adapt_workflow_for_trace_format(
    workflow: WorkflowCandidate,
    *,
    segment: Segment,
    trajectory: Trajectory | None,
) -> WorkflowCandidate:
    if trajectory is None:
        return workflow
    if not _is_terminal_or_skills_bench_trace(trajectory):
        return workflow

    action_events = [event for event in segment.events if event.action]
    for step, event in zip(workflow.steps, action_events):
        command = _extract_terminal_command(event)
        if command:
            step.operation = command
            step.intent = (event.observation.summary or command)[:160]
        if not step.verification and event.result and event.result.output_text:
            step.verification = event.result.output_text.splitlines()[0][:160]

    workflow.objective = trajectory.instruction[:180] if trajectory.instruction else workflow.objective
    workflow.metadata = {
        **workflow.metadata,
        "induce_profile": "terminal-or-skills-bench",
        "benchmark": trajectory.benchmark.value,
        "harness": trajectory.harness,
        "task_id": trajectory.task_id,
    }
    return workflow


def _is_terminal_or_skills_bench_trace(trajectory: Trajectory) -> bool:
    benchmark = (trajectory.benchmark.value or "").strip().lower()
    harness = (trajectory.harness or "").strip().lower()
    return benchmark == "terminal-bench" or "terminal-bench" in harness or "skills-bench" in harness


def _segment_summary_for_trace(events: list[Event]) -> str:
    for event in events:
        command = _extract_terminal_command(event)
        if command:
            return command[:140]
        if event.observation and event.observation.summary:
            return event.observation.summary[:140]
    return "terminal segment"


def _extract_terminal_command(event: Event) -> str:
    if not event.action:
        return ""
    if event.action.raw:
        return " ".join(event.action.raw.split())[:200]
    command = event.action.arguments.get("command")
    if isinstance(command, str) and command.strip():
        return " ".join(command.split())[:200]
    return event.action.name[:200]


def _workflow_summary(workflow: WorkflowCandidate) -> str:
    pieces = [workflow.objective, workflow.trigger]
    pieces.extend(workflow.preconditions)
    pieces.extend(step.operation for step in workflow.steps)
    pieces.extend(workflow.verification)
    pieces.extend(workflow.failure_modes)
    return " | ".join(item.strip() for item in pieces if item and item.strip())


def _trajectory_summary(trajectory: Trajectory) -> str:
    parts = [trajectory.instruction]
    for event in trajectory.events[-3:]:
        if event.observation and event.observation.summary:
            parts.append(event.observation.summary)
        if event.action and event.action.raw:
            parts.append(event.action.raw)
        elif event.action and event.action.name:
            parts.append(event.action.name)
        if event.result and event.result.output_text:
            parts.append(event.result.output_text)
    return " | ".join(item.strip() for item in parts if item and item.strip())


def _filter_trajectories(
    trajectories: list[Trajectory],
    *,
    supported_benchmarks: list[str] | None,
    supported_agents: list[str] | None,
    terminal_bench_dataset_filters: list[str] | None,
) -> list[Trajectory]:
    allowed_benchmarks = {item.strip().lower() for item in (supported_benchmarks or []) if item and item.strip()}
    allowed_agents = {item.strip().lower() for item in (supported_agents or []) if item and item.strip()}
    allowed_terminal_dataset_specs = _normalize_dataset_filter_set(terminal_bench_dataset_filters)
    benchmark_spec_filters = _normalize_dataset_filter_set(
        [item for item in allowed_benchmarks if "terminal-bench" in item]
    )

    filtered = []
    for trajectory in trajectories:
        benchmark_ok = True
        agent_ok = True
        terminal_dataset_ok = True
        observed_terminal_dataset = _infer_terminal_bench_dataset_spec(trajectory)
        if allowed_benchmarks:
            benchmark_ok = trajectory.benchmark.value.lower() in allowed_benchmarks
            if not benchmark_ok and trajectory.benchmark.value.lower() == "terminal-bench" and benchmark_spec_filters:
                benchmark_ok = _dataset_filter_matches(
                    observed_spec=observed_terminal_dataset,
                    allowed_specs=benchmark_spec_filters,
                )
        if allowed_terminal_dataset_specs and trajectory.benchmark.value.lower() == "terminal-bench":
            terminal_dataset_ok = _dataset_filter_matches(
                observed_spec=observed_terminal_dataset,
                allowed_specs=allowed_terminal_dataset_specs,
            )
        if allowed_agents:
            agent_ok = trajectory.agent.strip().lower() in allowed_agents
        if benchmark_ok and agent_ok and terminal_dataset_ok:
            filtered.append(trajectory)
    return filtered


def _split_train_eval_trajectories(
    trajectories: list[Trajectory],
    *,
    split_mode: SplitMode,
    holdout_ratio: float,
) -> tuple[list[Trajectory], list[Trajectory], set[str], set[str]]:
    if split_mode == SplitMode.IN_TASK:
        task_ids = {trajectory.task_id for trajectory in trajectories}
        return trajectories, trajectories, task_ids, task_ids

    seen_tasks, holdout_tasks = split_tasks_for_cross_task_generalization(
        [trajectory.task_id for trajectory in trajectories],
        holdout_ratio=holdout_ratio,
    )
    train = [trajectory for trajectory in trajectories if trajectory.task_id in seen_tasks]
    evaluate = [trajectory for trajectory in trajectories if trajectory.task_id in holdout_tasks]
    if not evaluate:
        evaluate = list(trajectories)
    return train, evaluate, seen_tasks, holdout_tasks


def _failure_signals_by_task(failure_analysis: dict) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    by_task = failure_analysis.get("by_task") or {}
    for task_id, report in by_task.items():
        signals = []
        for item in report.get("failure_signals") or []:
            signature = item.get("signature")
            if signature:
                signals.append(str(signature))
        result[str(task_id)] = signals
    return result


def _build_query_from_trajectory(trajectory: Trajectory) -> str:
    snippets = [trajectory.instruction]
    terminal_profile = _extract_terminal_bench_profile(trajectory, parameter_keys=None)
    if terminal_profile is not None:
        snippets.append(f"dataset:{terminal_profile['dataset_spec']}")
        for key, value in terminal_profile["parameters"].items():
            if isinstance(value, list):
                if value:
                    snippets.append(f"{key}:{'|'.join(str(item) for item in value)}")
                continue
            text = str(value or "").strip()
            if text:
                snippets.append(f"{key}:{text}")
    for event in trajectory.events[-3:]:
        if event.observation and event.observation.summary:
            snippets.append(event.observation.summary)
        if event.action and event.action.raw:
            snippets.append(event.action.raw)
        elif event.action and event.action.name:
            snippets.append(event.action.name)
    return " ".join(part.strip() for part in snippets if part).strip()


def _trajectory_failed(trajectory: Trajectory) -> bool:
    if trajectory.score is not None:
        return trajectory.score < 1.0
    if not trajectory.completed:
        return True
    return any(event.result is not None and not event.result.ok for event in trajectory.events)


def _build_qwen3_runtime(config: SkillFailureStudyConfig) -> Qwen3EmbeddingRuntime:
    return Qwen3EmbeddingRuntime(
        model=config.qwen3_embedding_model,
        local_base_url=config.qwen3_embedding_local_base_url,
        remote_base_url=config.qwen3_embedding_base_url,
        api_key=config.qwen3_embedding_api_key,
        timeout_sec=config.qwen3_embedding_timeout_sec,
        batch_size=config.qwen3_embedding_batch_size,
        strict=config.qwen3_embedding_strict,
    )


def _record_text_for_embedding(record: SkillRecord) -> str:
    return "\n".join(
        part
        for part in [
            record.name,
            record.description,
            record.body,
        ]
        if part
    )


def _resolve_qwen3_embedding_endpoints(*, local_base_url: str | None, remote_base_url: str | None) -> list[str]:
    local = (
        local_base_url
        or os.environ.get("PROCMEM_QWEN3_EMBED_LOCAL_BASE_URL")
        or "http://127.0.0.1:8000/v1"
    )
    remote = (
        remote_base_url
        or os.environ.get("PROCMEM_QWEN3_EMBED_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENROUTER_BASE_URL")
        or "https://openrouter.ai/api/v1"
    )
    endpoints = [_normalize_base_url(local), _normalize_base_url(remote)]
    ordered = []
    seen = set()
    for endpoint in endpoints:
        if not endpoint or endpoint in seen:
            continue
        ordered.append(endpoint)
        seen.add(endpoint)
    return ordered


def _normalize_base_url(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.rstrip("/")


def _is_local_embedding_endpoint(endpoint: str) -> bool:
    normalized = (endpoint or "").strip().lower()
    return normalized.startswith("http://127.0.0.1") or normalized.startswith("http://localhost")


def _request_embeddings(
    *,
    endpoint: str,
    model: str,
    texts: list[str],
    api_key: str,
    timeout_sec: int,
) -> list[list[float]]:
    payload = {
        "model": model,
        "input": texts,
    }
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url=f"{endpoint.rstrip('/')}/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout_sec))) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"embedding request failed with status {exc.code}: {detail}") from exc

    data = body.get("data") or []
    if not isinstance(data, list) or len(data) != len(texts):
        raise RuntimeError("embedding response size mismatch")
    vectors: list[list[float]] = []
    for item in sorted(data, key=lambda row: int(row.get("index", 0))):
        vector = item.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise RuntimeError("invalid embedding payload")
        vectors.append([float(value) for value in vector])
    return vectors


def _cosine_dense(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for index in range(length):
        left_value = float(left[index])
        right_value = float(right[index])
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if dot <= 0 or left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def _vectorize_text(text: str) -> Counter[str]:
    tokens = [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2]
    return Counter(tokens)


def _normalize_dataset_filter_set(filters: list[str] | None) -> set[str]:
    normalized: set[str] = set()
    for raw in filters or []:
        spec = _normalize_dataset_spec(raw)
        if spec:
            normalized.add(spec)
    return normalized


def _dataset_filter_matches(*, observed_spec: str, allowed_specs: set[str]) -> bool:
    if not allowed_specs:
        return True
    observed = _normalize_dataset_spec(observed_spec)
    if not observed:
        return False
    if observed in allowed_specs:
        return True
    observed_name, observed_version, _ = _split_dataset_spec(observed)
    for allowed in allowed_specs:
        allowed_name, allowed_version, _ = _split_dataset_spec(allowed)
        if observed_name == allowed_name:
            # Accept exact match, or when one side omits version metadata.
            if allowed_version is None or observed_version is None or observed_version == allowed_version:
                return True
    return False

def _extract_terminal_bench_profile(
    trajectory: Trajectory,
    *,
    parameter_keys: list[str] | None,
) -> dict | None:
    if trajectory.benchmark.value.lower() != "terminal-bench":
        return None
    metadata = trajectory.metadata or {}
    dataset_spec = _infer_terminal_bench_dataset_spec(trajectory)
    dataset_name, dataset_version, _ = _split_dataset_spec(dataset_spec)

    base_parameters = {
        "dataset_spec": dataset_spec,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "task_path": _string_or_none(metadata.get("task_path")),
        "task_category": _string_or_none(metadata.get("task_category")),
        "task_difficulty": _string_or_none(metadata.get("task_difficulty")),
        "task_tags": _string_list_or_empty(metadata.get("task_tags") or metadata.get("tags")),
        "task_checksum": _string_or_none(metadata.get("task_checksum")),
    }
    selected_keys = [key.strip() for key in (parameter_keys or []) if key and key.strip()]
    if not selected_keys:
        selected_keys = list(base_parameters.keys())
    selected_parameters: dict[str, str | list[str]] = {}
    for key in selected_keys:
        value = base_parameters.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            if value:
                selected_parameters[key] = value
            continue
        text = str(value).strip()
        if text:
            selected_parameters[key] = text
    return {
        "dataset_spec": dataset_spec,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "parameters": selected_parameters,
    }


def _infer_terminal_bench_dataset_spec(trajectory: Trajectory) -> str:
    metadata = trajectory.metadata or {}
    direct_candidates = [
        metadata.get('dataset_spec'),
        metadata.get('dataset'),
        metadata.get('dataset_id'),
        metadata.get('source'),
    ]
    fallback_terminal_bench = ''
    for candidate in direct_candidates:
        spec = _normalize_dataset_spec(candidate)
        if spec and ('terminal-bench' in spec):
            if spec == 'terminal-bench':
                fallback_terminal_bench = spec
            else:
                return spec
    if fallback_terminal_bench:
        return fallback_terminal_bench

    dataset_name = _string_or_none(metadata.get('dataset_name'))
    dataset_version = _string_or_none(metadata.get('dataset_version'))
    if dataset_name:
        return _compose_dataset_spec(dataset_name, dataset_version)

    harness = (trajectory.harness or '').strip().lower()
    if harness:
        prefix = harness.split('/', 1)[0]
        spec = _normalize_dataset_spec(prefix)
        if spec and 'terminal-bench' in spec:
            return spec

    return 'terminal-bench'

def _terminal_context_lines_for_memory(trajectory: Trajectory) -> list[str]:
    profile = _extract_terminal_bench_profile(trajectory, parameter_keys=None)
    if profile is None:
        return []
    lines = [f"dataset:{profile['dataset_spec']}\n"]
    parameters = profile.get("parameters") or {}
    for key, value in parameters.items():
        if key == "dataset_spec":
            continue
        if isinstance(value, list):
            if value:
                lines.append(f"{key}:{'|'.join(value)}\n")
            continue
        text = str(value).strip()
        if text:
            lines.append(f"{key}:{text}\n")
    return lines


def _split_dataset_spec(value: str) -> tuple[str, str | None, str]:
    spec = _normalize_dataset_spec(value)
    if not spec:
        return "", None, ""
    if "@" not in spec:
        return spec, None, spec
    name, version = spec.split("@", 1)
    normalized_name = name.strip()
    normalized_version = version.strip() or None
    if not normalized_version:
        return normalized_name, None, normalized_name
    return normalized_name, normalized_version, f"{normalized_name}@{normalized_version}"


def _compose_dataset_spec(name: str, version: str | None) -> str:
    normalized_name = _normalize_dataset_spec(name)
    if not normalized_name:
        return "terminal-bench"
    normalized_version = str(version or "").strip()
    if not normalized_version:
        return normalized_name
    return f"{normalized_name}@{normalized_version}"


def _normalize_dataset_spec(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        name = _string_or_none(value.get("name") or value.get("dataset_name"))
        version = _string_or_none(value.get("version") or value.get("dataset_version"))
        if name:
            return _compose_dataset_spec(name, version)
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = text.split("/", 1)[0].strip()
    if not text:
        return ""
    if "@" not in text:
        return text
    name, version = text.split("@", 1)
    name = name.strip()
    version = version.strip()
    if not name:
        return ""
    if not version:
        return name
    return f"{name}@{version}"


def _string_or_none(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _string_list_or_empty(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",")]
        return [item for item in parts if item]
    if isinstance(value, (list, tuple, set)):
        rendered = [str(item).strip() for item in value if str(item).strip()]
        unique = sorted(set(rendered))
        return unique
    text = str(value).strip()
    return [text] if text else []


def _build_terminal_bench_profile_summary(trajectories: list[Trajectory]) -> dict:
    dataset_counter: Counter[str] = Counter()
    dataset_tasks: dict[str, set[str]] = {}
    parameter_presence: Counter[str] = Counter()
    parameter_values: dict[str, set[str]] = {}
    parameter_keys = [
        "dataset_name",
        "dataset_version",
        "task_path",
        "task_category",
        "task_difficulty",
        "task_tags",
        "task_checksum",
    ]

    for trajectory in trajectories:
        profile = _extract_terminal_bench_profile(trajectory, parameter_keys=None)
        if profile is None:
            continue
        dataset_spec = profile["dataset_spec"]
        dataset_counter[dataset_spec] += 1
        dataset_tasks.setdefault(dataset_spec, set()).add(trajectory.task_id)

        params = profile["parameters"]
        for key in parameter_keys:
            value = params.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                if not value:
                    continue
                parameter_presence[key] += 1
                bucket = parameter_values.setdefault(key, set())
                for item in value:
                    bucket.add(str(item))
                continue
            text = str(value).strip()
            if text:
                parameter_presence[key] += 1
                parameter_values.setdefault(key, set()).add(text)

    total = len(trajectories)
    dataset_profiles = [
        {
            "dataset_spec": dataset_spec,
            "trajectory_count": count,
            "task_count": len(dataset_tasks.get(dataset_spec, set())),
        }
        for dataset_spec, count in sorted(dataset_counter.items())
    ]
    parameter_coverage = {}
    parameter_cardinality = {}
    for key in parameter_keys:
        present = int(parameter_presence.get(key, 0))
        parameter_coverage[key] = {
            "present_count": present,
            "coverage": (present / total) if total else 0.0,
        }
        parameter_cardinality[key] = len(parameter_values.get(key, set()))
    return {
        "dataset_profiles": dataset_profiles,
        "parameter_coverage": parameter_coverage,
        "parameter_cardinality": parameter_cardinality,
    }


def _recommended_analysis_for_benchmark(
    *,
    benchmark: str,
    avg_steps: float,
    success_rate: float,
    tool_count: int,
) -> list[str]:
    normalized = benchmark.strip().lower()
    if normalized == "terminal-bench":
        return [
            "failure-attribution",
            "retrieval-noise-sensitivity",
            "cross-task-generalization",
        ]
    if normalized == "alfworld":
        return [
            "atomic-segmentation",
            "workflow-clustering",
            "cross-task-generalization",
        ]
    if normalized in {"webarena", "mind2web"}:
        return [
            "context-injection-vs-retrieval",
            "ui-step-segmentation",
            "failure-recovery-analysis",
        ]

    recommendations = ["failure-attribution"]
    if avg_steps >= 6:
        recommendations.append("long-horizon-segmentation")
    if success_rate < 0.5:
        recommendations.append("error-recovery-analysis")
    if tool_count >= 3:
        recommendations.append("multi-tool-composition-analysis")
    return recommendations
