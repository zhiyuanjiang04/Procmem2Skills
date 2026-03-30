from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from procmem2skills.research.skill_failure_study import (
    InjectionStrategy,
    Qwen3EmbeddingRuntime,
    RetrievalMethod,
    SelfGeneratedMode,
    SplitMode,
    SkillFailureCategory,
    SkillFailureStudyConfig,
    build_experiment_cells,
    build_benchmark_analysis,
    classify_skill_failure_case,
    expand_index_to_pool_size,
    analyze_stored_traces,
    retrieve_skills,
    run_skill_failure_study,
    split_tasks_for_cross_task_generalization,
)
from procmem2skills.models import Action, BenchmarkKind, Event, ExecutionMode, ExecutionResult, Observation, Trajectory
from procmem2skills.recorder.jsonl import write_trajectories
from procmem2skills.runtime.retrieval import SkillIndex


class SkillFailureStudyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="procmem2skills-study-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_experiment_cells_expands_methods_and_pool_sizes(self) -> None:
        config = SkillFailureStudyConfig(
            retrieval_methods=[RetrievalMethod.PAGE_INDEX, RetrievalMethod.EMBEDDING_BASED],
            pool_sizes=[50, 500],
            injection_strategies=[InjectionStrategy.DIRECT_INLINE],
            top_k=3,
        )
        cells = build_experiment_cells(config)
        labels = [(cell.retrieval_method.value, cell.pool_size, cell.top_k) for cell in cells]
        self.assertEqual(
            labels,
            [
                ("page-index", 50, 3),
                ("page-index", 500, 3),
                ("embedding-based", 50, 3),
                ("embedding-based", 500, 3),
            ],
        )

    def test_build_experiment_cells_covers_pdf_axes(self) -> None:
        config = SkillFailureStudyConfig(
            retrieval_methods=[
                RetrievalMethod.PAGE_INDEX,
                RetrievalMethod.CONTEXT_INJECTION,
                RetrievalMethod.EMBEDDING_BASED,
            ],
            pool_sizes=[50, 500, 5000],
            injection_strategies=[InjectionStrategy.DIRECT_INLINE],
            self_generated_modes=[
                SelfGeneratedMode.ALL_PROCEDURAL_MEMORIES,
                SelfGeneratedMode.SUCCESS_ONLY_MEMORIES,
                SelfGeneratedMode.SKILLS_PLUS_PROCEDURAL_MEMORY,
            ],
            split_modes=[SplitMode.IN_TASK, SplitMode.CROSS_TASK_HOLDOUT],
            top_k=4,
        )
        cells = build_experiment_cells(config)
        self.assertEqual(len(cells), 54)
        signatures = {
            (
                cell.self_generated_mode.value,
                cell.split_mode.value,
                cell.retrieval_method.value,
                cell.pool_size,
            )
            for cell in cells
        }
        self.assertIn(
            (
                "skills-plus-procedural-memory",
                "cross-task-holdout",
                "embedding-based",
                5000,
            ),
            signatures,
        )

    def test_expand_index_to_pool_size_adds_noise_skills(self) -> None:
        self._write_skill("terminal-python-inline", "Run inline Python diagnostics quickly.")
        self._write_skill("terminal-grep", "Search files with grep.")
        index = SkillIndex.from_repository(self.temp_dir)

        expanded = expand_index_to_pool_size(index, pool_size=5, seed=7)
        self.assertEqual(len(expanded.records), 5)
        noise_ids = [skill_id for skill_id in expanded.records if skill_id.startswith("noise-skill-")]
        self.assertEqual(len(noise_ids), 3)

    def test_retrieve_skills_supports_page_context_embedding_modes(self) -> None:
        self._write_skill("terminal-python-inline", "Run inline Python diagnostics quickly.")
        self._write_skill("terminal-grep", "Search files with grep.")
        index = SkillIndex.from_repository(self.temp_dir)

        page_hits = retrieve_skills(index, query="need python diagnostics", method=RetrievalMethod.PAGE_INDEX, top_k=2)
        self.assertEqual(page_hits[0], "terminal-python-inline")

        embedding_hits = retrieve_skills(
            index,
            query="need python diagnostics",
            method=RetrievalMethod.EMBEDDING_BASED,
            top_k=2,
        )
        self.assertEqual(embedding_hits[0], "terminal-python-inline")

        context_hits = retrieve_skills(
            index,
            query="need python diagnostics",
            method=RetrievalMethod.CONTEXT_INJECTION,
            top_k=10,
        )
        self.assertEqual(set(context_hits), {"terminal-python-inline", "terminal-grep"})

    def test_retrieve_skills_supports_qwen3_embedding_alias(self) -> None:
        self._write_skill("terminal-python-inline", "Run inline Python diagnostics quickly.")
        self._write_skill("terminal-grep", "Search files with grep.")
        index = SkillIndex.from_repository(self.temp_dir)

        runtime = Qwen3EmbeddingRuntime(
            model="Qwen/Test-Embed",
            local_base_url="http://127.0.0.1:8000/v1",
            remote_base_url="http://remote-embed/v1",
            api_key="test-key",
            timeout_sec=1,
            batch_size=8,
        )

        def fake_request(*, endpoint, model, texts, api_key, timeout_sec):
            vectors = []
            for text in texts:
                lowered = text.lower()
                if "python" in lowered:
                    vectors.append([1.0, 0.0])
                elif "grep" in lowered:
                    vectors.append([0.0, 1.0])
                else:
                    vectors.append([0.5, 0.5])
            return vectors

        with patch("procmem2skills.research.skill_failure_study._request_embeddings", side_effect=fake_request):
            hits = retrieve_skills(
                index,
                query="need python diagnostics",
                method=RetrievalMethod.QWEN3_EMBEDDING,
                top_k=2,
                qwen3_runtime=runtime,
            )
        self.assertEqual(hits[0], "terminal-python-inline")

    def test_qwen3_runtime_prefers_local_then_remote_endpoint(self) -> None:
        self._write_skill("terminal-python-inline", "Run inline Python diagnostics quickly.")
        self._write_skill("terminal-grep", "Search files with grep.")
        index = SkillIndex.from_repository(self.temp_dir)
        runtime = Qwen3EmbeddingRuntime(
            model="Qwen/Test-Embed",
            local_base_url="http://127.0.0.1:8000/v1",
            remote_base_url="http://remote-embed/v1",
            api_key="test-key",
            timeout_sec=1,
            batch_size=8,
        )
        calls: list[str] = []

        def fake_request(*, endpoint, model, texts, api_key, timeout_sec):
            calls.append(endpoint)
            if endpoint.startswith("http://127.0.0.1"):
                raise RuntimeError("local endpoint unavailable")
            vectors = []
            for text in texts:
                lowered = text.lower()
                if "python" in lowered:
                    vectors.append([1.0, 0.0])
                elif "grep" in lowered:
                    vectors.append([0.0, 1.0])
                else:
                    vectors.append([0.5, 0.5])
            return vectors

        with patch("procmem2skills.research.skill_failure_study._request_embeddings", side_effect=fake_request):
            hits = retrieve_skills(
                index,
                query="python diagnostics",
                method=RetrievalMethod.QWEN3_EMBEDDING,
                top_k=1,
                qwen3_runtime=runtime,
            )

        self.assertEqual(hits, ["terminal-python-inline"])
        self.assertEqual(calls[0], "http://127.0.0.1:8000/v1")
        self.assertEqual(calls[1], "http://remote-embed/v1")
        self.assertEqual(runtime.active_endpoint, "http://remote-embed/v1")
        self.assertFalse(runtime.used_fallback)

    def test_qwen3_strict_mode_raises_without_lexical_fallback(self) -> None:
        self._write_skill("terminal-python-inline", "Run inline Python diagnostics quickly.")
        index = SkillIndex.from_repository(self.temp_dir)
        runtime = Qwen3EmbeddingRuntime(
            model="Qwen/Test-Embed",
            local_base_url="http://127.0.0.1:8000/v1",
            remote_base_url="http://remote-embed/v1",
            api_key="test-key",
            timeout_sec=1,
            batch_size=8,
            strict=True,
        )

        def fake_request(*, endpoint, model, texts, api_key, timeout_sec):
            raise RuntimeError("embedding service down")

        with patch("procmem2skills.research.skill_failure_study._request_embeddings", side_effect=fake_request):
            with self.assertRaises(RuntimeError):
                retrieve_skills(
                    index,
                    query="python diagnostics",
                    method=RetrievalMethod.QWEN3_EMBEDDING,
                    top_k=1,
                    qwen3_runtime=runtime,
                )

    def test_build_experiment_cells_expands_injection_strategies(self) -> None:
        config = SkillFailureStudyConfig(
            retrieval_methods=[RetrievalMethod.PAGE_INDEX],
            pool_sizes=[50],
            self_generated_modes=[SelfGeneratedMode.SUCCESS_ONLY_MEMORIES],
            split_modes=[SplitMode.IN_TASK],
            injection_strategies=[
                InjectionStrategy.NO_SKILL,
                InjectionStrategy.DIRECT_INLINE,
                InjectionStrategy.CLAUDE_STYLE_PROGRESSIVE,
            ],
            top_k=3,
        )
        cells = build_experiment_cells(config)
        self.assertEqual(len(cells), 3)
        signatures = {(cell.injection_strategy.value, cell.retrieval_method.value) for cell in cells}
        self.assertEqual(
            signatures,
            {
                ("no-skill", "page-index"),
                ("direct-inline", "page-index"),
                ("claude-style-progressive", "page-index"),
            },
        )

    def test_classify_case_detects_unable_to_retrieve(self) -> None:
        category = classify_skill_failure_case(
            oracle_skill_ids=["terminal-python-inline"],
            retrieved_skill_ids=[],
            trajectory_failed=True,
            failure_signals=["timeout"],
            noise_skill_ids=set(),
            executed_commands=2,
        )
        self.assertEqual(category, SkillFailureCategory.UNABLE_TO_RETRIEVE_RELATED_SKILLS)

    def test_classify_case_detects_misled_by_noise(self) -> None:
        category = classify_skill_failure_case(
            oracle_skill_ids=["terminal-python-inline"],
            retrieved_skill_ids=["noise-skill-7-0001", "noise-skill-7-0002"],
            trajectory_failed=True,
            failure_signals=["permission denied"],
            noise_skill_ids={"noise-skill-7-0001", "noise-skill-7-0002"},
            executed_commands=1,
        )
        self.assertEqual(category, SkillFailureCategory.MISLED_BY_NOISY_SKILLS)

    def test_classify_case_detects_pick_right_but_fail_to_use(self) -> None:
        category = classify_skill_failure_case(
            oracle_skill_ids=["terminal-python-inline"],
            retrieved_skill_ids=["terminal-python-inline", "terminal-grep"],
            trajectory_failed=True,
            failure_signals=["AssertionError: output mismatch"],
            noise_skill_ids=set(),
            executed_commands=3,
        )
        self.assertEqual(category, SkillFailureCategory.PICK_RELATED_BUT_FAIL_TO_USE)

    def test_classify_case_detects_agent_misuse(self) -> None:
        category = classify_skill_failure_case(
            oracle_skill_ids=["terminal-python-inline"],
            retrieved_skill_ids=["terminal-python-inline"],
            trajectory_failed=True,
            failure_signals=["timeout"],
            noise_skill_ids=set(),
            executed_commands=0,
        )
        self.assertEqual(category, SkillFailureCategory.AGENT_MISUSE_OF_RELATED_SKILLS)

    def test_classify_case_detects_error_inside_skill(self) -> None:
        category = classify_skill_failure_case(
            oracle_skill_ids=["terminal-python-inline"],
            retrieved_skill_ids=["terminal-python-inline"],
            trajectory_failed=True,
            failure_signals=["Skill is broken: invalid command template"],
            noise_skill_ids=set(),
            executed_commands=2,
        )
        self.assertEqual(category, SkillFailureCategory.ERROR_INSIDE_SKILLS_THEMSELVES)

    def test_split_tasks_for_cross_task_generalization(self) -> None:
        seen, holdout = split_tasks_for_cross_task_generalization(
            ["task-z", "task-a", "task-b", "task-c"],
            holdout_ratio=0.25,
        )
        self.assertEqual(seen, {"task-a", "task-b", "task-c"})
        self.assertEqual(holdout, {"task-z"})

    def test_run_skill_failure_study_writes_json_report(self) -> None:
        skill_repo = self.temp_dir / "skills"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._write_skill("terminal-python-inline", "Run inline Python diagnostics quickly.", root=skill_repo)
        self._write_skill("terminal-grep", "Search files with grep.", root=skill_repo)

        trajectory_path = self.temp_dir / "trajectories.jsonl"
        trajectories = [
            Trajectory(
                episode_id="ep-1",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                harness="terminal-bench/harness",
                agent="codex",
                task_id="task-python",
                instruction="Run python diagnostics",
                mode=ExecutionMode.OFFLINE_BOOTSTRAP,
                completed=True,
                score=1.0,
                events=[
                    Event(
                        step_id=1,
                        observation=Observation(summary="Inspect repo"),
                        action=Action(tool="terminal", name="python", arguments={"command": "python -c 'print(1)'"}),
                        result=ExecutionResult(ok=True, output_text="1"),
                    )
                ],
            ),
            Trajectory(
                episode_id="ep-2",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                harness="terminal-bench/harness",
                agent="codex",
                task_id="task-python",
                instruction="Run python diagnostics for failing tests",
                mode=ExecutionMode.OFFLINE_BOOTSTRAP,
                completed=False,
                score=0.0,
                events=[
                    Event(
                        step_id=1,
                        observation=Observation(summary="Inspect repo"),
                        action=Action(tool="terminal", name="python", arguments={"command": "python -m pytest -q"}),
                        result=ExecutionResult(ok=False, output_text="timeout while running pytest"),
                    )
                ],
            ),
        ]
        write_trajectories(trajectory_path, trajectories)

        output_path = self.temp_dir / "study-report.json"
        report = run_skill_failure_study(
            trajectory_path=trajectory_path,
            skill_repository=skill_repo,
            output_path=output_path,
            config=SkillFailureStudyConfig(
                retrieval_methods=[RetrievalMethod.PAGE_INDEX],
                pool_sizes=[50],
                injection_strategies=[InjectionStrategy.DIRECT_INLINE],
                top_k=3,
            ),
        )
        self.assertTrue(output_path.exists())
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["study_version"], "0.1")
        self.assertEqual(len(payload["cells"]), 1)
        self.assertIn("summary", payload)
        self.assertIn("cell_count", payload["summary"])
        self.assertEqual(report["summary"]["cell_count"], 1)

    def test_run_skill_failure_study_filters_benchmark_and_agent(self) -> None:
        skill_repo = self.temp_dir / "skills-filter"
        self._write_skill("terminal-python-inline", "Run inline Python diagnostics quickly.", root=skill_repo)
        self._write_skill("terminal-grep", "Search files with grep.", root=skill_repo)

        trajectories = [
            self._make_trajectory(
                episode_id="ep-tb-codex-ok",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="task-a",
                instruction="run python checks",
                score=1.0,
                completed=True,
                ok=True,
            ),
            self._make_trajectory(
                episode_id="ep-tb-codex-fail",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="task-b",
                instruction="run grep checks",
                score=0.0,
                completed=False,
                ok=False,
            ),
            self._make_trajectory(
                episode_id="ep-tb-claude-ok",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="claude-code",
                task_id="task-c",
                instruction="run python checks",
                score=1.0,
                completed=True,
                ok=True,
            ),
            self._make_trajectory(
                episode_id="ep-alf-codex-ok",
                benchmark=BenchmarkKind.ALFWORLD,
                agent="codex",
                task_id="task-d",
                instruction="clean mug",
                score=1.0,
                completed=True,
                ok=True,
            ),
        ]
        trajectory_path = self.temp_dir / "filter-trajectories.jsonl"
        write_trajectories(trajectory_path, trajectories)

        report = run_skill_failure_study(
            trajectory_path=trajectory_path,
            skill_repository=skill_repo,
            output_path=self.temp_dir / "study-filter.json",
            config=SkillFailureStudyConfig(
                retrieval_methods=[RetrievalMethod.PAGE_INDEX],
                pool_sizes=[50],
                injection_strategies=[InjectionStrategy.DIRECT_INLINE],
                self_generated_modes=[SelfGeneratedMode.ALL_PROCEDURAL_MEMORIES],
                split_modes=[SplitMode.IN_TASK],
                supported_benchmarks=[BenchmarkKind.TERMINAL_BENCH.value],
                supported_agents=["codex"],
            ),
        )
        self.assertEqual(report["filtered_trajectory_count"], 2)
        self.assertEqual(report["summary"]["cell_count"], 1)
        cell = report["cells"][0]
        self.assertEqual(cell["self_generated_mode"], "all-procedural-memories")
        self.assertEqual(cell["split_mode"], "in-task")
        self.assertEqual(cell["total_cases"], 2)

    def test_build_benchmark_analysis_recommends_suitable_analysis(self) -> None:
        trajectories = [
            self._make_trajectory(
                episode_id="tb-1",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="tb-task",
                instruction="run regression test",
                score=0.0,
                completed=False,
                ok=False,
            ),
            self._make_trajectory(
                episode_id="alf-1",
                benchmark=BenchmarkKind.ALFWORLD,
                agent="codex",
                task_id="alf-task",
                instruction="clean mug",
                score=1.0,
                completed=True,
                ok=True,
            ),
        ]
        report = build_benchmark_analysis(trajectories)
        self.assertIn("terminal-bench", report)
        self.assertIn("alfworld", report)
        self.assertIn("failure-attribution", report["terminal-bench"]["recommended_analysis"])
        self.assertIn("atomic-segmentation", report["alfworld"]["recommended_analysis"])

    def test_build_benchmark_analysis_reports_terminal_bench_versions_and_params(self) -> None:
        trajectories = [
            self._make_trajectory(
                episode_id="tb-v2-0",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="task-v2-0",
                instruction="run regression test",
                score=0.0,
                completed=False,
                ok=False,
                metadata={
                    "dataset_spec": "terminal-bench@2.0",
                    "task_category": "build",
                    "task_difficulty": "easy",
                },
            ),
            self._make_trajectory(
                episode_id="tb-v2-1",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="task-v2-1",
                instruction="run regression test",
                score=1.0,
                completed=True,
                ok=True,
                metadata={
                    "dataset_spec": "terminal-bench@2.1",
                    "task_category": "security",
                    "task_difficulty": "hard",
                },
            ),
        ]
        report = build_benchmark_analysis(trajectories)
        terminal = report["terminal-bench"]
        dataset_profiles = {item["dataset_spec"] for item in terminal["dataset_profiles"]}
        self.assertEqual(dataset_profiles, {"terminal-bench@2.0", "terminal-bench@2.1"})
        self.assertIn("multi-version-slice-comparison", terminal["recommended_analysis"])
        self.assertIn("task-parameter-sensitivity", terminal["recommended_analysis"])
        self.assertEqual(terminal["parameter_cardinality"]["task_category"], 2)

    def test_run_skill_failure_study_filters_terminal_bench_dataset_versions(self) -> None:
        skill_repo = self.temp_dir / "skills-version-filter"
        self._write_skill("terminal-python-inline", "Run inline Python diagnostics quickly.", root=skill_repo)

        trajectories = [
            self._make_trajectory(
                episode_id="ep-v2-0",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="task-a",
                instruction="run python checks",
                score=1.0,
                completed=True,
                ok=True,
                metadata={"dataset_spec": "terminal-bench@2.0"},
            ),
            self._make_trajectory(
                episode_id="ep-v2-1",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="task-b",
                instruction="run python checks",
                score=0.0,
                completed=False,
                ok=False,
                metadata={"dataset_spec": "terminal-bench@2.1"},
            ),
        ]
        trajectory_path = self.temp_dir / "version-filter-trajectories.jsonl"
        write_trajectories(trajectory_path, trajectories)

        report = run_skill_failure_study(
            trajectory_path=trajectory_path,
            skill_repository=skill_repo,
            output_path=self.temp_dir / "version-filter-report.json",
            config=SkillFailureStudyConfig(
                retrieval_methods=[RetrievalMethod.PAGE_INDEX],
                pool_sizes=[50],
                injection_strategies=[InjectionStrategy.DIRECT_INLINE],
                self_generated_modes=[SelfGeneratedMode.ALL_PROCEDURAL_MEMORIES],
                split_modes=[SplitMode.IN_TASK],
                supported_benchmarks=[BenchmarkKind.TERMINAL_BENCH.value],
                terminal_bench_dataset_filters=["terminal-bench@2.0"],
            ),
        )
        self.assertEqual(report["filtered_trajectory_count"], 1)
        cell = report["cells"][0]
        self.assertEqual(cell["terminal_bench_dataset_counts"], {"terminal-bench@2.0": 1})

    def test_self_generated_modes_follow_procedural_memory_design(self) -> None:
        skill_repo = self.temp_dir / "skills-self-modes"
        self._write_skill("terminal-python-inline", "Run inline Python diagnostics quickly.", root=skill_repo)
        self._write_skill("terminal-grep", "Search files with grep.", root=skill_repo)

        trajectories = [
            self._make_trajectory(
                episode_id="ep-succ-a",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="task-a",
                instruction="run python checks",
                score=1.0,
                completed=True,
                ok=True,
            ),
            self._make_trajectory(
                episode_id="ep-fail-b",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="task-b",
                instruction="run grep checks",
                score=0.0,
                completed=False,
                ok=False,
            ),
            self._make_trajectory(
                episode_id="ep-succ-c",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="task-c",
                instruction="run sed checks",
                score=1.0,
                completed=True,
                ok=True,
            ),
        ]
        trajectory_path = self.temp_dir / "self-modes-trajectories.jsonl"
        write_trajectories(trajectory_path, trajectories)

        report = run_skill_failure_study(
            trajectory_path=trajectory_path,
            skill_repository=skill_repo,
            output_path=self.temp_dir / "self-modes-report.json",
            config=SkillFailureStudyConfig(
                retrieval_methods=[RetrievalMethod.PAGE_INDEX],
                pool_sizes=[50],
                injection_strategies=[InjectionStrategy.DIRECT_INLINE],
                self_generated_modes=[
                    SelfGeneratedMode.ALL_PROCEDURAL_MEMORIES,
                    SelfGeneratedMode.SUCCESS_ONLY_MEMORIES,
                    SelfGeneratedMode.SKILLS_PLUS_PROCEDURAL_MEMORY,
                ],
                split_modes=[SplitMode.IN_TASK],
                supported_benchmarks=[BenchmarkKind.TERMINAL_BENCH.value],
                supported_agents=["codex"],
                minimum_task_count_for_distillation=2,
            ),
        )

        by_mode = {cell["self_generated_mode"]: cell for cell in report["cells"]}
        all_proc = by_mode["all-procedural-memories"]
        succ_only = by_mode["success-only-memories"]
        hybrid = by_mode["skills-plus-procedural-memory"]

        self.assertEqual(all_proc["memory_strategy"], "task-procedural-direct")
        self.assertGreater(all_proc["procedural_memory_store"]["terminal-bench"]["failure_labeled_entries"], 0)

        self.assertEqual(succ_only["memory_strategy"], "task-procedural-direct")
        self.assertEqual(succ_only["procedural_memory_store"]["terminal-bench"]["failure_labeled_entries"], 0)

        self.assertEqual(hybrid["memory_strategy"], "procedural-plus-atomic-skills")
        self.assertGreater(hybrid["atomic_skill_count"], 0)

    def test_run_skill_failure_study_persists_trace_artifacts(self) -> None:
        skill_repo = self.temp_dir / "skills-traces"
        self._write_skill("terminal-python-inline", "Run inline Python diagnostics quickly.", root=skill_repo)
        self._write_skill("terminal-grep", "Search files with grep.", root=skill_repo)

        trajectories = [
            self._make_trajectory(
                episode_id="ep-trace-a",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="task-a",
                instruction="run python checks",
                score=1.0,
                completed=True,
                ok=True,
            ),
            self._make_trajectory(
                episode_id="ep-trace-b",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="task-b",
                instruction="run grep checks",
                score=0.0,
                completed=False,
                ok=False,
            ),
        ]
        trajectory_path = self.temp_dir / "trace-artifacts-trajectories.jsonl"
        write_trajectories(trajectory_path, trajectories)

        report = run_skill_failure_study(
            trajectory_path=trajectory_path,
            skill_repository=skill_repo,
            output_path=self.temp_dir / "trace-artifacts-report.json",
            config=SkillFailureStudyConfig(
                retrieval_methods=[RetrievalMethod.PAGE_INDEX],
                pool_sizes=[50],
                injection_strategies=[InjectionStrategy.DIRECT_INLINE],
                self_generated_modes=[SelfGeneratedMode.ALL_PROCEDURAL_MEMORIES],
                split_modes=[SplitMode.IN_TASK],
                supported_benchmarks=[BenchmarkKind.TERMINAL_BENCH.value],
                supported_agents=["codex"],
                minimum_task_count_for_distillation=2,
            ),
        )

        artifacts = report["trace_artifacts"]
        trace_dir = Path(artifacts["trace_dir"])
        manifest_path = Path(artifacts["manifest_path"])
        analysis_path = Path(artifacts["analysis_path"])

        self.assertTrue(trace_dir.exists())
        self.assertTrue(manifest_path.exists())
        self.assertTrue(analysis_path.exists())

        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(manifest_payload["cells"]), 1)

        analysis_payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        expected_total_cases = sum(int(cell["total_cases"]) for cell in report["cells"])
        self.assertEqual(analysis_payload["total_traces"], expected_total_cases)

        rerun_analysis = analyze_stored_traces(trace_dir)
        self.assertEqual(rerun_analysis["total_traces"], analysis_payload["total_traces"])

    def test_run_skill_failure_study_reports_injection_strategies(self) -> None:
        skill_repo = self.temp_dir / "skills-injection"
        self._write_skill("terminal-python-inline", "Run inline Python diagnostics quickly.", root=skill_repo)

        trajectories = [
            self._make_trajectory(
                episode_id="ep-inj-a",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="task-a",
                instruction="run python checks",
                score=1.0,
                completed=True,
                ok=True,
            ),
            self._make_trajectory(
                episode_id="ep-inj-b",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="task-b",
                instruction="run python checks",
                score=0.0,
                completed=False,
                ok=False,
            ),
        ]
        trajectory_path = self.temp_dir / "injection-trajectories.jsonl"
        write_trajectories(trajectory_path, trajectories)

        report = run_skill_failure_study(
            trajectory_path=trajectory_path,
            skill_repository=skill_repo,
            output_path=self.temp_dir / "injection-report.json",
            config=SkillFailureStudyConfig(
                retrieval_methods=[RetrievalMethod.PAGE_INDEX],
                pool_sizes=[50],
                self_generated_modes=[SelfGeneratedMode.SUCCESS_ONLY_MEMORIES],
                split_modes=[SplitMode.IN_TASK],
                injection_strategies=[
                    InjectionStrategy.NO_SKILL,
                    InjectionStrategy.DIRECT_INLINE,
                    InjectionStrategy.CLAUDE_STYLE_PROGRESSIVE,
                ],
                supported_benchmarks=[BenchmarkKind.TERMINAL_BENCH.value],
                supported_agents=["codex"],
                minimum_task_count_for_distillation=1,
            ),
        )

        by_strategy = {cell["injection_strategy"]: cell for cell in report["cells"]}
        self.assertEqual(set(by_strategy), {"no-skill", "direct-inline", "claude-style-progressive"})
        self.assertEqual(by_strategy["no-skill"]["retrieved_skill_count"], 0)
        self.assertGreaterEqual(by_strategy["direct-inline"]["retrieved_skill_count"], 1)
        self.assertGreaterEqual(by_strategy["claude-style-progressive"]["retrieved_skill_count"], 1)

    def test_experiment_cells_are_decoupled_without_cross_strategy_contamination(self) -> None:
        skill_repo = self.temp_dir / "skills-decouple"
        self._write_skill("terminal-python-inline", "Run inline Python diagnostics quickly.", root=skill_repo)
        self._write_skill("terminal-grep", "Search files with grep.", root=skill_repo)

        trajectories = [
            self._make_trajectory(
                episode_id="ep-dec-a",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="task-a",
                instruction="run python checks",
                score=1.0,
                completed=True,
                ok=True,
            ),
            self._make_trajectory(
                episode_id="ep-dec-b",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="task-b",
                instruction="run grep checks",
                score=0.0,
                completed=False,
                ok=False,
            ),
        ]
        trajectory_path = self.temp_dir / "decouple-trajectories.jsonl"
        write_trajectories(trajectory_path, trajectories)

        report = run_skill_failure_study(
            trajectory_path=trajectory_path,
            skill_repository=skill_repo,
            output_path=self.temp_dir / "decouple-report.json",
            config=SkillFailureStudyConfig(
                retrieval_methods=[RetrievalMethod.PAGE_INDEX],
                pool_sizes=[50],
                self_generated_modes=[SelfGeneratedMode.SUCCESS_ONLY_MEMORIES],
                split_modes=[SplitMode.IN_TASK],
                injection_strategies=[
                    InjectionStrategy.NO_SKILL,
                    InjectionStrategy.DIRECT_INLINE,
                    InjectionStrategy.CLAUDE_STYLE_PROGRESSIVE,
                ],
                supported_benchmarks=[BenchmarkKind.TERMINAL_BENCH.value],
                supported_agents=["codex"],
                minimum_task_count_for_distillation=1,
            ),
        )

        for cell in report["cells"]:
            self.assertTrue(all(case["injection_strategy"] == cell["injection_strategy"] for case in cell["cases"]))
            if cell["injection_strategy"] == "no-skill":
                self.assertEqual(cell["retrieved_skill_count"], 0)
                self.assertTrue(all(not case["retrieved_skill_ids"] for case in cell["cases"]))
            else:
                self.assertGreaterEqual(cell["retrieved_skill_count"], 1)
                self.assertTrue(any(case["retrieved_skill_ids"] for case in cell["cases"]))

    def test_procedural_memory_dedupe_uses_embedding_dbscan_qwen(self) -> None:
        skill_repo = self.temp_dir / "skills-proc-dbscan"
        self._write_skill("terminal-python-inline", "Run inline Python diagnostics quickly.", root=skill_repo)

        trajectories = [
            self._make_trajectory(
                episode_id="ep-dbscan-a",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="task-a",
                instruction="run python checks",
                score=1.0,
                completed=True,
                ok=True,
            ),
            self._make_trajectory(
                episode_id="ep-dbscan-b",
                benchmark=BenchmarkKind.TERMINAL_BENCH,
                agent="codex",
                task_id="task-b",
                instruction="run grep checks",
                score=0.0,
                completed=False,
                ok=False,
            ),
        ]
        trajectory_path = self.temp_dir / "dbscan-trajectories.jsonl"
        write_trajectories(trajectory_path, trajectories)

        with patch("procmem2skills.research.skill_failure_study.WorkflowClusterer") as clusterer_cls:
            clusterer = clusterer_cls.return_value
            clusterer.cluster.return_value = []
            clusterer.dedupe_workflows.return_value = []
            run_skill_failure_study(
                trajectory_path=trajectory_path,
                skill_repository=skill_repo,
                output_path=self.temp_dir / "dbscan-report.json",
                config=SkillFailureStudyConfig(
                    retrieval_methods=[RetrievalMethod.PAGE_INDEX],
                    pool_sizes=[50],
                    injection_strategies=[InjectionStrategy.DIRECT_INLINE],
                    self_generated_modes=[SelfGeneratedMode.ALL_PROCEDURAL_MEMORIES],
                    split_modes=[SplitMode.IN_TASK],
                    supported_benchmarks=[BenchmarkKind.TERMINAL_BENCH.value],
                    supported_agents=["codex"],
                    minimum_task_count_for_distillation=1,
                    qwen3_embedding_model="Qwen/Test-Embed",
                ),
            )

        self.assertIsNotNone(clusterer_cls.call_args)
        kwargs = clusterer_cls.call_args.kwargs
        self.assertEqual(kwargs.get("cluster_backend"), "embedding-dbscan")
        self.assertEqual(kwargs.get("embedding_model"), "Qwen/Test-Embed")

    def _write_skill(self, skill_id: str, description: str, root: Path | None = None) -> None:
        parent = root or self.temp_dir
        skill_dir = parent / skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)
        body = f"""---
name: {skill_id}
description: {description}
---

# {skill_id}

{description}
"""
        (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")

    def _make_trajectory(
        self,
        *,
        episode_id: str,
        benchmark: BenchmarkKind,
        agent: str,
        task_id: str,
        instruction: str,
        score: float,
        completed: bool,
        ok: bool,
        metadata: dict | None = None,
    ) -> Trajectory:
        return Trajectory(
            episode_id=episode_id,
            benchmark=benchmark,
            harness=f"{benchmark.value}/harness",
            agent=agent,
            task_id=task_id,
            instruction=instruction,
            mode=ExecutionMode.OFFLINE_BOOTSTRAP,
            metadata=metadata or {},
            completed=completed,
            score=score,
            events=[
                Event(
                    step_id=1,
                    observation=Observation(summary="step"),
                    action=Action(tool="terminal", name="python", arguments={"command": "python -m pytest -q"}),
                    result=ExecutionResult(ok=ok, output_text="ok" if ok else "failed"),
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
