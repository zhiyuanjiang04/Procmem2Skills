from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from procmem2skills.adapters.mind2web import normalize_mind2web_step
from procmem2skills.adapters.terminal_bench import normalize_terminal_bench_step
from procmem2skills.analysis.taxonomy import build_taxonomy_report
from procmem2skills.evaluation.pipeline import SkillDistillationPipeline
from procmem2skills.evaluation.replay_transfer import evaluate_replay_transfer
from procmem2skills.importers import import_alfworld, import_mind2web, import_terminal_bench, import_webarena
from procmem2skills.importers import webarena as webarena_importer
from procmem2skills.inducer.workflow import induce_workflow
from procmem2skills.miner.clustering import WorkflowClusterer
from procmem2skills.miner.atomic_skills import canonical_key
from procmem2skills.models import Action, BenchmarkKind, Event, ExecutionResult, Observation, Trajectory, WorkflowCandidate, WorkflowStep
from procmem2skills.normalization import generic_trigger_phrase, summarize_observation
from procmem2skills.packager.skill_writer import SkillWriter
from procmem2skills.segmenter.heuristics import segment_trajectory


class ImportAndDistillTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="procmem2skills-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _build_terminal_success_trajectory(self, *, episode_id: str, task_id: str, command: str) -> Trajectory:
        return Trajectory(
            episode_id=episode_id,
            benchmark=BenchmarkKind.TERMINAL_BENCH,
            harness="terminal-bench/harness",
            agent="codex",
            task_id=task_id,
            instruction=f"Run `{command}` and verify output.",
            events=[
                normalize_terminal_bench_step(
                    {
                        "summary": "Run command",
                        "cwd": "/app",
                        "command": command,
                        "stdout": "ok",
                        "stderr": "",
                        "ok": True,
                        "exit_code": 0,
                    },
                    1,
                )
            ],
            completed=True,
            score=1.0,
        )

    def test_per_task_aggregation_keeps_task_scoped_skills(self) -> None:
        task_a = self._build_terminal_success_trajectory(episode_id="ep-a", task_id="task-a", command="pytest -q")
        task_b = self._build_terminal_success_trajectory(episode_id="ep-b", task_id="task-b", command="pytest -q")

        global_result = SkillDistillationPipeline(
            min_support=1,
            workflow_aggregation_mode="global",
        ).distill([task_a, task_b])
        per_task_result = SkillDistillationPipeline(
            min_support=1,
            workflow_aggregation_mode="per-task",
            per_task_skill_namespace=True,
        ).distill([task_a, task_b])

        global_ids = {skill.skill_id for skill in global_result.skills}
        per_task_ids = {skill.skill_id for skill in per_task_result.skills}

        self.assertIn("terminal-pytest", global_ids)
        self.assertIn("task-a--terminal-pytest", per_task_ids)
        self.assertIn("task-b--terminal-pytest", per_task_ids)
        self.assertEqual(
            {skill.metadata.get("aggregation_scope") for skill in per_task_result.skills},
            {"per-task"},
        )

    def test_global_dbscan_qwen_mode_uses_qwen_embedding_profile(self) -> None:
        pipeline = SkillDistillationPipeline(workflow_aggregation_mode="global-dbscan-qwen")

        self.assertEqual(pipeline.clusterer.cluster_backend, "embedding-dbscan")
        self.assertEqual(pipeline.clusterer.embedding_model, "Qwen/Qwen3-Embedding-0.6B")
        self.assertTrue(pipeline.clusterer.embedding_strict)

    def test_cross_benchmark_browser_skills_cluster_together(self) -> None:
        trajectories = []
        trajectories.extend(import_mind2web(self.repo_root / "examples/raw-mind2web.json"))
        trajectories.extend(import_webarena(self.repo_root / "examples/raw-webarena.json"))
        trajectories.extend(import_terminal_bench(self.repo_root / "examples/raw-terminal-bench.json"))

        result = SkillDistillationPipeline(min_support=1).distill(trajectories)
        skill_ids = {skill.skill_id for skill in result.skills}

        self.assertIn("browser-click", skill_ids)
        self.assertIn("browser-type", skill_ids)

        browser_click = next(skill for skill in result.skills if skill.skill_id == "browser-click")
        browser_type = next(skill for skill in result.skills if skill.skill_id == "browser-type")

        self.assertEqual(set(browser_click.benchmark_origins), {"mind2web", "webarena"})
        self.assertEqual(set(browser_type.benchmark_origins), {"mind2web", "webarena"})
        self.assertGreaterEqual(len(result.clusters), 2)

        written = SkillWriter().write_repository(result.skills, self.temp_dir / "skills")
        self.assertTrue(any(path.name.startswith("browser-click") for path in written))

    def test_mind2web_click_skill_uses_step_specific_preconditions(self) -> None:
        trajectories = import_mind2web(self.repo_root / "examples/raw-mind2web.json")
        result = SkillDistillationPipeline(min_support=1).distill(trajectories)

        browser_click = next(skill for skill in result.skills if skill.skill_id == "browser-click")
        browser_type = next(skill for skill in result.skills if skill.skill_id == "browser-type")

        self.assertIn("Page shows: Refund Policy", browser_click.preconditions)
        self.assertNotIn("<a>Refund Policy</a>", " ".join(browser_click.preconditions))
        self.assertTrue(all("type refund policy" not in item.lower() for item in browser_type.preconditions))
        self.assertNotIn("Page shows: Refund Policy", browser_type.preconditions)
        self.assertLessEqual(len(browser_click.preconditions), 4)

    def test_mind2web_browser_click_trigger_is_generalized(self) -> None:
        trajectories = import_mind2web(self.repo_root / "examples/raw-mind2web.json")
        result = SkillDistillationPipeline(min_support=1).distill(trajectories)

        browser_click = next(skill for skill in result.skills if skill.skill_id == "browser-click")

        self.assertEqual(browser_click.trigger, "When the agent needs to click a relevant page element.")

    def test_mind2web_long_candidate_payload_produces_bounded_skill_id(self) -> None:
        raw_step = {
            "action_repr": "Click the best candidate",
            "operation": {"op": "click"},
            "pos_candidates": [
                {
                    "backend_node_id": 123,
                    "tag": "button",
                    "text": "DDR3L 8GB Desktop Memory Module by Avarum RAM " * 10,
                    "raw_html": "<button>huge candidate</button>" * 40,
                }
            ],
            "cleaned_html": "<button>candidate</button>",
        }
        event = normalize_mind2web_step(raw_step, 1)
        key = canonical_key(
            WorkflowStep(
                order=1,
                intent="Click the best candidate",
                tool=event.action.tool,
                operation=f"{event.action.name}(element={event.action.arguments['element']})",
            )
        )
        self.assertLessEqual(len(key), 64)

    def test_terminal_git_clone_skill_key_is_compact(self) -> None:
        key = canonical_key(
            WorkflowStep(
                order=1,
                intent="Clone the target repository",
                tool="terminal",
                operation='git(command=git clone --depth 1 --branch 0.5.3 https://example.com/repo.git)',
            )
        )

        self.assertEqual(key, "terminal-git-clone")

    def test_terminal_operand_commands_do_not_leak_arguments_into_skill_key(self) -> None:
        cases = {
            "cd(command=cd pyknotid)": "terminal-cd",
            "ls(command=ls src)": "terminal-ls",
            'grep(command=grep -n "import gcd" pyknotid/make/torus.py)': "terminal-grep",
            "cd(command=cd /app/pyknotid && pip install -e .)": "terminal-pip-install",
            "pip3(command=pip3 install setuptools cython)": "terminal-pip-install",
            'cd(command=cd /tmp && python3 -c "print(1)")': "terminal-python-inline",
            "bash(command=# comment\\nsed -i 's/a/b/' foo.py)": "terminal-sed",
        }

        for operation, expected in cases.items():
            with self.subTest(operation=operation):
                key = canonical_key(
                    WorkflowStep(
                        order=1,
                        intent="Execute a terminal command",
                        tool="terminal",
                        operation=operation,
                    )
                )
                self.assertEqual(key, expected)

    def test_terminal_trigger_uses_semantic_abstraction(self) -> None:
        self.assertEqual(
            generic_trigger_phrase("terminal", 'grep(command=grep -n "import gcd" pyknotid/make/torus.py)'),
            "When the agent needs to search files or code for a relevant pattern from the terminal.",
        )
        self.assertEqual(
            generic_trigger_phrase("terminal", "cd(command=cd /tmp)"),
            "When the agent needs to move into a relevant working directory from the terminal.",
        )

    def test_terminal_multiline_comment_batch_stays_semantic_through_workflow_induction(self) -> None:
        trajectory = Trajectory(
            episode_id="tb-1",
            benchmark=BenchmarkKind.TERMINAL_BENCH,
            harness="terminal-bench/harness",
            agent="agent",
            task_id="build-cython-ext",
            instruction="Apply the compatibility fix.",
            events=[
                normalize_terminal_bench_step(
                    {
                        "summary": "Apply the compatibility fix",
                        "cwd": "/app/pyknotid",
                        "command": "# Fix compatibility\nsed -i 's/a/b/' foo.py",
                        "stdout": "",
                        "ok": True,
                        "exit_code": 0,
                    },
                    1,
                )
            ],
        )

        workflow = induce_workflow(segment_trajectory(trajectory)[0])
        step = workflow.steps[0]

        self.assertEqual(step.operation, "sed(command=# Fix compatibility\\nsed -i 's/a/b/' foo.py)")
        self.assertEqual(canonical_key(step), "terminal-sed")

    def test_alfworld_import_preserves_text_actions(self) -> None:
        trajectories = import_alfworld(self.repo_root / "examples/raw-alfworld.json")

        self.assertEqual(len(trajectories), 1)
        trajectory = trajectories[0]
        self.assertEqual(trajectory.benchmark.value, "alfworld")
        self.assertEqual(len(trajectory.events), 3)
        self.assertEqual(trajectory.events[0].action.tool, "text-world")
        self.assertEqual(trajectory.events[1].action.arguments["command"], "clean mug with faucet")

    def test_terminal_bench_atif_bundle_imports_realistic_rollout(self) -> None:
        bundle_dir = self.temp_dir / "terminal-bench-job" / "build-cython-ext__demo"
        agent_dir = bundle_dir / "agent"
        agent_dir.mkdir(parents=True)

        trajectory_payload = {
            "schema_version": "ATIF-v1.6",
            "session_id": "session-demo",
            "agent": {"name": "terminus-2", "model_name": "anthropic/claude-sonnet-4"},
            "steps": [
                {
                    "source": "user",
                    "message": "Task Description:\nFix the failing test.\n\nCurrent terminal state:\nroot@demo:/app#",
                },
                {
                    "source": "agent",
                    "message": "Inspect the repo and run the failing test.",
                    "tool_calls": [
                        {
                            "function_name": "bash_command",
                            "arguments": {"keystrokes": "ls\n", "duration": 0.1},
                        },
                        {
                            "function_name": "bash_command",
                            "arguments": {"keystrokes": "pytest -q\n", "duration": 1.0},
                        },
                        {
                            "function_name": "bash_command",
                            "arguments": {"keystrokes": "", "duration": 5.0},
                        },
                    ],
                    "observation": {
                        "results": [
                            {
                                "content": (
                                    "New Terminal Output:\n\n"
                                    "root@demo:/app# ls\nREADME.md\ntests\n"
                                    "root@demo:/app# pytest -q\n"
                                    "1 failed, 3 passed\n"
                                    "Traceback (most recent call last):\n"
                                    "AssertionError\n"
                                    "root@demo:/app#"
                                )
                            }
                        ]
                    },
                },
            ],
        }
        config_payload = {
            "task": {
                "path": "build-cython-ext",
                "category": "build",
                "difficulty": "easy",
                "tags": ["python", "c-extension"],
            },
            "trial_name": "build-cython-ext__demo",
        }
        result_payload = {
            "task_name": "build-cython-ext",
            "trial_name": "build-cython-ext__demo",
            "source": "terminal-bench",
            "dataset": "terminal-bench@2.0",
            "task_checksum": "checksum-demo",
            "verifier_result": {"rewards": {"reward": 1.0}},
        }

        for relative_path, payload in {
            Path("agent/trajectory.json"): trajectory_payload,
            Path("config.json"): config_payload,
            Path("result.json"): result_payload,
        }.items():
            target = bundle_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)

        trajectories = import_terminal_bench(bundle_dir)

        self.assertEqual(len(trajectories), 1)
        trajectory = trajectories[0]
        self.assertEqual(trajectory.task_id, "build-cython-ext")
        self.assertEqual(trajectory.score, 1.0)
        self.assertIn("Fix the failing test.", trajectory.instruction)
        self.assertEqual(len(trajectory.events), 2)
        self.assertEqual(trajectory.events[0].action.arguments["command"], "ls")
        self.assertEqual(trajectory.events[0].observation.text, "/app")
        self.assertTrue(trajectory.events[0].result.ok)
        self.assertEqual(trajectory.events[1].action.arguments["command"], "pytest -q")
        self.assertFalse(trajectory.events[1].result.ok)
        self.assertEqual(trajectory.events[1].thought, "Inspect the repo and run the failing test.")
        self.assertEqual(trajectory.metadata.get("dataset_spec"), "terminal-bench@2.0")
        self.assertEqual(trajectory.metadata.get("dataset_name"), "terminal-bench")
        self.assertEqual(trajectory.metadata.get("dataset_version"), "2.0")
        self.assertEqual(trajectory.metadata.get("task_category"), "build")
        self.assertEqual(trajectory.metadata.get("task_difficulty"), "easy")
        self.assertEqual(trajectory.metadata.get("task_tags"), ["c-extension", "python"])

    def test_terminal_bench_atif_bundle_imports_codex_exec_command(self) -> None:
        bundle_dir = self.temp_dir / "terminal-bench-job" / "filter-js__demo"
        agent_dir = bundle_dir / "agent"
        agent_dir.mkdir(parents=True)

        trajectory_payload = {
            "schema_version": "ATIF-v1.5",
            "session_id": "session-codex-demo",
            "agent": {"name": "codex", "model_name": "openai/gpt-5.3-codex"},
            "steps": [
                {
                    "source": "user",
                    "message": "Task Description:\nInspect files.\n\nCurrent terminal state:\nroot@demo:/app#",
                },
                {
                    "source": "agent",
                    "message": "List the workspace first.",
                    "tool_calls": [
                        {
                            "tool_call_id": "call_1",
                            "function_name": "exec_command",
                            "arguments": {"cmd": "ls -la /app"},
                        }
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": "call_1",
                                "content": "Chunk ID: test\nWall time: 0.0000 seconds\nProcess exited with code 0\nOutput:\nREADME.md\n",
                            }
                        ]
                    },
                },
            ],
        }
        config_payload = {"task": {"path": "filter-js-from-html"}, "trial_name": "filter-js__demo"}
        result_payload = {
            "task_name": "filter-js-from-html",
            "trial_name": "filter-js__demo",
            "source": "terminal-bench",
            "task_checksum": "checksum-codex-demo",
            "verifier_result": {"rewards": {"reward": 1.0}},
        }

        for relative_path, payload in {
            Path("agent/trajectory.json"): trajectory_payload,
            Path("config.json"): config_payload,
            Path("result.json"): result_payload,
        }.items():
            target = bundle_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)

        trajectories = import_terminal_bench(bundle_dir)

        self.assertEqual(len(trajectories), 1)
        trajectory = trajectories[0]
        self.assertEqual(len(trajectory.events), 1)
        self.assertEqual(trajectory.events[0].action.arguments["command"], "ls -la /app")
        self.assertTrue(trajectory.events[0].result.ok)
        self.assertEqual(trajectory.events[0].thought, "List the workspace first.")

    def test_webarena_browsergym_result_dir_imports(self) -> None:
        result_dir = self.temp_dir / "browsergym-webarena.7"
        result_dir.mkdir(parents=True)
        (result_dir / "summary_info.json").write_text(
            json.dumps({"cum_reward": 1.0, "terminated": True, "err_msg": None}),
            encoding="utf-8",
        )
        for step in range(2):
            (result_dir / f"step_{step}.pkl.gz").write_bytes(b"placeholder")

        fake_exp_result = SimpleNamespace(
            exp_args=SimpleNamespace(env_args=SimpleNamespace(task_name="browsergym/webarena.7")),
            summary_info={"cum_reward": 1.0, "terminated": True, "err_msg": None},
            tape={
                "steps": [
                    {
                        "kind": "browsergym_observation",
                        "metadata": {"step": 1},
                        "obs": {
                            "goal": "Open the refund policy page.",
                            "axtree_txt": "Refund Policy",
                            "open_pages_urls": ["https://demo.example.com/search"],
                            "active_page_index": 0,
                        },
                        "screenshot": "screenshot_step_1.png",
                    },
                    {
                        "kind": "browsergym_thought",
                        "metadata": {"step": 1},
                        "text": "The refund policy result looks correct.",
                    },
                    {
                        "kind": "browsergym_action",
                        "metadata": {"step": 1},
                        "name": "click",
                        "arguments": ["text=Refund Policy"],
                    },
                    {
                        "kind": "browsergym_observation",
                        "metadata": {"step": 2},
                        "obs": {
                            "goal": "Open the refund policy page.",
                            "axtree_txt": "search box",
                            "open_pages_urls": ["https://demo.example.com"],
                            "active_page_index": 0,
                        },
                        "screenshot": "screenshot_step_2.png",
                    },
                    {
                        "kind": "browsergym_action",
                        "metadata": {"step": 2},
                        "name": "fill",
                        "arguments": ["#search", "refund policy"],
                    },
                ]
            },
        )

        with patch.object(webarena_importer, "_load_browsergym_exp_result", return_value=fake_exp_result):
            trajectories = import_webarena(result_dir)

        self.assertEqual(len(trajectories), 1)
        trajectory = trajectories[0]
        self.assertEqual(trajectory.task_id, "webarena.7")
        self.assertEqual(trajectory.score, 1.0)
        self.assertEqual(trajectory.events[0].action.name, "click")
        self.assertEqual(trajectory.events[0].action.arguments["selector"], "text=Refund Policy")
        self.assertEqual(trajectory.events[0].thought, "The refund policy result looks correct.")
        self.assertEqual(trajectory.events[1].action.name, "type")
        self.assertEqual(trajectory.events[1].action.arguments["value"], "refund policy")
        self.assertEqual(trajectory.events[1].observation.text, "https://demo.example.com")

    def test_clusterer_does_not_merge_browser_workflows_on_cosine_only(self) -> None:
        trajectories = [
            Trajectory(
                episode_id="ep-1",
                benchmark=BenchmarkKind.WEB_ARENA,
                harness="browsergym/webarena",
                agent="browser-agent",
                task_id="wa-1",
                instruction="Open a support article",
            ),
            Trajectory(
                episode_id="ep-2",
                benchmark=BenchmarkKind.WEB_ARENA,
                harness="browsergym/webarena",
                agent="browser-agent",
                task_id="wa-2",
                instruction="Open a support article",
            ),
        ]
        workflows = [
            WorkflowCandidate(
                workflow_id="wf-1",
                source_segment_id="ep-1-seg-1",
                objective="Open the relevant support article",
                trigger="When the agent needs to open the relevant support article.",
                preconditions=["Page shows: search results"],
                steps=[
                    WorkflowStep(order=1, intent="Type the query", tool="browser", operation="type(value=refund policy)"),
                    WorkflowStep(order=2, intent="Click the result", tool="browser", operation="click(element=refund result)"),
                ],
                verification=["target article opened"],
                fingerprint="fp-1",
            ),
            WorkflowCandidate(
                workflow_id="wf-2",
                source_segment_id="ep-2-seg-1",
                objective="Open the relevant support article",
                trigger="When the agent needs to open the relevant support article.",
                preconditions=["Page shows: navigation menu"],
                steps=[
                    WorkflowStep(order=1, intent="Click the support tab", tool="browser", operation="click(element=support tab)"),
                    WorkflowStep(order=2, intent="Click the article link", tool="browser", operation="click(element=article link)"),
                ],
                verification=["support article visible"],
                fingerprint="fp-2",
            ),
        ]

        clusters = WorkflowClusterer(similarity_threshold=0.1, structure_threshold=0.6).cluster(workflows, trajectories)

        self.assertEqual(len(clusters), 2)

    def test_browser_observation_prefers_meaningful_text_over_dom_noise(self) -> None:
        conditions = summarize_observation(
            "browser",
            "",
            (
                "<html backend_node_id='145'>"
                "<a backend_node_id='600'><text backend_node_id='601'>Skip to main content</text></a>"
                "<button backend_node_id='647' aria_label='Book a reservation. Toggle open a menu of reservation types'>"
                "<span><text>Book a reservation</text></span>"
                "</button>"
                "<label><text>Reservation type</text></label>"
                "<select><option><text>Dine in</text></option><option><text>Pickup</text></option></select>"
                "</html>"
            ),
        )

        self.assertEqual(conditions, ["Page shows: Reservation type"])

    def test_taxonomy_report_groups_tasks_hierarchically(self) -> None:
        trajectories = []
        trajectories.extend(import_mind2web(self.repo_root / "examples/raw-mind2web.json"))
        trajectories.extend(import_webarena(self.repo_root / "examples/raw-webarena.json"))
        trajectories.extend(import_alfworld(self.repo_root / "examples/raw-alfworld.json"))
        trajectories.extend(import_terminal_bench(self.repo_root / "examples/raw-terminal-bench.json"))

        report = build_taxonomy_report(trajectories)

        self.assertEqual(report.total_tasks, 4)
        self.assertIn("browser", report.hierarchy)
        self.assertIn("terminal", report.hierarchy)
        self.assertIn("text-world", report.hierarchy)

    def test_replay_transfer_detects_recoverable_divergence(self) -> None:
        success = Trajectory(
            episode_id="success",
            benchmark=BenchmarkKind.TERMINAL_BENCH,
            harness="terminal-bench/harness",
            agent="opus",
            task_id="build-cython-ext",
            instruction="Build the extension and run pytest.",
            completed=True,
            score=1.0,
            events=[
                Event(
                    step_id=1,
                    observation=Observation(summary="Repo cloned", text="/app/pyknotid"),
                    action=Action(tool="terminal", name="git", arguments={"command": "git clone https://example.com/repo.git"}),
                    result=ExecutionResult(ok=True, output_text="cloned"),
                ),
                Event(
                    step_id=2,
                    observation=Observation(summary="Tests failing", text="/app/pyknotid"),
                    action=Action(tool="terminal", name="pytest", arguments={"command": "pytest -q"}),
                    result=ExecutionResult(ok=True, output_text="1 failed"),
                ),
            ],
        )
        failure = Trajectory(
            episode_id="failure",
            benchmark=BenchmarkKind.TERMINAL_BENCH,
            harness="terminal-bench/harness",
            agent="sonnet",
            task_id="build-cython-ext",
            instruction="Build the extension and run pytest.",
            completed=True,
            score=0.0,
            events=[
                Event(
                    step_id=1,
                    observation=Observation(summary="Repo cloned", text="/app/pyknotid"),
                    action=Action(tool="terminal", name="git", arguments={"command": "git clone https://example.com/repo.git"}),
                    result=ExecutionResult(ok=True, output_text="cloned"),
                ),
                Event(
                    step_id=2,
                    observation=Observation(summary="Tests failing", text="/app/pyknotid"),
                    action=Action(tool="terminal", name="ls", arguments={"command": "ls"}),
                    result=ExecutionResult(ok=True, output_text="README.md tests"),
                ),
            ],
        )
        result = SkillDistillationPipeline(min_support=1).distill([success])
        skill_repo = self.temp_dir / "transfer-skills"
        SkillWriter().write_repository(result.skills, skill_repo)

        report = evaluate_replay_transfer(skill_repository=skill_repo, reference=success, target=failure, top_k=3)

        self.assertEqual(report.shared_prefix_steps, 1)
        self.assertIsNotNone(report.divergence)
        self.assertEqual(report.divergence.reference_family, "terminal-pytest")
        self.assertTrue(report.divergence.recoverable_top_k)


if __name__ == "__main__":
    unittest.main()
