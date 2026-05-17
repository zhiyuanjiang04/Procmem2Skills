#!/usr/bin/env python3
"""Generate the Prefill-Context Execution Eval design document as PDF."""

from fpdf import FPDF
from datetime import datetime


class DesignDocPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 8, "Prefill-Context Execution Pass-Rate Eval -- Design Document", align="C")
            self.ln(4)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def title_page(self):
        self.add_page()
        self.ln(50)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 14, "Prefill-Context\nExecution Pass-Rate Eval", align="C")
        self.ln(8)
        self.set_font("Helvetica", "", 16)
        self.set_text_color(80, 80, 80)
        self.cell(0, 10, "Design Document", align="C")
        self.ln(20)
        self.set_draw_color(60, 60, 60)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(15)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "procmem2skills / SkillsBench Evaluation Pipeline", align="C")
        self.ln(8)
        self.cell(0, 8, f"Version 1.0  |  {datetime.now().strftime('%Y-%m-%d')}", align="C")
        self.ln(8)
        self.cell(0, 8, "Model Under Test: Claude Sonnet 4.6", align="C")

    def section(self, title, level=1):
        self.ln(4)
        if level == 1:
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(20, 60, 120)
            self.cell(0, 10, title)
            self.ln(3)
            self.set_draw_color(20, 60, 120)
            self.line(10, self.get_y(), 200, self.get_y())
        elif level == 2:
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(40, 40, 40)
            self.cell(0, 9, title)
        elif level == 3:
            self.set_font("Helvetica", "BI", 11)
            self.set_text_color(60, 60, 60)
            self.cell(0, 8, title)
        self.ln(6)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, indent=10):
        x = self.get_x()
        self.set_x(x + indent)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.cell(5, 5.5, "-")
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def code_block(self, text):
        self.set_font("Courier", "", 9)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        w = self.w - self.l_margin - self.r_margin
        for line in text.split("\n"):
            self.set_x(x + 4)
            self.cell(w - 8, 5, line, fill=True)
            self.ln(5)
        self.ln(3)

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            n = len(headers)
            w = (self.w - self.l_margin - self.r_margin) / n
            col_widths = [w] * n
        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        for ri, row in enumerate(rows):
            fill = ri % 2 == 1
            if fill:
                self.set_fill_color(245, 245, 250)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, fill=fill, align="C")
            self.ln()
        self.ln(4)


def build_pdf():
    pdf = DesignDocPDF()
    pdf.alias_nb_pages()

    # ---- Title Page ----
    pdf.title_page()

    # ---- 1. Overview ----
    pdf.add_page()
    pdf.section("1. Overview")
    pdf.body(
        "This document describes the Prefill-Context Execution Pass-Rate Evaluation pipeline, "
        "a controlled experiment measuring how well Claude Sonnet 4.6 executes software engineering "
        "tasks when candidate skills (SKILL.md procedural documents) are injected into the system "
        "prompt as prefilled context.\n\n"
        "The pipeline isolates execution competence from retrieval competence: instead of asking "
        "the model to select which skill to use (tested separately in the Selection-Proxy eval), "
        "all candidate skills are provided upfront. The research question becomes: given the "
        "correct procedural guidance plus controlled noise, can the model complete the task?"
    )

    pdf.section("1.1 Research Questions", level=2)
    pdf.bullet("RQ1: Does prefilling ground-truth SKILL.md content improve execution pass rate?")
    pdf.bullet("RQ2: How does noise type (random / hard / easy distractors) affect pass rate?")
    pdf.bullet("RQ3: How does pool size (5 / 10 / 20 / 50 total candidates) affect pass rate?")
    pdf.bullet("RQ4: What is the gap between selection-proxy upper bound and actual execution?")

    pdf.section("1.2 Experimental Design", level=2)
    pdf.body(
        "The experiment follows a factorial design with three independent variables:"
    )
    pdf.table(
        ["Variable", "Levels", "Description"],
        [
            ["Pool Size", "5, 10, 20, 50", "Total candidates (GT + noise) in system prompt"],
            ["Noise Mode", "random, hard, easy", "How distractor skills are sampled"],
            ["Seed", "0 (expandable)", "RNG seed for deterministic pool construction"],
        ],
        col_widths=[35, 35, 120],
    )
    pdf.body(
        "Noise modes:\n"
        "  - random: uniform sample from 47K ClawHub corpus (excluding GT aliases)\n"
        "  - hard: embedding-nearest neighbors to task description (most confusable)\n"
        "  - easy: embedding-farthest skills (trivially distinguishable from GT)"
    )

    pdf.section("1.3 Datasets", level=2)
    pdf.table(
        ["Dataset", "Tasks", "Lightweight", "Status"],
        [
            ["SkillsBench (SB)", "88", "84 (5 heavy)", "Active -- host execution"],
            ["TerminalBench (TB)", "62 validated", "0", "Deferred -- needs container"],
        ],
        col_widths=[45, 25, 35, 85],
    )
    pdf.body(
        "Total trial space: 84 SB tasks x 4 pool sizes x 3 noise modes x 1 seed = 1,008 trials.\n"
        "Smoke mode: 5 tasks x 12 conditions = 60 trials."
    )

    # ---- 2. Architecture ----
    pdf.add_page()
    pdf.section("2. Pipeline Architecture")
    pdf.body("The pipeline consists of five modules executed in sequence per trial:")
    pdf.code_block(
        "Task Spec + GT Metadata\n"
        "        |\n"
        "  [pool_builder_v3.py]  -->  Build exact-sized candidate pool\n"
        "        |\n"
        "  [prompt_assembler.py] -->  Concatenate SKILL.md (GT real + noise stubs)\n"
        "        |\n"
        "  [run_trial.py]        -->  Per-trial orchestration:\n"
        "        |                    1. Setup isolated workspace\n"
        "        |                    2. Invoke claude -p with system prompt\n"
        "        |                    3. Run pytest against outputs\n"
        "        |                    4. Record pass/fail to JSONL\n"
        "        |\n"
        "  [aggregate.py]        -->  JSONL -> pass-rate tables (JSON + Markdown)\n"
        "        |\n"
        "  [run_exec_prefill_sonnet46.sh]  -->  Bash orchestrator"
    )

    pdf.section("2.1 Pool Builder (pool_builder_v3.py)", level=2)
    pdf.body(
        "Constructs a candidate pool of exactly pool_size skills with all ground-truth "
        "skills preserved. If |GT| > pool_size, the trial is skipped (no truncation).\n\n"
        "Key design decisions:\n"
        "  - Pool sizes are exact totals, not |GT| + n_noise (differs from v1 pool builder)\n"
        "  - GT-alias ClawHub IDs excluded from noise sampling to prevent leakage\n"
        "  - Stable seeding via SHA-256 hash of (prefix, task_id, pool_size, noise_mode, seed)\n"
        "  - Final candidate list is shuffled; GT positions recorded for bias analysis"
    )
    pdf.body(
        "Embedding-based noise modes use BAAI/bge-small-en-v1.5 (384-dim) encoder with a "
        "prebuilt FAISS IndexFlatIP over the 47,231-skill ClawHub corpus. Hard mode takes "
        "top-K by cosine similarity to task description; easy mode takes bottom-K."
    )

    pdf.section("2.2 Prompt Assembler (prompt_assembler.py)", level=2)
    pdf.body(
        "Generates the system prompt by concatenating SKILL.md content for all candidates.\n\n"
        "Lookup order per candidate slug:\n"
        "  1. {task_dir}/environment/skills/{name}/SKILL.md  (real GT content)\n"
        "  2. Corpus description -> synthesized stub            (noise content)\n\n"
        "GT skills receive full procedural content (the actual SKILL.md from SkillsBench). "
        "Noise skills receive a synthesized stub containing only the corpus description plus "
        "a note that full content is unavailable. This prevents the model from trivially "
        "distinguishing GT from noise by content quality alone -- it must rely on semantic "
        "relevance to the task.\n\n"
        "Output format uses XML-style wrapping:"
    )
    pdf.code_block(
        '<skill name="mesh-analysis">\n'
        "  [full SKILL.md content]\n"
        "</skill>\n"
        '<skill name="task-patterns">\n'
        "  [synthesized stub from corpus description]\n"
        "</skill>"
    )

    # ---- 3. Trial Execution ----
    pdf.add_page()
    pdf.section("3. Trial Execution (run_trial.py)")

    pdf.section("3.1 Workspace Setup", level=2)
    pdf.body(
        "Each trial receives an isolated workspace at /tmp/exec_prefill/<trial_id>/work/. "
        "The setup mirrors the task's Dockerfile COPY layout without actually building a container:"
    )
    pdf.bullet("Generic pass: copy environment/* (minus Dockerfile and skills/) to workspace root")
    pdf.bullet("Dockerfile-aware pass: parse COPY src dst lines, mirror file layout into workspace")
    pdf.bullet("Tests pass: copy tests/ to workspace/tests/ for tasks referencing /tests/")
    pdf.bullet("Pre-create output directories: output/, logs/, logs/verifier/scores/")

    pdf.section("3.2 Path Rewriting", level=2)
    pdf.body(
        "SkillsBench tasks are authored for container execution and hardcode container paths. "
        "The pipeline rewrites these paths in both instruction.md (user prompt) and test files:"
    )
    pdf.table(
        ["Container Path", "Workspace Mapping"],
        [
            ["/root/, /app/, /home/", "workspace root"],
            ["/data/", "workspace/data/"],
            ["/output/", "workspace/output/"],
            ["/logs/", "workspace/logs/"],
            ["/tests/", "workspace/tests/"],
        ],
        col_widths=[60, 130],
    )

    pdf.section("3.3 Agent Invocation", level=2)
    pdf.body("The model is invoked via Claude Code CLI in programmatic mode:")
    pdf.code_block(
        "claude -p \\\n"
        "  --model claude-sonnet-4-6 \\\n"
        "  --system-prompt <concatenated SKILL.md pool> \\\n"
        "  --output-format text \\\n"
        "  --allowed-tools Bash,Read,Write,Edit,Glob,Grep \\\n"
        "  --no-session-persistence \\\n"
        "  --permission-mode bypassPermissions \\\n"
        "  --cwd <workspace> < <user_prompt>"
    )
    pdf.body(
        "The agent receives the task instruction as user input and has full shell access within "
        "the workspace. A 900-second timeout guards against runaway trials. The agent is expected "
        "to read the prefilled skills, identify the relevant one, and follow its procedural "
        "instructions to produce the required outputs."
    )

    pdf.section("3.4 Test Evaluation", level=2)
    pdf.body(
        "After the agent exits, the task's test suite (tests/test_outputs.py) is executed via "
        "pytest against the workspace. Tests are path-rewritten using the same mapping as the "
        "instruction. A trial passes if pytest exits with code 0.\n\n"
        "The WORKDIR environment variable is set to the workspace path for tests that use it "
        "as a runtime override."
    )

    pdf.section("3.5 Task Filtering", level=2)
    pdf.body("The pipeline filters tasks at the orchestration step:")
    pdf.bullet("Heavy Dockerfile filter: tasks requiring texlive, nodejs, playwright, gcc, cuda, "
               "chromium, or docker are skipped (regex match on Dockerfile content)")
    pdf.bullet("TB tasks: skipped entirely (container execution not implemented)")
    pdf.bullet("Missing instruction.md: recorded as error")

    # ---- 4. Orchestration ----
    pdf.add_page()
    pdf.section("4. Orchestration & Concurrency")
    pdf.body(
        "The orchestrator (run_exec_prefill_sonnet46.sh or docker-compose) manages the trial grid. "
        "Trials are dispatched via Python asyncio with a configurable semaphore for concurrency "
        "control."
    )
    pdf.table(
        ["Parameter", "Default", "Description"],
        [
            ["CONCURRENCY", "2", "Max parallel claude -p invocations"],
            ["WORK_ROOT", "/tmp/exec_prefill", "Base directory for trial workspaces"],
            ["MODEL", "claude-sonnet-4-6", "Model under test"],
            ["POOL_SIZES", "5 10 20 50", "Candidate pool sizes"],
            ["NOISE_MODES", "random hard easy", "Distractor sampling strategies"],
            ["SEEDS", "0", "RNG seeds for pool construction"],
        ],
        col_widths=[40, 40, 110],
    )

    pdf.section("4.1 Resume Support", level=2)
    pdf.body(
        "The pipeline supports resuming interrupted runs via --resume. Completed trials are "
        "identified by the tuple (task_id, pool_size, noise_mode, seed) from the existing JSONL "
        "output file. Trials with a recorded pass/fail, error, or skip status are excluded from "
        "the pending set."
    )

    pdf.section("4.2 Output Format", level=2)
    pdf.body("Each trial emits a JSONL row with the following fields:")
    pdf.code_block(
        '{"trial_id": "sb__3d-scan-calc__sz5__random__s0",\n'
        ' "dataset": "sb", "task_id": "3d-scan-calc",\n'
        ' "pool_size": 5, "noise_mode": "random", "seed": 0,\n'
        ' "gt_slugs": ["mesh-analysis"], "gt_positions": [3],\n'
        ' "n_candidates": 5, "model": "claude-sonnet-4-6",\n'
        ' "agent_rc": 0, "agent_wall_s": 34.7,\n'
        ' "pass": true, "system_prompt_chars": 5311,\n'
        ' "test_log_tail": "2 passed in 0.01s"}'
    )

    # ---- 5. Containerization ----
    pdf.section("5. Containerization (Docker)")
    pdf.body(
        "The pipeline is packaged as a Docker image for reproducible, isolated execution. "
        "This ensures consistent Python/library versions and prevents host-environment contamination."
    )
    pdf.section("5.1 Image Specification", level=2)
    pdf.table(
        ["Component", "Detail"],
        [
            ["Base image", "python:3.12-slim"],
            ["Key deps", "faiss-cpu, sentence-transformers, pytest, torch (CPU)"],
            ["Claude CLI", "@anthropic-ai/claude-code via npm"],
            ["Auth", "Max plan -- mount ~/.claude and ~/.claude.json"],
            ["User", "Non-root evaluser (UID 1000)"],
            ["Data", "Baked into image (testsets/, data/, skillsbench_repo/)"],
            ["Results", "Volume-mounted ./results/"],
        ],
        col_widths=[40, 150],
    )
    pdf.body(
        "Critical constraint: Claude CLI's --permission-mode bypassPermissions is rejected "
        "under root. The image runs as a non-root user (evaluser) to satisfy this requirement.\n\n"
        "The container directory layout matches the code's path expectations:\n"
        "  /workspace/procmem2skills/  (REPO_ROOT -- testsets/ + data/)\n"
        "  /workspace/skillsbench_repo/  (task definitions)\n"
        "  /workspace/terminal-bench/  (TB tasks, deferred)"
    )

    # ---- 6. Data Assets ----
    pdf.add_page()
    pdf.section("6. Data Assets")
    pdf.table(
        ["Asset", "Size", "Description"],
        [
            ["skill_corpus.jsonl", "16 MB", "47,231 ClawHub skills (id, slug, description)"],
            ["skill_embeddings.npy", "70 MB", "47K x 384 float32 BGE-small embeddings"],
            ["index.faiss", "70 MB", "FAISS IndexFlatIP for cosine retrieval"],
            ["skill_metadata.jsonl", "< 1 MB", "id, slug, name, category per skill"],
            ["skillsbench_tasks.jsonl", "< 1 MB", "88 SB task specs (task_id, gt_skills, desc)"],
            ["terminal_bench_validated.jsonl", "< 1 MB", "62 TB task specs (Opus-judge validated)"],
            ["skillsbench_repo/tasks/", "581 MB", "89 SB task directories (env, tests, solution)"],
        ],
        col_widths=[55, 20, 115],
    )

    # ---- 7. Prior Results ----
    pdf.section("7. Prior Results: Selection-Proxy Eval")
    pdf.body(
        "The Selection-Proxy eval (a separate pipeline in skill_selection_eval/) measures the "
        "model's ability to select the correct skill from an XML-formatted candidate list, "
        "without executing the task. Key findings from the SkillsBench full grid "
        "(Sonnet 4.5, 1068 trials):"
    )
    pdf.table(
        ["Pool Size", "Random Hit@1", "Hard Hit@1", "Easy Hit@1"],
        [
            ["5", "80.9%", "59.6%", "85.4%"],
            ["50", "83.2%", "55.1%", "80.9%"],
            ["500", "76.4%", "41.6%", "71.9%"],
        ],
        col_widths=[30, 40, 40, 40],
    )
    pdf.body(
        "Headline finding: Selection collapse under hard noise -- Hit@1 degrades from 59.6% "
        "(sz=5) to 41.6% (sz=500) under embedding-nearest distractors. This pattern reproduced "
        "on TerminalBench validated tasks (69.4% -> 41.9%).\n\n"
        "The prefill-context execution eval extends this by asking: even when the correct "
        "skill's full SKILL.md is in the system prompt, does the model successfully USE it? "
        "The gap between selection-proxy upper bound and execution pass rate measures execution "
        "competence given perfect retrieval."
    )

    # ---- 8. Preliminary Execution Results ----
    pdf.section("8. Preliminary Execution Results (In Progress)")
    pdf.body(
        "Smoke run on GB10 (NVIDIA Grace-Blackwell, ARM64), Claude Sonnet 4.6 via Max plan, "
        "5 tasks x 12 conditions, concurrency 1. Results as of trial completion:"
    )
    pdf.table(
        ["Task", "Pass Rate", "N Trials", "Notes"],
        [
            ["3d-scan-calc", "80%", "5", "Stable; computational task"],
            ["adaptive-cruise-control", "50%", "4", "Mixed results"],
            ["civ6-adjacency-optimizer", "40%", "5", "Complex logic"],
            ["azure-bgp-oscillation", "33%", "3", "Network domain"],
            ["citation-check", "0%", "1", "Previously skipped (regex fix)"],
        ],
        col_widths=[45, 25, 20, 100],
    )
    pdf.ln(2)
    pdf.body("Pass rate by noise type (most striking finding):")
    pdf.table(
        ["Noise Mode", "Pass Rate", "N", "Interpretation"],
        [
            ["hard (emb-near)", "100%", "6", "Semantically related noise aids execution"],
            ["easy (emb-far)", "33%", "4", "Irrelevant noise distracts"],
            ["random", "20%", "5", "Unstructured noise most harmful"],
        ],
        col_widths=[40, 25, 15, 110],
    )
    pdf.body(
        "Preliminary observation: Hard noise (embedding-nearest distractors) achieves the highest "
        "pass rate, inverting the selection-proxy finding where hard noise was most harmful. "
        "Hypothesis: in execution mode, semantically similar noise skills provide additional "
        "relevant context that aids task completion, whereas in selection mode they cause confusion "
        "about which specific skill to pick."
    )

    # ---- 9. Known Limitations ----
    pdf.add_page()
    pdf.section("9. Known Limitations & Future Work")
    pdf.bullet("TerminalBench execution requires per-task Docker containers (not yet implemented)")
    pdf.bullet("5 SB tasks with heavy Dockerfiles (nodejs, playwright, etc.) are skipped")
    pdf.bullet("Single seed (0); multi-seed runs needed for confidence intervals")
    pdf.bullet("Max plan rate limits constrain concurrency to 1-2 parallel trials")
    pdf.bullet("Path rewriting heuristic may miss edge cases in custom task layouts")
    pdf.bullet("Noise skills use corpus-stub descriptions, not full SKILL.md content -- "
               "this asymmetry could leak GT identity to a sufficiently capable model")
    pdf.bullet("No cost tracking per trial (Max plan is flat-rate)")

    pdf.section("9.1 Planned Extensions", level=2)
    pdf.bullet("Full 1,008-trial SB grid with seeds {0, 1, 2} for variance estimation")
    pdf.bullet("Container-based TB execution (62 tasks x 12 conditions = 744 trials)")
    pdf.bullet("Cross-model comparison (Sonnet 4.6 vs Opus 4.6 vs GPT-4o)")
    pdf.bullet("Ablation: GT-only (no noise) as an upper-bound baseline")
    pdf.bullet("Correlation analysis: selection accuracy vs execution pass rate per task")

    # ---- 10. File Reference ----
    pdf.section("10. File Reference")
    pdf.table(
        ["File", "Purpose"],
        [
            ["testsets/exec_eval_prefill/run_trial.py", "Per-trial driver (workspace + claude + pytest)"],
            ["testsets/exec_eval_prefill/pool_builder_v3.py", "Exact-size pool with 3 noise modes"],
            ["testsets/exec_eval_prefill/prompt_assembler.py", "System prompt from SKILL.md content"],
            ["testsets/exec_eval_prefill/aggregate.py", "JSONL -> pass-rate tables"],
            ["testsets/run_exec_prefill_sonnet46.sh", "Bash orchestrator (full/smoke/resume)"],
            ["testsets/data/skillsbench_tasks.jsonl", "88 SB task specifications"],
            ["data/processed/skill_corpus.jsonl", "47K ClawHub skill corpus"],
            ["data/embeddings/skill_embeddings.npy", "BGE-small embedding matrix"],
            ["data/embeddings/index/index.faiss", "FAISS cosine similarity index"],
            ["Dockerfile", "Container image definition"],
            ["docker-compose.yml", "Orchestration with volume mounts"],
        ],
        col_widths=[80, 110],
    )

    return pdf


if __name__ == "__main__":
    pdf = build_pdf()
    out = "/home/beatxiaopi/claude-project/skills-testset-eval/design_document.pdf"
    pdf.output(out)
    print(f"PDF written to {out}")
