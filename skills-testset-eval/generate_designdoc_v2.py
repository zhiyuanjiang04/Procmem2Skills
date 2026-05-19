#!/usr/bin/env python3
"""Generate V2 Design Document — incorporates bug fixes and reviewer feedback."""

from fpdf import FPDF
from datetime import datetime


class V2DocPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 8, "Prefill-Context Execution Eval V2 -- Design Document", align="C")
            self.ln(4)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section(self, title, level=1):
        self.ln(3)
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
        self.set_font("Courier", "", 8)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(30, 30, 30)
        w = self.w - self.l_margin - self.r_margin
        for line in text.split("\n"):
            self.set_x(self.l_margin + 4)
            self.cell(w - 8, 4.5, line[:115], fill=True)
            self.ln(4.5)
        self.ln(2)

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            n = len(headers)
            w = (self.w - self.l_margin - self.r_margin) / n
            col_widths = [w] * n
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        for ri, row in enumerate(rows):
            fill = ri % 2 == 1
            if fill:
                self.set_fill_color(245, 245, 250)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, fill=fill, align="C")
            self.ln()
        self.ln(3)


def build():
    pdf = V2DocPDF()
    pdf.alias_nb_pages()

    # ---- Title ----
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 14, "Prefill-Context\nExecution Pass-Rate Eval", align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(180, 40, 40)
    pdf.cell(0, 10, "V2 Design Document", align="C")
    pdf.ln(15)
    pdf.set_draw_color(60, 60, 60)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Incorporates V1 failure mode audit + reviewer feedback", align="C")
    pdf.ln(8)
    pdf.cell(0, 8, f"Version 2.0  |  {datetime.now().strftime('%Y-%m-%d')}", align="C")
    pdf.ln(8)
    pdf.cell(0, 8, "Model: Claude Sonnet 4.6 via Max Plan  |  Platform: GB10 ARM64", align="C")

    # ---- 1. V1 Post-Mortem ----
    pdf.add_page()
    pdf.section("1. V1 Post-Mortem: Why the Results Were Wrong")
    pdf.body(
        "V1 reported a 43.3% pass rate (26/60). Independent review identified that "
        "this number was contaminated by three categories of infrastructure bugs that "
        "inflated the failure count. A failure mode audit of all 41 V1 FAIL trials revealed:"
    )
    pdf.table(
        ["Failure Mode", "Count", "%", "Is Real Failure?"],
        [
            ["PATH_ERROR (correct answer)", "11", "26.8%", "NO - false negative"],
            ["TIMEOUT (900s)", "9", "22.0%", "YES - model too slow"],
            ["FORMAT_MISMATCH", "9", "22.0%", "PARTIAL - GT/test misalign"],
            ["PATH_ERROR (no answer)", "6", "14.6%", "UNCLEAR"],
            ["LOGIC_ERROR", "6", "14.6%", "YES - genuine failure"],
        ],
        col_widths=[55, 18, 18, 55],
    )
    pdf.body(
        "Key findings from the audit:\n\n"
        "1. Only 6/41 failures (14.6%) were genuine LOGIC_ERRORs where the model produced "
        "wrong answers. The majority were infrastructure artifacts.\n\n"
        "2. 11 failures were false negatives: the model computed the correct answer but wrote "
        "it to a path the rewritten tests could not find (e.g., output/mass_report.json "
        "instead of workspace root).\n\n"
        "3. ALL 9 azure-bgp failures were FORMAT_MISMATCH (test ERRORS, not FAILURES). "
        "The GT SKILL.md guided the model to produce output in a format the test suite did "
        "not expect. This is a SKILL.md-test alignment defect, not a model competence issue.\n\n"
        "4. V1's corrected pass rate (after accounting for false negatives) was ~60%, not 43%."
    )

    pdf.section("1.1 Invalidated V1 Claims", level=2)
    pdf.bullet("'GT-only is worst (20%)' -- partially an artifact of azure-bgp format "
               "mismatch. The SKILL.md taught a format the test didn't accept.")
    pdf.bullet("'Noise helps execution (43% > 40%)' -- the 3pp difference was within "
               "statistical noise at N=5, and contaminated by false negatives.")
    pdf.bullet("'Hard noise is best (50%)' -- may be a length/attention artifact rather "
               "than a semantic diversity effect, given the GT/stub asymmetry.")

    # ---- 2. V2 Changes ----
    pdf.add_page()
    pdf.section("2. V2 Pipeline Changes")
    pdf.body(
        "V2 implements four fixes to address the measurement errors identified in V1, "
        "plus two additional baseline conditions from reviewer feedback."
    )

    pdf.section("2.1 Fix: Output Path Reconciliation", level=2)
    pdf.body(
        "Problem: The model sometimes writes correct output to alternative paths "
        "(output/file.json, /root/file.json) instead of the workspace root where "
        "rewritten tests expect it.\n\n"
        "Fix: After the agent exits and before running tests, a new "
        "_reconcile_output_paths() function copies files from common alternative "
        "locations (output/, root/, app/, app/output/) to the workspace root."
    )
    pdf.code_block(
        "def _reconcile_output_paths(work: Path) -> None:\n"
        "    for sub in ('output', 'root', 'app', 'app/output'):\n"
        "        d = work / sub\n"
        "        if d.is_dir():\n"
        "            for f in d.rglob('*'):\n"
        "                if f.is_file():\n"
        "                    dst = work / f.relative_to(d)\n"
        "                    if not dst.exists():\n"
        "                        shutil.copy2(f, dst)"
    )

    pdf.section("2.2 Fix: System Prompt File (ARG_MAX)", level=2)
    pdf.body(
        "Problem: Large system prompts (>128KB for sz=50 hard noise) exceeded Linux "
        "ARG_MAX when passed as --system-prompt command-line argument, causing 6 trials "
        "to crash silently.\n\n"
        "Fix: Write system prompt to a temporary file and use --system-prompt-file flag."
    )

    pdf.section("2.3 Fix: Dockerfile Filter Regex", level=2)
    pdf.body(
        "Problem: The regex /tex/ matched 'bibtexparser' in citation-check's Dockerfile, "
        "causing all 12 citation-check trials to be skipped as 'heavy Dockerfile'.\n\n"
        "Fix: Changed regex from 'tex' to 'texlive' to avoid false matches."
    )

    pdf.section("2.4 Fix: Sequential Container Execution", level=2)
    pdf.body(
        "Problem: Python asyncio with multiple coroutines caused the process to hang "
        "after 3-16 trials when running inside Docker containers.\n\n"
        "Fix: --max-trials 1 flag ensures each container runs exactly one trial then exits. "
        "A bash loop (run_sequential.sh) manages the trial queue with resume support."
    )

    pdf.section("2.5 New: Baseline Conditions", level=2)
    pdf.table(
        ["Baseline", "Description", "Purpose"],
        [
            ["noskill", "No SKILL.md in system prompt", "Is skill injection useful at all?"],
            ["gtonly", "Only GT skill(s), zero noise", "Is noise harmful or beneficial?"],
        ],
        col_widths=[25, 60, 70],
    )

    # ---- 3. V2 Experimental Design ----
    pdf.add_page()
    pdf.section("3. V2 Experimental Design")

    pdf.section("3.1 Main Experiment (unchanged from V1)", level=2)
    pdf.table(
        ["Variable", "Levels", "Description"],
        [
            ["Pool Size", "5, 10, 20, 50", "Total candidates (GT + noise)"],
            ["Noise Mode", "random, hard, easy", "Distractor sampling strategy"],
            ["Seed", "0", "Deterministic pool construction"],
        ],
        col_widths=[35, 35, 120],
    )
    pdf.body("5 tasks x 4 sizes x 3 noise modes = 60 trials (main experiment).")

    pdf.section("3.2 Baselines (new in V2)", level=2)
    pdf.body(
        "Two additional conditions run per task (5 trials each, 10 total):\n"
        "- noskill: Model receives only the task instruction, no SKILL.md content.\n"
        "- gtonly: Model receives only the GT skill(s), no noise distractors.\n\n"
        "These establish the lower bound (can the model solve it without help?) and "
        "the upper bound (does the correct skill actually help?)."
    )

    pdf.section("3.3 Failure Mode Classification (new in V2)", level=2)
    pdf.body(
        "Every FAIL trial is automatically classified into one of five categories:"
    )
    pdf.table(
        ["Mode", "Detection Rule", "Interpretation"],
        [
            ["TIMEOUT", "agent_wall_s >= 900", "Model too slow / stuck in loop"],
            ["PATH_ERROR_TRUE_PASS", "FileNotFoundError + correct answer in stdout",
             "FALSE NEGATIVE - fix needed"],
            ["PATH_ERROR", "FileNotFoundError, no correct answer detected", "Ambiguous"],
            ["FORMAT_MISMATCH", "Test ERRORS (not FAILURES)", "SKILL.md/test misalignment"],
            ["LOGIC_ERROR", "Test FAILURES (assertions failed)", "Genuine model error"],
        ],
        col_widths=[38, 52, 55],
    )
    pdf.body(
        "This classification runs in the cron monitoring job every 10 minutes. "
        "If 3 consecutive failures share the same non-LOGIC mode, it triggers a "
        "bug alert for investigation."
    )

    # ---- 4. Architecture Changes ----
    pdf.add_page()
    pdf.section("4. Architecture (V1 -> V2 Delta)")

    pdf.section("4.1 Trial Execution Flow", level=2)
    pdf.code_block(
        "V1 flow:                          V2 flow:\n"
        "                                  \n"
        "setup_workspace()                 setup_workspace()\n"
        "patch_tests()                     patch_tests()\n"
        "patch_instruction()               patch_instruction()\n"
        "build_system_prompt()             build_system_prompt()\n"
        "                                  write system_prompt to temp file  [NEW]\n"
        "claude -p --system-prompt ...     claude -p --system-prompt-file .. [CHANGED]\n"
        "                                  _reconcile_output_paths()         [NEW]\n"
        "run_tests()                       run_tests()\n"
        "                                  classify_failure_mode()           [NEW]\n"
        "emit JSONL                        emit JSONL"
    )

    pdf.section("4.2 Orchestration", level=2)
    pdf.code_block(
        "V1: docker compose up\n"
        "    -> python run_trial.py --concurrency 2\n"
        "    -> asyncio creates ALL coroutines\n"
        "    -> HANGS after 3-16 trials\n"
        "\n"
        "V2: bash run_sequential.sh\n"
        "    -> while loop\n"
        "       -> docker run --rm (fresh container per trial)\n"
        "          -> python run_trial.py --max-trials 1 --resume\n"
        "          -> exits after 1 trial\n"
        "       -> check progress, stall detection (3-strike abort)\n"
        "    -> aggregate results"
    )

    pdf.section("4.3 Monitoring (cron job)", level=2)
    pdf.body(
        "A 10-minute cron job automatically:\n"
        "1. Checks container status\n"
        "2. Reports PASS/FAIL/total progress\n"
        "3. Classifies failure modes for all FAIL trials\n"
        "4. Detects bug patterns (3+ consecutive same-type failures)\n"
        "5. Compares against V1 results"
    )

    # ---- 5. Known Limitations ----
    pdf.add_page()
    pdf.section("5. Known Limitations (Honest Assessment)")

    pdf.section("5.1 GT/Noise Content Asymmetry (NOT fixed in V2)", level=2)
    pdf.body(
        "GT skills receive full SKILL.md (often 1000+ words). Noise skills receive "
        "50-word corpus stubs. A sufficiently capable model can identify GT by content "
        "length alone, without semantic understanding.\n\n"
        "This confound affects noise-mode comparisons (hard vs easy vs random) but does "
        "NOT affect the baseline comparison (noskill vs gtonly vs prefill), because:\n"
        "- noskill has no SKILL.md at all\n"
        "- gtonly has one full document with nothing to compare against\n\n"
        "Fix for future work: symmetric content ablation where all candidates receive "
        "full SKILL.md content."
    )

    pdf.section("5.2 azure-bgp FORMAT_MISMATCH (NOT fixed in V2)", level=2)
    pdf.body(
        "The azure-bgp GT SKILL.md guides the model to produce output in a structure "
        "that the test suite's parametrized assertions cannot parse. This is a "
        "SkillsBench data quality issue, not a pipeline bug.\n\n"
        "V2 does NOT modify task data (SKILL.md, tests, instructions). The FORMAT_MISMATCH "
        "failures for azure-bgp will persist. They should be excluded from noise-mode "
        "analysis or treated as a separate data-quality finding."
    )

    pdf.section("5.3 Sample Size (N=5 per task)", level=2)
    pdf.body(
        "The smoke run uses 5 tasks. Individual task-level or cell-level comparisons "
        "are not statistically significant. The smoke run's purpose is to validate "
        "the pipeline and identify measurement bugs, not to draw conclusions.\n\n"
        "The full run (84 tasks x 12 conditions = 1,008 trials) is required for "
        "paper-grade statistical power."
    )

    pdf.section("5.4 Pool Size / Context Length Confound", level=2)
    pdf.body(
        "Pool size and system prompt length are perfectly correlated:\n"
        "- sz=5: ~5K chars\n"
        "- sz=50: ~35K chars\n\n"
        "Any pool-size effect may be driven by context length, not pool composition. "
        "Isolating this requires a 'random long text' baseline with matched token count "
        "but no procedural content."
    )

    # ---- 6. V2 Early Results ----
    pdf.section("6. V2 Early Results (In Progress)")
    pdf.body(
        "V2 re-run is in progress. Early results show dramatic improvement from the "
        "path reconciliation fix:"
    )
    pdf.table(
        ["Metric", "V1", "V2 (partial)"],
        [
            ["3d-scan-calc pass rate", "10/12 (83%)", "12/12 (100%)"],
            ["Overall pass rate (so far)", "26/60 (43%)", "15/16 (94%)"],
            ["PATH_ERROR_TRUE_PASS count", "11", "0"],
            ["False negative rate", "26.8% of failures", "0%"],
        ],
        col_widths=[55, 45, 45],
    )
    pdf.body(
        "The two 3d-scan-calc trials that failed in V1 due to path errors now pass in V2. "
        "The single V2 failure so far is a genuine TIMEOUT (adaptive-cruise-control sz5 "
        "random, 900s). No PATH_ERROR or FORMAT_MISMATCH bugs detected.\n\n"
        "V2 run is ongoing. Final results will be available after all 60 main trials + "
        "10 baseline trials complete (~4-6 hours from start)."
    )

    # ---- 7. Reviewer Response Plan ----
    pdf.add_page()
    pdf.section("7. Reviewer Concern Response Plan")

    pdf.table(
        ["Concern", "Priority", "Status in V2", "Evidence"],
        [
            ["Path rewriting false negatives", "P0", "FIXED",
             "_reconcile_output_paths()"],
            ["Failure mode classification", "P0", "FIXED",
             "Auto-classification in cron"],
            ["GT/test format alignment audit", "P0", "DOCUMENTED",
             "azure-bgp identified, not fixed"],
            ["Missing noskill/gtonly baselines", "P1", "IMPLEMENTED",
             "Code ready, V1 data collected"],
            ["GT/noise content asymmetry", "P1", "NOT FIXED",
             "Needs symmetric ablation exp"],
            ["N=5 sample size", "P2", "ACKNOWLEDGED",
             "Smoke run only; full grid needed"],
            ["Pool size / context length", "P2", "NOT FIXED",
             "Needs random-long-text baseline"],
        ],
        col_widths=[45, 18, 30, 55],
    )

    pdf.section("7.1 What V2 CAN Claim", level=2)
    pdf.bullet("The V1 pass rate was artificially depressed by infrastructure bugs "
               "(path rewriting, ARG_MAX, Dockerfile regex).")
    pdf.bullet("After fixing measurement errors, the true pass rate is significantly "
               "higher than V1's 43%.")
    pdf.bullet("PATH_ERROR_TRUE_PASS (false negatives) have been eliminated.")
    pdf.bullet("Failure mode classification enables clean separation of model failures "
               "from infrastructure failures.")

    pdf.section("7.2 What V2 CANNOT Claim (yet)", level=2)
    pdf.bullet("'Noise helps execution' -- requires symmetric content ablation and N>>5.")
    pdf.bullet("'Hard noise is best' -- confounded by GT/stub length asymmetry.")
    pdf.bullet("'Larger pools help' -- confounded by context length.")
    pdf.bullet("'GT-only hurts' -- partially driven by azure-bgp format defect.")

    # ---- 8. Recommended Next Experiments ----
    pdf.section("8. Recommended Experiments (Post-V2)")
    pdf.table(
        ["Experiment", "Purpose", "Trials", "Priority"],
        [
            ["Symmetric full-doc ablation", "Isolate length-matching confound",
             "~250", "P1"],
            ["Random long-text baseline", "Isolate context-length effect",
             "~60", "P1"],
            ["Multi-seed replication (0,1,2)", "Confidence intervals",
             "~180", "P1"],
            ["Full 84-task grid", "Statistical power",
             "1,008", "P2"],
            ["Failure mode deep-dive", "Classify FAIL stdout",
             "0 (analysis)", "P0"],
            ["azure-bgp SKILL.md fix", "Eliminate format mismatch",
             "~12", "P1"],
        ],
        col_widths=[50, 45, 20, 20],
    )

    # ---- 9. File Reference ----
    pdf.add_page()
    pdf.section("9. File Reference (V2 Additions)")
    pdf.table(
        ["File", "V1/V2", "Purpose"],
        [
            ["run_trial.py", "MODIFIED", "_reconcile_output_paths, --system-prompt-file, "
             "--max-trials, noskill/gtonly modes"],
            ["pool_builder_v3.py", "MODIFIED", "noskill + gtonly noise modes"],
            ["run_sequential.sh", "NEW", "One-trial-per-container bash orchestrator"],
            ["run_baselines.sh", "NEW", "noskill + gtonly baseline runner"],
            ["generate_report.py", "NEW", "Analysis report PDF generator"],
            ["generate_designdoc_v2.py", "NEW", "This document"],
            ["results/sb_exec.jsonl", "V2 DATA", "60-trial main experiment (in progress)"],
            ["results/sb_baselines.jsonl", "V2 DATA", "10-trial baselines"],
            ["results/sb_exec_v1.jsonl", "ARCHIVED", "V1 main experiment (43.3%)"],
            ["results/sb_baselines_v1.jsonl", "ARCHIVED", "V1 baselines"],
            ["results/analysis_report.pdf", "V1 REPORT", "V1 analysis (superseded)"],
        ],
        col_widths=[55, 25, 110],
    )

    return pdf


if __name__ == "__main__":
    pdf = build()
    out = "results/design_document_v2.pdf"
    pdf.output(out)
    print(f"PDF written to {out}")
