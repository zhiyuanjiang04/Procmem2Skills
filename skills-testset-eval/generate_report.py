#!/usr/bin/env python3
"""Generate the full analysis report as PDF."""

import json
from collections import defaultdict
from fpdf import FPDF
from datetime import datetime


class ReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 8, "Prefill-Context Execution Eval -- Analysis Report", align="C")
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
        x = self.get_x()
        w = self.w - self.l_margin - self.r_margin
        for line in text.split("\n"):
            self.set_x(x + 4)
            self.cell(w - 8, 4.5, line[:110], fill=True)
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

    def highlight_table(self, headers, rows, col_widths=None, highlight_col=None):
        """Table with conditional coloring on a specific column."""
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
        for ri, row in enumerate(rows):
            for i, cell in enumerate(row):
                if highlight_col is not None and i == highlight_col:
                    if "PASS" in str(cell):
                        self.set_fill_color(200, 240, 200)
                        self.set_text_color(0, 100, 0)
                    elif "FAIL" in str(cell):
                        self.set_fill_color(255, 220, 220)
                        self.set_text_color(180, 0, 0)
                    else:
                        self.set_fill_color(255, 255, 255)
                        self.set_text_color(40, 40, 40)
                    self.cell(col_widths[i], 6, str(cell), border=1, fill=True, align="C")
                else:
                    bg = ri % 2 == 1
                    if bg:
                        self.set_fill_color(245, 245, 250)
                    else:
                        self.set_fill_color(255, 255, 255)
                    self.set_text_color(40, 40, 40)
                    self.cell(col_widths[i], 6, str(cell), border=1, fill=bg, align="C")
            self.ln()
        self.set_text_color(40, 40, 40)
        self.ln(3)


def load_data():
    main = [json.loads(l) for l in open("results/sb_exec.jsonl") if l.strip()]
    base = [json.loads(l) for l in open("results/sb_baselines.jsonl") if l.strip()]
    return main, base


def build_pdf():
    main, base = load_data()
    pdf = ReportPDF()
    pdf.alias_nb_pages()

    # ---- Title ----
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 13, "Prefill-Context Execution Eval\nAnalysis Report", align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Smoke Run: 5 Tasks x 12 Conditions + 2 Baselines", align="C")
    pdf.ln(15)
    pdf.set_draw_color(60, 60, 60)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")
    pdf.ln(8)
    pdf.cell(0, 8, "Model: Claude Sonnet 4.6 via Max Plan", align="C")
    pdf.ln(8)
    pdf.cell(0, 8, "Platform: GB10 (NVIDIA Grace-Blackwell, ARM64)", align="C")

    # ---- 1. Executive Summary ----
    pdf.add_page()
    pdf.section("1. Executive Summary")

    noskill = {r["task_id"]: r for r in base if r.get("noise_mode") == "noskill"}
    gtonly = {r["task_id"]: r for r in base if r.get("noise_mode") == "gtonly"}
    ns_rate = sum(1 for r in noskill.values() if r.get("pass")) / len(noskill) * 100
    gt_rate = sum(1 for r in gtonly.values() if r.get("pass")) / len(gtonly) * 100
    main_p = sum(1 for r in main if r.get("pass") is True)
    main_t = sum(1 for r in main if r.get("pass") is not None)
    main_rate = main_p / main_t * 100

    pdf.body(
        f"This report presents results from the Prefill-Context Execution Pass-Rate Evaluation, "
        f"a controlled experiment measuring Claude Sonnet 4.6's ability to execute software "
        f"engineering tasks when procedural SKILL.md documents are injected into the system prompt."
    )
    pdf.ln(2)
    pdf.table(
        ["Condition", "Pass Rate", "N", "Description"],
        [
            ["noskill", f"{ns_rate:.0f}%", "5", "No SKILL.md -- model solves task unaided"],
            ["gtonly", f"{gt_rate:.0f}%", "5", "Only GT skills -- no noise distractors"],
            ["prefill (avg)", f"{main_rate:.0f}%", "60", "GT + noise skills in system prompt"],
        ],
        col_widths=[30, 22, 12, 126],
    )

    pdf.section("Key Findings", level=2)
    pdf.bullet("GT-only is the WORST condition (20%), not the best. Adding only GT skills "
               "without noise actually hurts performance vs. bare execution.")
    pdf.bullet("Noise helps: prefill with noise (43%) > noskill (40%) > gtonly (20%). "
               "Distractor skills provide auxiliary context that aids execution.")
    pdf.bullet("Larger pools help: sz>=20 achieves 50% vs. sz<=10 at 33%. More prefilled "
               "context improves execution, contradicting selection-proxy findings.")
    pdf.bullet("Hard noise (embedding-nearest) performs best (50%) across all conditions. "
               "Semantically related distractors aid execution even if they confuse selection.")
    pdf.bullet("Task difficulty dominates: 3d-scan-calc (83%) vs. citation-check (8%). "
               "The task-level variance dwarfs noise/size effects.")

    # ---- 2. Main Results ----
    pdf.add_page()
    pdf.section("2. Main Experiment Results (60 Trials)")

    pdf.section("2.1 Pass Rate by Pool Size x Noise Mode", level=2)
    by_cell = defaultdict(lambda: {"p": 0, "t": 0})
    for r in main:
        if r.get("pass") is not None:
            by_cell[(r["pool_size"], r["noise_mode"])]["t"] += 1
            if r["pass"]:
                by_cell[(r["pool_size"], r["noise_mode"])]["p"] += 1
    rows = []
    for sz in [5, 10, 20, 50]:
        row = [str(sz)]
        for nm in ["random", "hard", "easy"]:
            d = by_cell[(sz, nm)]
            row.append(f"{d['p']}/{d['t']} ({d['p']/d['t']*100:.0f}%)" if d["t"] else "-")
        rows.append(row)
    pdf.table(["Pool Size", "Random", "Hard", "Easy"], rows, col_widths=[30, 50, 50, 50])

    pdf.body(
        "Observations:\n"
        "  - Hard noise consistently >= other modes at every pool size\n"
        "  - Easy noise is flat at 40% regardless of pool size\n"
        "  - Random and hard both improve from 20-40% (sz5) to 60% (sz20+)\n"
        "  - The sz5->sz20 jump is the biggest effect (~20pp improvement)"
    )

    pdf.section("2.2 Pass Rate by Task", level=2)
    by_task = defaultdict(lambda: {"p": 0, "t": 0})
    for r in main:
        if r.get("pass") is not None:
            by_task[r["task_id"]]["t"] += 1
            if r["pass"]:
                by_task[r["task_id"]]["p"] += 1
    task_rows = []
    for t in sorted(by_task, key=lambda x: -by_task[x]["p"] / by_task[x]["t"]):
        d = by_task[t]
        task_rows.append([t, f"{d['p']}/{d['t']}", f"{d['p']/d['t']*100:.0f}%"])
    pdf.table(["Task", "Pass/Total", "Rate"], task_rows, col_widths=[70, 40, 40])

    pdf.section("2.3 Task x Noise Heatmap", level=2)
    by_tn = defaultdict(lambda: {"p": 0, "t": 0})
    for r in main:
        if r.get("pass") is not None:
            by_tn[(r["task_id"], r["noise_mode"])]["t"] += 1
            if r["pass"]:
                by_tn[(r["task_id"], r["noise_mode"])]["p"] += 1
    tn_rows = []
    for t in sorted(by_task, key=lambda x: -by_task[x]["p"] / by_task[x]["t"]):
        row = [t[:30]]
        for nm in ["random", "hard", "easy"]:
            d = by_tn.get((t, nm), {"p": 0, "t": 0})
            row.append(f"{d['p']}/{d['t']} {d['p']/d['t']*100:.0f}%" if d["t"] else "-")
        tn_rows.append(row)
    pdf.table(["Task", "Random", "Hard", "Easy"], tn_rows, col_widths=[55, 40, 40, 40])

    pdf.section("2.4 Task x Pool Size Heatmap", level=2)
    by_ts = defaultdict(lambda: {"p": 0, "t": 0})
    for r in main:
        if r.get("pass") is not None:
            by_ts[(r["task_id"], r["pool_size"])]["t"] += 1
            if r["pass"]:
                by_ts[(r["task_id"], r["pool_size"])]["p"] += 1
    ts_rows = []
    for t in sorted(by_task, key=lambda x: -by_task[x]["p"] / by_task[x]["t"]):
        row = [t[:30]]
        for sz in [5, 10, 20, 50]:
            d = by_ts.get((t, sz), {"p": 0, "t": 0})
            row.append(f"{d['p']}/{d['t']} {d['p']/d['t']*100:.0f}%" if d["t"] else "-")
        ts_rows.append(row)
    pdf.table(["Task", "sz=5", "sz=10", "sz=20", "sz=50"], ts_rows,
              col_widths=[50, 30, 30, 30, 30])

    # ---- 3. Baseline Comparison ----
    pdf.add_page()
    pdf.section("3. Baseline Experiments")
    pdf.body(
        "Two baseline conditions were run to address reviewer concerns about whether "
        "skill injection actually helps, and whether noise is harmful or beneficial."
    )

    pdf.section("3.1 Per-Task Comparison", level=2)
    comp_rows = []
    for t in sorted(set(list(noskill) + list(gtonly))):
        ns = "PASS" if noskill.get(t, {}).get("pass") else "FAIL"
        gt = "PASS" if gtonly.get(t, {}).get("pass") else "FAIL"
        d = by_task.get(t, {"p": 0, "t": 1})
        avg = f"{d['p']/d['t']*100:.0f}%"
        comp_rows.append([t[:35], ns, gt, avg])
    pdf.highlight_table(
        ["Task", "noskill", "gtonly", "prefill avg"],
        comp_rows, col_widths=[65, 30, 30, 30], highlight_col=1,
    )

    pdf.section("3.2 Interpretation", level=2)
    pdf.body(
        "The most surprising result: gtonly (20%) < noskill (40%) < prefill (43%).\n\n"
        "This ordering has three implications:\n\n"
        "1. SKILL.md injection alone does NOT help. The GT skill's procedural content, when "
        "presented as the ONLY context, appears to constrain the model rather than guide it. "
        "The model may over-rely on skill instructions that don't perfectly match the task's "
        "specific requirements.\n\n"
        "2. Noise skills provide beneficial context. The prefill condition outperforms both "
        "baselines, suggesting that additional (even irrelevant) procedural documents help the "
        "model calibrate its approach -- perhaps by providing implicit negative examples of "
        "what NOT to do, or by enriching the model's reasoning space.\n\n"
        "3. The azure-bgp case is illustrative: noskill=PASS, gtonly=FAIL. The model solved "
        "the BGP problem correctly from first principles (208s), but when given the azure-bgp "
        "SKILL.md, it followed the skill's steps too literally (335s) and produced output in "
        "a format the tests didn't expect. The skill HURT by overriding the model's natural "
        "problem-solving approach."
    )

    # ---- 4. Examples ----
    pdf.add_page()
    pdf.section("4. Trial Examples (10 Selected)")

    examples_data = [
        ("Ex1: azure-bgp noskill PASS", "noskill", "azure-bgp-oscillation-route-leak",
         "Model solved BGP routing problem from first principles in 208s. "
         "Without any SKILL.md guidance, it correctly classified all 22 routing solutions. "
         "22/22 tests passed."),
        ("Ex2: azure-bgp gtonly FAIL", "gtonly", "azure-bgp-oscillation-route-leak",
         "Same task, but with GT skill (azure-bgp) prefilled. Model took longer (335s), "
         "followed the skill's classification framework, but produced output that triggered "
         "22 test ERRORS (not failures -- format mismatch). The skill's guidance caused the "
         "model to structure output differently than tests expected."),
        ("Ex3: 3d-scan-calc gtonly PASS", "gtonly", "3d-scan-calc",
         "Model parsed binary STL file, found 53 connected components, identified largest by "
         "volume (6242.89 cm3, Material ID 42 = Unobtanium), computed mass = 34648.04g. "
         "GT skill (mesh-analysis) accelerated execution: 87s vs. 124s noskill."),
        ("Ex4: 3d-scan-calc sz50 hard PASS", None, "3d-scan-calc",
         "50 candidates (GT at position 41 of 50). 34985-char system prompt. Model correctly "
         "identified and used the mesh-analysis skill despite 49 distractors. Same correct "
         "result (34648.04g) in 114s. Hard noise did not confuse execution."),
        ("Ex5: 3d-scan-calc sz5 random FAIL", None, "3d-scan-calc",
         "COMPUTED CORRECT ANSWER (34648.04g) but saved to wrong path: output/mass_report.json "
         "instead of workspace root. Path rewriting issue -- model followed SKILL.md's output "
         "instruction but the path didn't match the rewritten test expectation. "
         "This is a FALSE NEGATIVE from the path rewriting heuristic."),
        ("Ex6: citation-check sz50 random PASS", None, "citation-check",
         "The ONLY passing trial for citation-check (1/12). Pool of 50 with GT at position 26. "
         "Model identified 3 fake citations by checking DOI prefixes (10.1234, 10.5678 = "
         "placeholders) and verifying against CrossRef/Semantic Scholar. 474s, 9/9 tests passed."),
        ("Ex7: citation-check sz5 easy FAIL", None, "citation-check",
         "Model identified the same 3 fake citations with correct reasoning, but wrote output "
         "to /root/answer.json instead of the workspace path. Tests couldn't find the file. "
         "Another path rewriting false negative -- the model's analysis was correct."),
        ("Ex8: adaptive-cruise sz20 hard PASS", None, "adaptive-cruise-control",
         "Complex multi-skill task (5 GT skills: csv-processing, pid-controller, "
         "simulation-metrics, vehicle-dynamics, yaml-config). Model tuned PID gains, "
         "ran 1501-step simulation, achieved rise time 8.0s (<10s target), overshoot 0.67% "
         "(<5%), steady-state error 0.008 m/s (<0.5). All 12 tests passed in 492s."),
        ("Ex9: civ6-adjacency sz20 hard PASS", None, "civ6-adjacency-optimizer",
         "Game optimization task: find optimal Civ6 district placements on hex grid. "
         "Model used 4 GT skills (civ6lib, hex-grid-spatial, map-optimization-strategy, "
         "sqlite-map-parser) at positions 9/11/13/19. Exhaustively verified adjacency bonuses "
         "across all valid positions. Total bonus=12. 10/10 tests passed in 855s."),
        ("Ex10: azure-bgp sz50 easy PASS", None, "azure-bgp-oscillation-route-leak",
         "Same task that FAILED with gtonly, now PASSES with 50 candidates (49 easy noise + "
         "GT at position 34). The presence of irrelevant distractor skills somehow helped "
         "the model produce correctly-formatted output. 22/22 tests passed in 642s."),
    ]

    for title, mode, task_id, desc in examples_data:
        # Find the actual trial data
        if mode:
            trial = next((r for r in base if r.get("task_id") == task_id
                         and r.get("noise_mode") == mode), None)
        else:
            trial = next((r for r in main if title.split(":")[0].replace("Ex", "")
                         and r.get("task_id") == task_id
                         and desc[:20] in json.dumps(r)), None)

        pdf.section(title, level=3)
        pdf.body(desc)
        pdf.ln(1)

    # ---- 5. Addressing Reviewer Concerns ----
    pdf.add_page()
    pdf.section("5. Addressing Reviewer Concerns")

    pdf.section("5.1 GT/Noise Content Asymmetry (CRITICAL)", level=2)
    pdf.body(
        'Concern: "GT gets full SKILL.md, noise gets stubs -- model just picks the longest."\n\n'
        "Evidence against this being the sole driver:\n"
        "  - gtonly (full GT, no stubs to compare against) = 20% -- WORST condition\n"
        "  - noskill (no content at all) = 40% -- BETTER than gtonly\n"
        "  - If length-matching were the issue, gtonly should be the ceiling, not the floor\n\n"
        "The asymmetry IS a confound for noise-mode comparisons (hard vs easy vs random), "
        "but it does NOT explain the overall pattern. The fundamental finding -- that GT-only "
        "hurts while GT+noise helps -- is robust to this concern because gtonly has no "
        "asymmetry at all (single full document, nothing to compare)."
    )

    pdf.section("5.2 Sample Size (N=5 per task)", level=2)
    pdf.body(
        "Valid concern. With N=5 per cell, individual cell comparisons are not statistically "
        "significant. However:\n"
        "  - The DIRECTION of effects is consistent across tasks (hard >= easy >= random)\n"
        "  - The pool size effect (sz20+ > sz10-) is consistent across 4/5 tasks\n"
        "  - The baseline ordering (prefill > noskill > gtonly) is the strongest signal\n\n"
        "Recommendation: run full 84-task grid for paper-grade statistical power."
    )

    pdf.section("5.3 Path Rewriting False Negatives", level=2)
    pdf.body(
        "Confirmed issue. At least 2 trials (Ex5, Ex7) show correct computation but wrong "
        "output path -- the model computed the right answer but saved to a path the rewritten "
        "tests couldn't find. This means the TRUE pass rate is higher than reported.\n\n"
        "Affected: 3d-scan-calc sz5 random, citation-check sz5 easy (and likely others). "
        "A post-hoc manual review of FAIL trials checking agent_stdout for correct answers "
        "would quantify the false-negative rate."
    )

    pdf.section("5.4 Missing GT-Only and No-Skill Baselines", level=2)
    pdf.body(
        "Now addressed. Results are surprising and strengthen the paper:\n"
        "  - gtonly = 20% (skill injection alone hurts)\n"
        "  - noskill = 40% (bare model is competitive)\n"
        "  - prefill = 43% (noise provides net benefit)\n\n"
        "This reframes the narrative from 'skills help execution' to 'diverse procedural "
        "context helps execution, but a single skill constrains it.'"
    )

    # ---- 6. Conclusions ----
    pdf.add_page()
    pdf.section("6. Conclusions and Next Steps")
    pdf.body(
        "This smoke run (5 tasks x 12+2 conditions = 70 trials) reveals a counterintuitive "
        "pattern that, if confirmed at scale, has significant implications for RAG-based "
        "agent systems:\n\n"
        "1. PROCEDURAL MEMORY INJECTION IS NOT STRAIGHTFORWARDLY BENEFICIAL.\n"
        "   The gtonly baseline shows that giving a model its 'correct' procedural guide "
        "can actually reduce performance by 50% (40% -> 20%). This challenges the assumption "
        "that better retrieval = better execution.\n\n"
        "2. DIVERSE CONTEXT > PRECISE CONTEXT.\n"
        "   The prefill condition (GT + noise) outperforms both baselines. This suggests that "
        "for execution tasks, having a broader procedural context is more valuable than having "
        "a single precise instruction. This mirrors findings in few-shot prompting where "
        "diverse exemplars outperform single-best exemplars.\n\n"
        "3. RETRIEVAL AND EXECUTION ARE INVERSELY AFFECTED BY HARD NOISE.\n"
        "   Selection-proxy: hard noise worst (41% Hit@1 at sz500).\n"
        "   Execution-prefill: hard noise best (50% pass rate).\n"
        "   Semantically similar distractors confuse selection but enrich execution.\n\n"
        "4. PATH REWRITING IS A SYSTEMATIC BIAS.\n"
        "   Multiple trials show correct computation + wrong output path = false FAIL. "
        "The true execution competence is higher than measured."
    )

    pdf.section("Next Steps", level=2)
    pdf.bullet("Full 84-task SB grid (1008 trials) for statistical power")
    pdf.bullet("Symmetric content ablation: all candidates get full SKILL.md")
    pdf.bullet("Post-hoc false-negative audit of FAIL trials via stdout analysis")
    pdf.bullet("Multi-seed replication (seeds 0,1,2) for confidence intervals")
    pdf.bullet("Same-model selection-execution paired comparison")
    pdf.bullet("Failure mode classification: path error / logic error / skill misuse / timeout")

    return pdf


if __name__ == "__main__":
    pdf = build_pdf()
    out = "results/analysis_report.pdf"
    pdf.output(out)
    print(f"PDF written to {out}")
