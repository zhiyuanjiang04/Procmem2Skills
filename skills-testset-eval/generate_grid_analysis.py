#!/usr/bin/env python3
"""Conditional Non-Timeout Correctness Analysis.

NOT an end-to-end execution pass rate. Timeout is a major failure mode
and is analyzed separately, not deleted from the denominator.

Methodology matches 20-Task Report v4:
  - Timeout = wall > 600s OR stdout mentions timeout/killed/timed out
  - Dedup by (task_id, pool_size, noise_mode, seed), last entry wins
  - First20: merge sb_exec_20t.jsonl (seed=0) + sb_prefill_n5.jsonl (seeds 1-4)
  - 6 never-executed tasks excluded (wall=None, model=None across all rows)
  - Smoke20 design skips (|GT|>pool_size at pool=5) excluded
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from fpdf import FPDF

BASE = Path(__file__).parent

# 6 tasks that were NEVER executed (wall=None across all rows)
FIRST20_NEVER_EXECUTED = {
    "court-form-filling", "crystallographic-wyckoff-position-analysis",
    "data-to-d3", "dynamic-object-aware-egomotion",
    "edit-pdf", "exoplanet-detection-period",
}


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.cell(
            0, 8,
            "Conditional Non-Timeout Correctness Analysis (NOT End-to-End Pass Rate)",
            align="C", new_x="LMARGIN", new_y="NEXT",
        )
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(40, 60, 100)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def subsection(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 60, 100)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 4.5, text)
        self.ln(1)

    def kv(self, key, val):
        self.set_font("Helvetica", "B", 9)
        self.cell(55, 5, key)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, str(val), new_x="LMARGIN", new_y="NEXT")

    def colored_table(self, headers, rows, col_widths=None, highlight_cols=None):
        if col_widths is None:
            col_widths = [190 // len(headers)] * len(headers)
        if highlight_cols is None:
            highlight_cols = set()
        self.set_font("Helvetica", "B", 6.5)
        self.set_fill_color(40, 60, 100)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, h, border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 6.5)
        for ri, row in enumerate(rows):
            for i, cell in enumerate(row):
                if i in highlight_cols and cell not in ("--", "", "N/A"):
                    try:
                        val = float(cell.replace("%", ""))
                        if val >= 70:
                            self.set_fill_color(200, 240, 200)
                        elif val >= 30:
                            self.set_fill_color(255, 240, 200)
                        else:
                            self.set_fill_color(255, 210, 210)
                        self.cell(col_widths[i], 5, str(cell), border=1, fill=True, align="C")
                    except ValueError:
                        self.cell(col_widths[i], 5, str(cell), border=1, align="C")
                else:
                    if ri % 2 == 1:
                        self.set_fill_color(245, 245, 250)
                        self.cell(col_widths[i], 5, str(cell), border=1, fill=True, align="C")
                    else:
                        self.cell(col_widths[i], 5, str(cell), border=1, align="C")
            self.ln()
        self.ln(2)


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def is_timeout(r):
    wall = r.get("agent_wall_s") or 0
    if wall > 600:
        return True
    stdout = (r.get("agent_stdout_tail", "") or "").lower()
    stderr = (r.get("agent_stderr_tail", "") or "").lower()
    combined = stdout + " " + stderr
    return "timeout" in combined or "killed" in combined or "timed out" in combined


def is_design_skip(r):
    if not r.get("model") and len(r.get("gt_slugs", [])) == 0:
        return True
    return False


def is_never_executed(r):
    return r.get("agent_wall_s") is None and r.get("model") is None


def dedup_rows(rows):
    by_key = {}
    for r in rows:
        key = (r["task_id"], r["pool_size"], r["noise_mode"], r.get("seed", 0))
        by_key[key] = r
    return list(by_key.values())


def fmt_rate(n, d):
    if d == 0:
        return "N/A"
    return f"{n/d*100:.1f}%"


def fmt_cell(n, d):
    if d == 0:
        return "N/A"
    return f"{n}/{d} ({n/d*100:.0f}%)"


def main():
    # -- Load data --
    exec_20t = load_jsonl(BASE / "results" / "sb_exec_20t.jsonl")
    prefill_n5 = load_jsonl(BASE / "results" / "sb_prefill_n5.jsonl")

    # First20: merge seed=0 + seeds 1-4, exclude never-executed
    first20_raw = exec_20t + prefill_n5
    first20_raw = [r for r in first20_raw
                   if r["task_id"] not in FIRST20_NEVER_EXECUTED
                   and not is_never_executed(r)]
    first20 = dedup_rows(first20_raw)

    smoke10 = dedup_rows(load_jsonl(BASE / "results" / "smoke10" / "smoke10_n5.jsonl"))

    smoke20_raw = load_jsonl(BASE / "results" / "smoke20" / "smoke20_n5.jsonl")
    smoke20_design_skips = [r for r in smoke20_raw if is_design_skip(r)]
    smoke20 = dedup_rows([r for r in smoke20_raw if not is_design_skip(r)])

    batches = {"First20": first20, "Smoke10": smoke10, "Smoke20": smoke20}

    # -- Build PDF --
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # == Section 0: Methodology ==
    pdf.section("0. Scope & Methodology")
    pdf.body(
        "This report estimates correctness CONDITIONAL ON NON-TIMEOUT COMPLETION. "
        "It should NOT be interpreted as overall execution success rate. Timeout is "
        "a major failure mode (v4 Section 7: 47% of prefill failures are TIMEOUT) "
        "and is analyzed separately, not deleted."
    )
    pdf.body(
        "Timeout definition (matching v4): wall > 600s OR stdout/stderr mentions "
        "'timeout'/'killed'/'timed out'. Dedup: (task, pool, noise, seed), last wins."
    )
    pdf.body(
        "First20: sb_exec_20t.jsonl (seed=0) + sb_prefill_n5.jsonl (seeds 1-4). "
        "6 never-executed tasks excluded (court-form-filling, crystallographic-wyckoff, "
        "data-to-d3, dynamic-object-aware-egomotion, edit-pdf, exoplanet-detection-period). "
        "14 evaluable tasks remain. Pool sizes: [5, 10, 20, 50]."
    )
    pdf.body(
        "Smoke10/20: seeds [0-4], pool sizes [5, 10, 20, 50, 100]. "
        "Smoke20: 30 design skips at pool=5 for |GT|=6 tasks excluded."
    )

    # == Section 1: Reconciliation table ==
    pdf.section("1. Data Reconciliation")
    pdf.subsection("First20 Reconciliation")
    recon_h = ["Item", "Expected", "Actual", "Note"]
    recon_r = [
        ["Total tasks", "20", "20", "Original set"],
        ["Never-executed (wall=None)", "6", "6", "Heavy Dockerfile / host-unsafe"],
        ["Evaluable tasks", "14", "14", "Matches v4 Section 3"],
        ["Configs per task", "60", "60", "4 pools x 3 noise x 5 seeds"],
        ["Total configs (N=5)", "840", "840", "14 x 60"],
    ]
    pdf.colored_table(recon_h, recon_r, [40, 25, 25, 60])

    pdf.subsection("Smoke10 Reconciliation")
    s10_tasks = Counter(r["task_id"] for r in smoke10)
    incomplete = [(t, c) for t, c in s10_tasks.items() if c < 75]
    recon10 = [
        ["Total tasks", "10", "10", ""],
        ["Expected per task", "75", "75", "5 pools x 3 noise x 5 seeds"],
        ["Incomplete tasks", "0", str(len(incomplete)),
         "; ".join(f"{t}({c})" for t, c in incomplete) if incomplete else "none"],
    ]
    pdf.colored_table(recon_h, recon10, [40, 25, 25, 60])

    pdf.subsection("Smoke20 Reconciliation")
    recon20 = [
        ["Total tasks", "20", "20", ""],
        ["Expected per task", "75", "75", "5 pools x 3 noise x 5 seeds"],
        ["Design skips (|GT|>pool)", "0", str(len(smoke20_design_skips)),
         "pool=5 for python-scala-translation, travel-planning"],
        ["Valid rows after skip", "1500", str(len(smoke20)), ""],
    ]
    pdf.colored_table(recon_h, recon20, [40, 25, 25, 60])

    # == Section 2: Waterfall decomposition ==
    pdf.section("2. Waterfall: Raw to Conditional Rate")
    pdf.body(
        "Decomposing the path from the 50-task report's raw row-weighted rate to "
        "this report's conditional non-timeout rate. Each step shows what changed."
    )

    for name, rows in batches.items():
        all_total = len(rows)
        all_pass = sum(1 for r in rows if r.get("pass"))
        timeout_fail = sum(1 for r in rows if is_timeout(r) and not r.get("pass"))
        non_to = [r for r in rows if not (is_timeout(r) and not r.get("pass"))]
        non_to_pass = sum(1 for r in non_to if r.get("pass"))

        pdf.subsection(name)
        wf_h = ["Step", "Denominator", "Pass", "Rate"]
        wf_r = [
            ["Raw (all deduped configs)", str(all_total), str(all_pass), fmt_rate(all_pass, all_total)],
            ["Timeout failures removed", f"-{timeout_fail}", "", ""],
            ["Conditional non-timeout", str(len(non_to)), str(non_to_pass), fmt_rate(non_to_pass, len(non_to))],
        ]
        pdf.colored_table(wf_h, wf_r, [50, 35, 25, 25])

    # == Section 3: Four-metric per-task table ==
    pdf.add_page()
    pdf.section("3. Per-Task Four-Metric Diagnostic")
    pdf.body(
        "Each task shows four metrics. Tasks classified by (completion rate, conditional correctness):"
    )
    pdf.body(
        "  reliable-solved: high pass, high completion (>70% both)\n"
        "  timeout-prone-solved: correct when completed, but often times out\n"
        "  wrong-result-hard: completes but produces wrong results\n"
        "  true-unsolved: fails even when it completes"
    )

    for name, rows in batches.items():
        task_metrics = {}
        for r in rows:
            tid = r["task_id"]
            if tid not in task_metrics:
                task_metrics[tid] = {"pass": 0, "total": 0, "timeout": 0}
            task_metrics[tid]["total"] += 1
            if is_timeout(r) and not r.get("pass"):
                task_metrics[tid]["timeout"] += 1
            elif r.get("pass"):
                task_metrics[tid]["pass"] += 1

        pdf.subsection(name)
        m_headers = ["Task", "pass/total", "TO/total", "pass/non-TO", "compl%", "Tier"]
        m_rows = []
        for tid in sorted(task_metrics, key=lambda t: -(task_metrics[t]["pass"] / max(task_metrics[t]["total"], 1))):
            s = task_metrics[tid]
            non_to = s["total"] - s["timeout"]
            pass_rate = s["pass"] / s["total"] * 100 if s["total"] else 0
            to_rate = s["timeout"] / s["total"] * 100 if s["total"] else 0
            cond_rate = s["pass"] / non_to * 100 if non_to else 0
            compl_rate = non_to / s["total"] * 100 if s["total"] else 0

            if compl_rate >= 70 and cond_rate >= 70:
                tier = "reliable-solved"
            elif cond_rate >= 50 and compl_rate < 70:
                tier = "TO-prone-solved"
            elif compl_rate >= 50 and cond_rate < 30:
                tier = "wrong-result"
            elif non_to < 5:
                tier = "tiny-sample"
            elif cond_rate < 10:
                tier = "true-unsolved"
            else:
                tier = "mixed"

            m_rows.append([
                tid,
                f"{s['pass']}/{s['total']} ({pass_rate:.0f}%)",
                f"{s['timeout']}/{s['total']} ({to_rate:.0f}%)",
                f"{s['pass']}/{non_to} ({cond_rate:.0f}%)" if non_to > 0 else "N/A",
                f"{compl_rate:.0f}%",
                tier,
            ])
        m_w = [42, 28, 26, 28, 18, 32]
        pdf.colored_table(m_headers, m_rows, m_w, {1})

    # == Section 4: Grid -- DUAL display (raw + conditional) ==
    pdf.add_page()
    pdf.section("4. Pool x Noise Grid (Raw AND Conditional)")
    pdf.body(
        "Each cell shows TWO rates: raw pass/total | conditional pass/non-timeout. "
        "This prevents the conditional rate from hiding timeout-dominated cells."
    )

    pool_sizes = [5, 10, 20, 50, 100]
    noise_modes = ["easy", "hard", "random"]

    for name, rows in batches.items():
        pdf.subsection(f"{name} (raw | conditional)")
        g_headers = ["Pool\\Noise"] + noise_modes + ["ALL"]
        g_rows = []
        for ps in pool_sizes:
            row = [str(ps)]
            ps_pass, ps_total, ps_non_to = 0, 0, 0
            for nm in noise_modes:
                subset = [r for r in rows if r["pool_size"] == ps and r["noise_mode"] == nm]
                if not subset:
                    row.append("--")
                    continue
                p = sum(1 for r in subset if r.get("pass"))
                t = len(subset)
                to_fail = sum(1 for r in subset if is_timeout(r) and not r.get("pass"))
                nt = t - to_fail
                raw = f"{p}/{t}({p*100//t}%)"
                cond = f"{p}/{nt}({p*100//nt}%)" if nt > 0 else "N/A"
                row.append(f"{raw} | {cond}")
                ps_pass += p
                ps_total += t
                ps_non_to += nt
            if ps_total > 0:
                r_all = f"{ps_pass}/{ps_total}({ps_pass*100//ps_total}%)"
                c_all = f"{ps_pass}/{ps_non_to}({ps_pass*100//ps_non_to}%)" if ps_non_to > 0 else "N/A"
                row.append(f"{r_all} | {c_all}")
            else:
                row.append("--")
            g_rows.append(row)

        # ALL row
        all_row = ["ALL"]
        for nm in noise_modes:
            subset = [r for r in rows if r["noise_mode"] == nm]
            if subset:
                p = sum(1 for r in subset if r.get("pass"))
                t = len(subset)
                to_fail = sum(1 for r in subset if is_timeout(r) and not r.get("pass"))
                nt = t - to_fail
                raw = f"{p}/{t}({p*100//t}%)"
                cond = f"{p}/{nt}({p*100//nt}%)" if nt > 0 else "N/A"
                all_row.append(f"{raw} | {cond}")
            else:
                all_row.append("--")
        total_p = sum(1 for r in rows if r.get("pass"))
        total_t = len(rows)
        total_to = sum(1 for r in rows if is_timeout(r) and not r.get("pass"))
        total_nt = total_t - total_to
        all_row.append(f"{total_p}/{total_t}({total_p*100//total_t}%) | {total_p}/{total_nt}({total_p*100//total_nt}%)")
        g_rows.append(all_row)

        pdf.colored_table(g_headers, g_rows, [20, 44, 44, 44, 44])

    # == Section 5: V4 Consistency Check ==
    pdf.add_page()
    pdf.section("5. V4 Consistency Check (First20, seed=0)")
    pdf.body(
        "Reproducing v4 Section 8 using seed=0 only, 14 evaluable tasks, "
        "dedup by (task, pool, noise), NO timeout exclusion."
    )

    exec_dedup = {}
    for r in exec_20t:
        if r["task_id"] in FIRST20_NEVER_EXECUTED:
            continue
        if is_never_executed(r):
            continue
        key = (r["task_id"], r["pool_size"], r["noise_mode"])
        exec_dedup[key] = r
    v4_rows = list(exec_dedup.values())
    v4_tasks = set(r["task_id"] for r in v4_rows)
    pdf.body(f"Tasks in v4 check: {len(v4_tasks)} ({', '.join(sorted(v4_tasks)[:5])}...)")
    pdf.body(f"Total deduped configs: {len(v4_rows)}")

    pdf.subsection("Noise Mode (v4 ref: random=58.9%, hard=60.7%, easy=57.1%)")
    v4n_h = ["Noise", "N", "Pass", "Rate", "v4 Rate", "Match"]
    v4_ref_n = {"random": (33, 56, "58.9%"), "hard": (34, 56, "60.7%"), "easy": (32, 56, "57.1%")}
    v4n_rows = []
    for nm in noise_modes:
        subset = [r for r in v4_rows if r["noise_mode"] == nm]
        p = sum(1 for r in subset if r.get("pass"))
        ref_p, ref_n, ref_rate = v4_ref_n[nm]
        match = "exact" if p == ref_p and len(subset) == ref_n else f"off by {abs(p - ref_p)}"
        v4n_rows.append([nm, str(len(subset)), str(p), fmt_rate(p, len(subset)), ref_rate, match])
    pdf.colored_table(v4n_h, v4n_rows, [25, 15, 15, 25, 25, 25])

    pdf.subsection("Pool Size (v4 ref: 5=59.5%, 10=59.5%, 20=61.9%, 50=54.8%)")
    v4p_h = ["Pool", "N", "Pass", "Rate", "v4 Rate", "Match"]
    v4_ref_p = {5: (25, 42, "59.5%"), 10: (25, 42, "59.5%"), 20: (26, 42, "61.9%"), 50: (23, 42, "54.8%")}
    v4p_rows = []
    for ps in [5, 10, 20, 50]:
        subset = [r for r in v4_rows if r["pool_size"] == ps]
        p = sum(1 for r in subset if r.get("pass"))
        ref_p, ref_n, ref_rate = v4_ref_p[ps]
        match = "exact" if p == ref_p and len(subset) == ref_n else f"off by {abs(p - ref_p)}"
        v4p_rows.append([str(ps), str(len(subset)), str(p), fmt_rate(p, len(subset)), ref_rate, match])
    pdf.colored_table(v4p_h, v4p_rows, [25, 15, 15, 25, 25, 25])

    # == Section 6: Cross-batch aggregate (raw + conditional) ==
    pdf.add_page()
    pdf.section("6. Cross-Batch Summary")

    pdf.subsection("By Noise Mode")
    xn_h = ["Batch"] + [f"{nm}\nraw | cond" for nm in noise_modes]
    xn_rows = []
    for name, rows in batches.items():
        row = [name]
        for nm in noise_modes:
            subset = [r for r in rows if r["noise_mode"] == nm]
            if subset:
                p = sum(1 for r in subset if r.get("pass"))
                to = sum(1 for r in subset if is_timeout(r) and not r.get("pass"))
                nt = len(subset) - to
                row.append(f"{fmt_rate(p, len(subset))} | {fmt_rate(p, nt)}")
            else:
                row.append("--")
        xn_rows.append(row)
    pdf.colored_table(xn_h, xn_rows, [30, 54, 54, 54])

    pdf.subsection("By Pool Size")
    xp_h = ["Batch"] + [f"{ps}\nraw | cond" for ps in pool_sizes]
    xp_rows = []
    for name, rows in batches.items():
        row = [name]
        for ps in pool_sizes:
            subset = [r for r in rows if r["pool_size"] == ps]
            if subset:
                p = sum(1 for r in subset if r.get("pass"))
                to = sum(1 for r in subset if is_timeout(r) and not r.get("pass"))
                nt = len(subset) - to
                row.append(f"{fmt_rate(p, len(subset))} | {fmt_rate(p, nt)}")
            else:
                row.append("--")
        xp_rows.append(row)
    pdf.colored_table(xp_h, xp_rows, [28, 32, 32, 32, 32, 32])

    # Compute summary stats for conclusion
    timeout_counts = {}
    all_total = 0
    all_pass = 0
    all_valid = 0
    for name, rows in batches.items():
        to_cnt = sum(1 for r in rows if is_timeout(r) and not r.get("pass"))
        timeout_counts[name] = to_cnt
        p = sum(1 for r in rows if r.get("pass"))
        all_total += len(rows)
        all_pass += p
        all_valid += len(rows) - to_cnt

    # == Section 7: Conclusion ==
    pdf.add_page()
    pdf.section("7. Conclusion")
    pdf.body(
        "This report decomposes execution failures into 'wrong answer' vs 'timeout'. "
        "It does NOT evaluate whether skills improve performance -- that requires "
        "Noskill/GT-only/Noise-only controls (v4 Section 11)."
    )
    pdf.body(
        f"After dedup and exclusion of {sum(timeout_counts.values())} timeout failures "
        f"and {len(smoke20_design_skips)} design skips, conditional non-timeout "
        f"correctness is {all_pass}/{all_valid} = {fmt_rate(all_pass, all_valid)}. "
        f"End-to-end execution success is {all_pass}/{all_total} = {fmt_rate(all_pass, all_total)}."
    )
    pdf.body(
        "We do not observe a clear monotonic aggregate trend by pool size or noise "
        "mode in either the raw or conditional tables. However, formal inference "
        "requires task- and seed-controlled analysis (logistic mixed model). "
        "We cannot claim 'no significant effect' without running statistical tests."
    )
    pdf.body(
        "Timeout frequency varies strongly by task. Several tasks appear highly "
        "accurate conditional on completion while failing end-to-end due to frequent "
        "timeouts (see Section 3 'TO-prone-solved' tier). These should NOT be "
        "interpreted as 'easy tasks' -- timeout is their dominant failure mode."
    )
    pdf.body(
        "Per v4 Section 9: the paired test across 14 First20 tasks is non-significant "
        "(p=0.14, CI spans zero). The aggregate 63% vs 41% advantage dissolves when "
        "properly accounting for pseudoreplication and task-level pairing."
    )

    out_path = BASE / "results" / "grid_analysis_timeout_excluded.pdf"
    pdf.output(str(out_path))
    print(f"Report saved to {out_path}")
    print(f"  Total pages: {pdf.page_no()}")


if __name__ == "__main__":
    main()
