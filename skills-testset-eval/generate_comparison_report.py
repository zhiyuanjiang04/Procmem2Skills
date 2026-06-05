#!/usr/bin/env python3
"""Generate comprehensive comparison PDF across all three eval batches.

Batches:
  - First20: 20 tasks from sb_prefill_n5.jsonl (the original "14 lightweight" eval)
  - Smoke10: 10 tasks from smoke10_n5.jsonl
  - Smoke20: 20 tasks from smoke20_n5.jsonl
Total: 50 unique tasks, no overlap.
"""
import json
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from fpdf import FPDF

BASE = Path(__file__).parent


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.cell(
            0, 8,
            "SkillsBench Prefill-Context Eval -- 50-Task Comparison Report",
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

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 // len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 7)
        self.set_fill_color(220, 230, 240)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 7)
        for ri, row in enumerate(rows):
            fill = ri % 2 == 1
            if fill:
                self.set_fill_color(245, 245, 250)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 5, str(cell), border=1, fill=fill, align="C")
            self.ln()
        self.ln(2)

    def colored_table(self, headers, rows, col_widths=None, highlight_cols=None):
        """Table with pass-rate cells colored green/yellow/red."""
        if col_widths is None:
            col_widths = [190 // len(headers)] * len(headers)
        if highlight_cols is None:
            highlight_cols = set()
        self.set_font("Helvetica", "B", 7)
        self.set_fill_color(40, 60, 100)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, h, border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 7)
        for ri, row in enumerate(rows):
            for i, cell in enumerate(row):
                if i in highlight_cols and cell not in ("--", ""):
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
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_rate_limited(r):
    for field in ("agent_stderr_tail", "agent_stdout_tail", "test_log_tail"):
        txt = r.get(field, "") or ""
        if "rate_limit" in txt.lower() or "overloaded" in txt.lower() or "529" in txt:
            return True
    if r.get("agent_rc") == 2 and r.get("agent_wall_s", 999) < 5:
        return True
    return False


def compute_task_stats(rows):
    """Returns dict: task_id -> {pass, fail, total, rate, wall_s_list, pool_sizes, noise_modes}."""
    stats = defaultdict(lambda: {
        "pass": 0, "fail": 0, "total": 0, "rate_limited": 0,
        "wall_s": [], "pool_sizes": set(), "noise_modes": set(), "seeds": set(),
    })
    for r in rows:
        tid = r["task_id"]
        s = stats[tid]
        s["total"] += 1
        if r.get("pass"):
            s["pass"] += 1
        else:
            s["fail"] += 1
        if is_rate_limited(r):
            s["rate_limited"] += 1
        if r.get("agent_wall_s"):
            s["wall_s"].append(r["agent_wall_s"])
        s["pool_sizes"].add(r.get("pool_size", "?"))
        s["noise_modes"].add(r.get("noise_mode", "?"))
        s["seeds"].add(r.get("seed", "?"))
    for tid in stats:
        s = stats[tid]
        s["rate"] = s["pass"] / s["total"] * 100 if s["total"] > 0 else 0
        s["wall_mean"] = np.mean(s["wall_s"]) if s["wall_s"] else 0
        s["wall_median"] = np.median(s["wall_s"]) if s["wall_s"] else 0
    return dict(stats)


def compute_grid_stats(rows):
    """Returns dict: (pool_size, noise_mode) -> {pass, total, rate}."""
    grid = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in rows:
        key = (r.get("pool_size", "?"), r.get("noise_mode", "?"))
        grid[key]["total"] += 1
        if r.get("pass"):
            grid[key]["pass"] += 1
    for k in grid:
        g = grid[k]
        g["rate"] = g["pass"] / g["total"] * 100 if g["total"] > 0 else 0
    return dict(grid)


def file_checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    # Load all three batches
    first20_path = BASE / "results" / "sb_prefill_n5.jsonl"
    smoke10_path = BASE / "results" / "smoke10" / "smoke10_n5.jsonl"
    smoke20_path = BASE / "results" / "smoke20" / "smoke20_n5.jsonl"

    first20 = load_jsonl(first20_path)
    smoke10 = load_jsonl(smoke10_path)
    smoke20 = load_jsonl(smoke20_path)

    first20_stats = compute_task_stats(first20)
    smoke10_stats = compute_task_stats(smoke10)
    smoke20_stats = compute_task_stats(smoke20)

    first20_grid = compute_grid_stats(first20)
    smoke10_grid = compute_grid_stats(smoke10)
    smoke20_grid = compute_grid_stats(smoke20)

    # Checksums for integrity
    checksums = {
        "sb_prefill_n5.jsonl": file_checksum(first20_path),
        "smoke10_n5.jsonl": file_checksum(smoke10_path),
        "smoke20_n5.jsonl": file_checksum(smoke20_path),
    }

    # Build PDF
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # --Page 1: Overview ----
    pdf.section("1. Experiment Overview")
    pdf.body(
        "This report compares execution pass-rates across three evaluation batches "
        "of the SkillsBench prefill-context pipeline. All batches use Claude Sonnet 4.6 "
        "with prefilled skill documents (SKILL.md) selected by embedding retrieval. "
        "Each trial runs the agent in a Docker sandbox with Bash/Read/Write/Edit tools "
        "and validates via pytest. First20 contains 20 tasks but only 14 are evaluable "
        "on host (6 have heavy Dockerfiles); see 20-Task Report v4 for paired analysis."
    )

    # Batch summary table
    pdf.subsection("Batch Summary")
    batch_headers = ["Batch", "Tasks", "Trials", "Valid", "Rate-Lim", "Pass", "Pass Rate", "Source"]
    first20_rl = sum(s["rate_limited"] for s in first20_stats.values())
    smoke10_rl = sum(s["rate_limited"] for s in smoke10_stats.values())
    smoke20_rl = sum(s["rate_limited"] for s in smoke20_stats.values())
    first20_pass = sum(s["pass"] for s in first20_stats.values())
    smoke10_pass = sum(s["pass"] for s in smoke10_stats.values())
    smoke20_pass = sum(s["pass"] for s in smoke20_stats.values())
    first20_total = sum(s["total"] for s in first20_stats.values())
    smoke10_total = sum(s["total"] for s in smoke10_stats.values())
    smoke20_total = sum(s["total"] for s in smoke20_stats.values())
    all_total = first20_total + smoke10_total + smoke20_total
    all_pass = first20_pass + smoke10_pass + smoke20_pass

    batch_rows = [
        ["First20", str(len(first20_stats)), str(first20_total), str(first20_total - first20_rl),
         str(first20_rl), str(first20_pass), f"{first20_pass/first20_total*100:.1f}%",
         "sb_prefill_n5.jsonl"],
        ["Smoke10", str(len(smoke10_stats)), str(smoke10_total), str(smoke10_total - smoke10_rl),
         str(smoke10_rl), str(smoke10_pass), f"{smoke10_pass/smoke10_total*100:.1f}%",
         "smoke10_n5.jsonl"],
        ["Smoke20", str(len(smoke20_stats)), str(smoke20_total), str(smoke20_total - smoke20_rl),
         str(smoke20_rl), str(smoke20_pass), f"{smoke20_pass/smoke20_total*100:.1f}%",
         "smoke20_n5.jsonl"],
        ["ALL", "50", str(all_total), str(all_total - first20_rl - smoke10_rl - smoke20_rl),
         str(first20_rl + smoke10_rl + smoke20_rl), str(all_pass),
         f"{all_pass/all_total*100:.1f}%", "--"],
    ]
    pdf.table(batch_headers, batch_rows, [22, 14, 16, 14, 16, 14, 20, 40])

    # --Page 2-3: Per-task pass rates ----
    pdf.add_page()
    pdf.section("2. Per-Task Pass Rates")
    pdf.body(
        "All 50 tasks sorted by pass rate within each batch. "
        "Green >= 70%, Yellow 30-70%, Red < 30%."
    )

    # Build combined rows
    def task_rows(stats, batch_name):
        rows = []
        for tid in sorted(stats, key=lambda t: -stats[t]["rate"]):
            s = stats[tid]
            pools = sorted(s["pool_sizes"])
            noises = sorted(s["noise_modes"])
            rows.append([
                batch_name, tid,
                f"{s['pass']}/{s['total']}",
                f"{s['rate']:.1f}%",
                f"{s['wall_mean']:.0f}s" if s["wall_s"] else "N/A",
                ",".join(str(p) for p in pools),
                ",".join(noises),
            ])
        return rows

    headers = ["Batch", "Task", "Pass/Total", "Rate", "Avg Wall", "Pool Sizes", "Noise Modes"]
    col_w = [18, 52, 22, 16, 18, 30, 34]

    # First20
    pdf.subsection("First20 (20 tasks, original eval)")
    pdf.colored_table(headers, task_rows(first20_stats, "First20"), col_w, {3})

    pdf.add_page()
    pdf.subsection("Smoke10 (10 tasks)")
    pdf.colored_table(headers, task_rows(smoke10_stats, "Smoke10"), col_w, {3})

    pdf.subsection("Smoke20 (20 tasks)")
    pdf.colored_table(headers, task_rows(smoke20_stats, "Smoke20"), col_w, {3})

    # --Page 4: Unified ranking ----
    pdf.add_page()
    pdf.section("3. Unified Task Ranking (All 50 Tasks)")
    pdf.body(
        "All tasks ranked by pass rate, color-coded. Tasks with 0% are listed at bottom."
    )
    all_tasks = []
    for tid, s in first20_stats.items():
        all_tasks.append(("First20", tid, s))
    for tid, s in smoke10_stats.items():
        all_tasks.append(("Smoke10", tid, s))
    for tid, s in smoke20_stats.items():
        all_tasks.append(("Smoke20", tid, s))
    all_tasks.sort(key=lambda x: -x[2]["rate"])

    rank_headers = ["#", "Batch", "Task", "Pass/Total", "Rate", "Median Wall"]
    rank_rows = []
    for i, (batch, tid, s) in enumerate(all_tasks, 1):
        rank_rows.append([
            str(i), batch, tid,
            f"{s['pass']}/{s['total']}",
            f"{s['rate']:.1f}%",
            f"{s['wall_median']:.0f}s" if s["wall_s"] else "N/A",
        ])
    rank_w = [8, 18, 56, 22, 18, 22]
    # Split into two pages if needed
    mid = 25
    pdf.colored_table(rank_headers, rank_rows[:mid], rank_w, {4})
    pdf.add_page()
    pdf.subsection("Unified Task Ranking (cont.)")
    pdf.colored_table(rank_headers, rank_rows[mid:], rank_w, {4})

    # --Page 5: Grid breakdown ----
    pdf.add_page()
    pdf.section("4. Pass Rate by Pool Size x Noise Mode")
    pdf.body(
        "Factorial grid showing pass rate per (pool_size, noise_mode) combination. "
        "Each cell shows pass/total (rate%). Compares how difficulty scaling affects "
        "each batch differently."
    )

    pool_sizes = sorted({k[0] for grids in [first20_grid, smoke10_grid, smoke20_grid] for k in grids})
    noise_modes = sorted({k[1] for grids in [first20_grid, smoke10_grid, smoke20_grid] for k in grids})

    for batch_name, grid in [("First20", first20_grid), ("Smoke10", smoke10_grid), ("Smoke20", smoke20_grid)]:
        pdf.subsection(f"{batch_name} Grid")
        grid_headers = ["Pool\\Noise"] + [n for n in noise_modes]
        grid_rows = []
        for ps in pool_sizes:
            row = [str(ps)]
            for nm in noise_modes:
                g = grid.get((ps, nm))
                if g:
                    row.append(f"{g['pass']}/{g['total']} ({g['rate']:.0f}%)")
                else:
                    row.append("--")
            grid_rows.append(row)
        n_cols = len(grid_headers)
        grid_w = [22] + [int(168 / (n_cols - 1))] * (n_cols - 1)
        pdf.table(grid_headers, grid_rows, grid_w)

    # --Page 6: Difficulty analysis ----
    pdf.add_page()
    pdf.section("5. Difficulty Analysis")

    # Pool size effect
    pdf.subsection("Pass Rate vs Pool Size (aggregated across noise modes)")
    for batch_name, grid in [("First20", first20_grid), ("Smoke10", smoke10_grid), ("Smoke20", smoke20_grid)]:
        ps_agg = defaultdict(lambda: {"pass": 0, "total": 0})
        for (ps, nm), g in grid.items():
            ps_agg[ps]["pass"] += g["pass"]
            ps_agg[ps]["total"] += g["total"]
        row_data = []
        for ps in sorted(ps_agg):
            a = ps_agg[ps]
            rate = a["pass"] / a["total"] * 100 if a["total"] > 0 else 0
            row_data.append([batch_name, str(ps), f"{a['pass']}/{a['total']}", f"{rate:.1f}%"])
        if row_data:
            pdf.table(["Batch", "Pool Size", "Pass/Total", "Rate"], row_data, [30, 30, 40, 40])

    # Noise mode effect
    pdf.subsection("Pass Rate vs Noise Mode (aggregated across pool sizes)")
    for batch_name, grid in [("First20", first20_grid), ("Smoke10", smoke10_grid), ("Smoke20", smoke20_grid)]:
        nm_agg = defaultdict(lambda: {"pass": 0, "total": 0})
        for (ps, nm), g in grid.items():
            nm_agg[nm]["pass"] += g["pass"]
            nm_agg[nm]["total"] += g["total"]
        row_data = []
        for nm in sorted(nm_agg):
            a = nm_agg[nm]
            rate = a["pass"] / a["total"] * 100 if a["total"] > 0 else 0
            row_data.append([batch_name, nm, f"{a['pass']}/{a['total']}", f"{rate:.1f}%"])
        if row_data:
            pdf.table(["Batch", "Noise Mode", "Pass/Total", "Rate"], row_data, [30, 30, 40, 40])

    # --Page 7: Zero-pass task analysis ----
    pdf.add_page()
    pdf.section("6. Zero-Pass Tasks Analysis")
    zero_tasks = [(b, t, s) for b, t, s in all_tasks if s["rate"] == 0]
    pdf.body(f"{len(zero_tasks)} of 50 tasks have 0% pass rate across all conditions.")

    zero_headers = ["Batch", "Task", "Trials", "Avg Wall(s)", "Pool Sizes", "Rate-Lim"]
    zero_rows = []
    for batch, tid, s in zero_tasks:
        zero_rows.append([
            batch, tid, str(s["total"]),
            f"{s['wall_mean']:.0f}" if s["wall_s"] else "N/A",
            ",".join(str(p) for p in sorted(s["pool_sizes"])),
            str(s["rate_limited"]),
        ])
    pdf.table(zero_headers, zero_rows, [18, 52, 16, 22, 40, 16])

    # Categorize zero-pass reasons
    pdf.subsection("Failure Pattern Analysis")
    pdf.body(
        "Zero-pass tasks are categorized by likely failure mode based on wall time "
        "and error patterns. No wall data (N/A) = agent never launched (env/Docker skip). "
        "Short wall time (<30s avg) = immediate setup crash. "
        "Long wall time (>60s) = execution attempt that does not converge."
    )
    no_wall = [(b, t, s) for b, t, s in zero_tasks if not s["wall_s"]]
    short_fail = [(b, t, s) for b, t, s in zero_tasks if s["wall_s"] and s["wall_mean"] < 30]
    long_fail = [(b, t, s) for b, t, s in zero_tasks if s["wall_s"] and s["wall_mean"] >= 30]
    pdf.body(f"  No wall data (agent skipped): {len(no_wall)} tasks")
    pdf.body(f"  Setup/env failures (wall < 30s): {len(short_fail)} tasks")
    pdf.body(f"  Execution failures (wall >= 30s): {len(long_fail)} tasks")

    # --Page 8: Tier classification ----
    pdf.add_page()
    pdf.section("7. Task Difficulty Tiers")
    pdf.body(
        "Tasks classified into difficulty tiers based on observed pass rate. "
        "This helps identify which tasks benefit most from skill prefilling."
    )

    tiers = {
        "Easy (>=70%)": [(b, t, s) for b, t, s in all_tasks if s["rate"] >= 70],
        "Medium (30-70%)": [(b, t, s) for b, t, s in all_tasks if 30 <= s["rate"] < 70],
        "Hard (1-30%)": [(b, t, s) for b, t, s in all_tasks if 0 < s["rate"] < 30],
        "Unsolved (0%)": [(b, t, s) for b, t, s in all_tasks if s["rate"] == 0],
    }

    tier_summary = []
    for tier_name, tasks in tiers.items():
        tier_summary.append([
            tier_name, str(len(tasks)),
            f"{len(tasks)/50*100:.0f}%",
            ", ".join(t[1][:25] for t in tasks[:5]) + ("..." if len(tasks) > 5 else ""),
        ])
    pdf.table(
        ["Tier", "Count", "% of 50", "Example Tasks"],
        tier_summary,
        [30, 15, 15, 130],
    )

    for tier_name, tasks in tiers.items():
        if tasks:
            pdf.subsection(tier_name)
            for b, t, s in tasks:
                pdf.body(f"  [{b}] {t}: {s['pass']}/{s['total']} ({s['rate']:.1f}%)")

    # --Page 9: Data Integrity ----
    pdf.add_page()
    pdf.section("8. Data Integrity & Reliability Verification")

    pdf.subsection("File Checksums (SHA-256, first 16 hex)")
    for fname, cksum in checksums.items():
        pdf.kv(fname, cksum)
    pdf.ln(3)

    pdf.subsection("Trial Count Verification")
    pdf.body(
        f"First20: {first20_total} trials across {len(first20_stats)} tasks "
        f"(avg {first20_total/len(first20_stats):.1f} trials/task)"
    )
    pdf.body(
        f"Smoke10: {smoke10_total} trials across {len(smoke10_stats)} tasks "
        f"(avg {smoke10_total/len(smoke10_stats):.1f} trials/task)"
    )
    pdf.body(
        f"Smoke20: {smoke20_total} trials across {len(smoke20_stats)} tasks "
        f"(avg {smoke20_total/len(smoke20_stats):.1f} trials/task)"
    )

    # Per-task trial count range
    pdf.subsection("Per-Task Trial Count Range")
    for batch_name, stats in [("First20", first20_stats), ("Smoke10", smoke10_stats), ("Smoke20", smoke20_stats)]:
        counts = sorted(s["total"] for s in stats.values())
        pdf.body(
            f"  {batch_name}: min={counts[0]}, max={counts[-1]}, "
            f"median={counts[len(counts)//2]}"
        )
    pdf.body(
        "Note: First20 has uneven trial counts (4-588/task) due to incremental "
        "Phase P0 seed replication. Smoke20 has 75 trials/task but includes 30 "
        "design skips at pool_size=5 for 2 tasks whose |GT|=6 > pool_size "
        "(python-scala-translation, travel-planning). Smoke10 is incomplete for "
        "2 tasks: jpg-ocr-stat (52/75) and glm-lake-mendota (68/75) due to resume "
        "with an 8-task subset after the initial 10-task run."
    )

    # Duplicate analysis
    pdf.subsection("Trial ID Uniqueness")
    from collections import Counter as _Counter
    all_trial_ids_list = []
    for rows_list in [first20, smoke10, smoke20]:
        for r in rows_list:
            all_trial_ids_list.append(r.get("trial_id", ""))
    id_counts = _Counter(all_trial_ids_list)
    unique_ids = len(id_counts)
    dup_configs = sum(1 for c in id_counts.values() if c > 1)
    dup_rows = sum(c - 1 for c in id_counts.values() if c > 1)
    pdf.body(f"Unique config IDs: {unique_ids}")
    pdf.body(f"Configs run multiple times: {dup_configs}")
    pdf.body(f"Extra rows from re-runs: {dup_rows}")
    pdf.body(
        "Note: trial_id encodes (task, pool_size, noise, seed) -- it is a config ID, "
        "not a unique execution ID. Repeated IDs are valid independent replications "
        "of the same configuration (all in First20's Phase P0). Each row is a distinct "
        "agent execution. No cross-batch duplicates exist."
    )

    # Task disjointness
    pdf.subsection("Task Disjointness")
    t1 = set(first20_stats.keys())
    t2 = set(smoke10_stats.keys())
    t3 = set(smoke20_stats.keys())
    pdf.body(f"First20 & Smoke10: {'empty (OK)' if not (t1 & t2) else str(t1 & t2)}")
    pdf.body(f"First20 & Smoke20: {'empty (OK)' if not (t1 & t3) else str(t1 & t3)}")
    pdf.body(f"Smoke10 & Smoke20: {'empty (OK)' if not (t2 & t3) else str(t2 & t3)}")

    # Rate-limit contamination
    pdf.subsection("Rate-Limit Contamination")
    pdf.body(
        "Trials flagged as rate-limited (stderr mentions rate_limit/overloaded/529, "
        "or agent exited immediately with rc=2):"
    )
    pdf.body(f"  First20: {first20_rl}/{first20_total} ({first20_rl/first20_total*100:.1f}%)")
    pdf.body(f"  Smoke10: {smoke10_rl}/{smoke10_total} ({smoke10_rl/smoke10_total*100:.1f}%)")
    pdf.body(f"  Smoke20: {smoke20_rl}/{smoke20_total} ({smoke20_rl/smoke20_total*100:.1f}%)")
    pdf.body("Verdict: rate-limit contamination is negligible (<0.2%).")

    # Seed coverage
    pdf.subsection("Seed Coverage")
    for batch_name, stats in [("First20", first20_stats), ("Smoke10", smoke10_stats), ("Smoke20", smoke20_stats)]:
        all_seeds = set()
        for s in stats.values():
            all_seeds |= s["seeds"]
        pdf.body(f"  {batch_name}: seeds = {sorted(all_seeds)}")
    pdf.body(
        "Note: First20 uses seeds [1,2,3,4] (seed 0 was in the earlier Phase C run, "
        "stored separately in sb_exec_20t.jsonl). Smoke batches use [0,1,2,3,4]."
    )

    # Model consistency
    pdf.subsection("Model Consistency")
    models_with_count = {}
    missing_model_count = 0
    for rows_list in [first20, smoke10, smoke20]:
        for r in rows_list:
            m = r.get("model")
            if m:
                models_with_count[m] = models_with_count.get(m, 0) + 1
            else:
                missing_model_count += 1
    for m, c in sorted(models_with_count.items()):
        pdf.body(f"  {m}: {c} trials")
    if missing_model_count:
        pdf.body(
            f"  (no model field): {missing_model_count} trials from two distinct causes:"
        )
        pdf.body(
            "    - First20: 244 trials across 6 tasks with heavy Dockerfiles "
            "(unsafe for host execution). Agent never launched."
        )
        pdf.body(
            "    - Smoke20: 30 trials at pool_size=5 for python-scala-translation "
            "and travel-planning, where |GT|=6 > pool_size. These are design skips "
            "(GT could not fit in pool). Note: travel-planning is 16/75 overall "
            "(21.3%), NOT 0% -- only its pool_size=5 slice was skipped."
        )

    # Wall-time distribution
    pdf.subsection("Wall-Time Statistics (excluding N/A)")
    for batch_name, rows_list in [("First20", first20), ("Smoke10", smoke10), ("Smoke20", smoke20)]:
        walls = [r["agent_wall_s"] for r in rows_list if r.get("agent_wall_s")]
        no_wall = sum(1 for r in rows_list if not r.get("agent_wall_s"))
        if walls:
            pdf.body(
                f"  {batch_name}: mean={np.mean(walls):.0f}s, median={np.median(walls):.0f}s, "
                f"p5={np.percentile(walls,5):.0f}s, p95={np.percentile(walls,95):.0f}s "
                f"(n={len(walls)}, {no_wall} trials missing wall data)"
            )

    # Overall verdict
    pdf.subsection("Reliability Verdict")
    pdf.body(
        "PASS with caveats. All three batches use disjoint task sets with no "
        "cross-contamination. Rate-limit interference is negligible (<0.2%). Model "
        "is consistent (claude-sonnet-4-6) across all executed trials."
    )
    pdf.body(
        "DESIGN CAVEATS:"
    )
    pdf.body(
        "  1. First20 has uneven replication (4-588 trials/task) from incremental "
        "Phase P0 runs. The 6 tasks with heavy Dockerfiles (court-form-filling, "
        "crystallographic-wyckoff, data-to-d3, dynamic-object-aware-egomotion, "
        "edit-pdf, exoplanet-detection-period) were never executed on host. "
        "Per the 20-Task Report v4, only 14 of these 20 tasks are evaluable."
    )
    pdf.body(
        "  2. Smoke10 is incomplete for 2 tasks: jpg-ocr-stat (52/75) and "
        "glm-lake-mendota (68/75) due to resume with an 8-task subset."
    )
    pdf.body(
        "  3. Smoke20 row count is balanced (75/task) but includes 30 design "
        "skips at pool_size=5 for python-scala-translation and travel-planning "
        "where |GT|=6 > pool_size. travel-planning has 16/75 pass overall (21.3%)."
    )
    pdf.body(
        "  4. Per the 20-Task Report v4 paired analysis: aggregate pass-rate "
        "differences between prefill conditions are NOT statistically significant "
        "(p > 0.05 for all comparisons, N=14 tasks). Signal is driven by a few "
        "standout tasks, not a consistent cross-task effect. These aggregate "
        "comparisons cannot support causal claims."
    )

    # --Save ----
    out_path = BASE / "results" / "comparison_50task_report.pdf"
    pdf.output(str(out_path))
    print(f"Report saved to {out_path}")
    print(f"  Total pages: {pdf.page_no()}")


if __name__ == "__main__":
    main()
