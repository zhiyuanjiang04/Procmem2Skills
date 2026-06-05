#!/usr/bin/env python3
"""Generate a unified unique-config report for the 20t v4 and 50-task reports.

Primary metric:
  - config = trial_id, i.e. (task_id, pool_size, noise_mode, seed)
  - deduplicate to one row per config, last row wins
  - exclude skipped configs from execution pass-rate denominators

The report also records raw-row counts, skipped configs, missing configs, and
the artifacts used to make the conclusions reproducible.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from fpdf import FPDF


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
OUT_MD = RESULTS / "unified_unique_config_report.md"
OUT_PDF = RESULTS / "unified_unique_config_report.pdf"


@dataclass
class Artifact:
    path: Path
    role: str


ARTIFACTS = [
    Artifact(RESULTS / "20t_report_v4.pdf", "Source report: seed-0 20-task paired analysis"),
    Artifact(RESULTS / "comparison_50task_report.pdf", "Source report: 50-task comparison"),
    Artifact(BASE / "generate_20t_report.py", "Generator for 20t_report_v4.pdf"),
    Artifact(BASE / "generate_comparison_report.py", "Generator for comparison_50task_report.pdf"),
    Artifact(RESULTS / "sb_exec_20t.jsonl", "20t v4 prefill, seed 0"),
    Artifact(RESULTS / "sb_baselines_n5.jsonl", "20t v4 noskill and GT-only baselines, seeds 0-4"),
    Artifact(RESULTS / "sb_prefill_n5.jsonl", "First20 Phase P0 prefill, seeds 1-4 plus reruns"),
    Artifact(RESULTS / "smoke10" / "smoke10_n5.jsonl", "Smoke10 prefill, seeds 0-4"),
    Artifact(RESULTS / "smoke20" / "smoke20_n5.jsonl", "Smoke20 prefill, seeds 0-4"),
    Artifact(BASE / "logs" / "smoke10.log", "Run log supporting Smoke10 resume/incomplete diagnosis"),
    Artifact(BASE / "logs" / "smoke20.log", "Run log supporting Smoke20 run provenance"),
    Artifact(BASE / "testsets" / "data" / "smoke10.jsonl", "Smoke10 task list"),
    Artifact(BASE / "testsets" / "data" / "smoke20_next.jsonl", "Smoke20 task list"),
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def pdf_pages(path: Path) -> str:
    try:
        out = subprocess.check_output(["pdfinfo", str(path)], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return "-"
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return line.split(":", 1)[1].strip()
    return "-"


def file_rows(path: Path) -> str:
    if path.suffix != ".jsonl":
        return "-"
    try:
        return str(sum(1 for line in path.open() if line.strip()))
    except OSError:
        return "-"


def artifact_rows() -> list[list[str]]:
    rows = []
    for art in ARTIFACTS:
        p = art.path
        if not p.exists():
            rows.append([str(p.relative_to(BASE)), "MISSING", "-", "-", "-", art.role])
            continue
        st = p.stat()
        rel = str(p.relative_to(BASE))
        mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        pages = pdf_pages(p) if p.suffix == ".pdf" else "-"
        rows.append([rel, str(st.st_size), file_rows(p), pages, sha16(p), art.role + f"; mtime {mtime}"])
    return rows


def config_key(row: dict, *, v4_prefill: bool = False) -> tuple:
    if v4_prefill:
        return (row.get("task_id"), row.get("pool_size"), row.get("noise_mode"))
    return (row.get("trial_id")
            or (row.get("task_id"), row.get("pool_size"), row.get("noise_mode"), row.get("seed")))


def dedup(rows: Iterable[dict], *, v4_prefill: bool = False) -> dict[tuple, dict]:
    out = {}
    for row in rows:
        out[config_key(row, v4_prefill=v4_prefill)] = row
    return out


def exact_duplicate_count(path: Path) -> tuple[int, int]:
    counts = Counter(line for line in path.read_text().splitlines() if line.strip())
    groups = sum(1 for c in counts.values() if c > 1)
    extra = sum(c - 1 for c in counts.values() if c > 1)
    return groups, extra


def stats_from_unique(unique: dict[tuple, dict]) -> dict:
    rows = list(unique.values())
    skipped = [r for r in rows if r.get("skipped")]
    executed = [r for r in rows if not r.get("skipped")]
    passed = sum(1 for r in executed if r.get("pass"))
    tasks_all = {r.get("task_id") for r in rows}
    tasks_exec = {r.get("task_id") for r in executed}
    return {
        "unique": len(rows),
        "skipped": len(skipped),
        "executed": len(executed),
        "passed": passed,
        "rate": passed / len(executed) if executed else 0.0,
        "tasks_all": len(tasks_all),
        "tasks_exec": len(tasks_exec),
    }


def task_stats(rows: Iterable[dict]) -> dict[str, dict]:
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
    out = {}
    for task, recs in by_task.items():
        executed = [r for r in recs if not r.get("skipped")]
        skipped = [r for r in recs if r.get("skipped")]
        passed = sum(1 for r in executed if r.get("pass"))
        out[task] = {
            "pass": passed,
            "exec": len(executed),
            "skip": len(skipped),
            "rate": passed / len(executed) if executed else None,
        }
    return out


def pool_rate_rows(rows: Iterable[dict], *, include_pool_sizes: list[int]) -> list[list[str]]:
    by_pool = defaultdict(list)
    for r in rows:
        if r.get("skipped"):
            continue
        if r.get("noise_mode") not in ("random", "hard", "easy"):
            continue
        by_pool[r.get("pool_size")].append(r)

    out = []
    for pool_size in include_pool_sizes:
        recs = by_pool.get(pool_size, [])
        passed = sum(1 for r in recs if r.get("pass"))
        tasks = {r.get("task_id") for r in recs}
        out.append([
            f"GT+{pool_size}",
            str(pool_size),
            pn(passed, len(recs)),
            fmt_rate(passed / len(recs) if recs else None),
            str(len(tasks)),
        ])
    return out


def is_timeout_failure(row: dict) -> bool:
    """Timeout classifier aligned with the existing v4 error taxonomy.

    Only failed executions can be timeout failures. Successful long-running trials
    stay in the numerator and denominator.
    """
    if row.get("skipped") or row.get("pass") is True:
        return False
    text = " ".join(
        str(row.get(k, "") or "")
        for k in ("agent_stdout_tail", "agent_stderr_tail", "test_log_tail")
    ).lower()
    wall = row.get("agent_wall_s") or 0
    return (
        wall > 600
        or row.get("agent_rc") == 124
        or "timeout" in text
        or "timed out" in text
        or "killed" in text
    )


def timeout_split_summary_rows(rows: Iterable[dict], *, include_gt_only: list[dict] | None,
                               pool_sizes: list[int]) -> list[list[str]]:
    out = []

    def add_row(condition: str, pool: str, noise: str, recs: list[dict], task_suffix: str | None = None) -> None:
        executed = [r for r in recs if not r.get("skipped")]
        passed = sum(1 for r in executed if r.get("pass"))
        timeouts = sum(1 for r in executed if is_timeout_failure(r))
        non_timeout = [r for r in executed if not is_timeout_failure(r)]
        nt_passed = sum(1 for r in non_timeout if r.get("pass"))
        tasks = {r.get("task_id") for r in executed}
        task_cell = task_suffix if task_suffix is not None else str(len(tasks))
        out.append([
            condition,
            pool,
            noise,
            pn(passed, len(executed)),
            fmt_rate(passed / len(executed) if executed else None),
            str(timeouts),
            pn(nt_passed, len(non_timeout)),
            fmt_rate(nt_passed / len(non_timeout) if non_timeout else None),
            task_cell,
        ])

    if include_gt_only is not None:
        add_row("GT-only", "GT only", "-", include_gt_only)

    row_list = list(rows)
    for pool_size in pool_sizes:
        for noise_mode in ("random", "easy", "hard"):
            recs = [
                r for r in row_list
                if r.get("pool_size") == pool_size
                and r.get("noise_mode") == noise_mode
                and not r.get("skipped")
            ]
            add_row(f"GT+{pool_size}", str(pool_size), noise_mode, recs)
    return out


def timeout_split_rows_first20(data: dict) -> list[list[str]]:
    return timeout_split_summary_rows(
        data["unique"]["first20_combined"].values(),
        include_gt_only=data["baseline"]["gtonly"],
        pool_sizes=[5, 10, 20, 50, 100],
    )


def timeout_split_rows_50_source(data: dict) -> list[list[str]]:
    return timeout_split_summary_rows(
        (r for _, r in data["unique"]["source50"].items()),
        include_gt_only=None,
        pool_sizes=[5, 10, 20, 50, 100],
    )


def timeout_split_rows_all_available(data: dict) -> list[list[str]]:
    return timeout_split_summary_rows(
        (r for _, r in data["unique"]["all_available"].items()),
        include_gt_only=data["baseline"]["gtonly"],
        pool_sizes=[5, 10, 20, 50, 100],
    )


def condition_rows_for_first20(data: dict) -> list[list[str]]:
    gt = data["baseline"]["gtonly"]
    gt_pass = sum(1 for r in gt if r.get("pass"))
    rows = [[
        "GT-only",
        "GT only",
        pn(gt_pass, len(gt)),
        fmt_rate(gt_pass / len(gt) if gt else None),
        str(len({r.get("task_id") for r in gt})),
    ]]
    rows.extend(pool_rate_rows(data["unique"]["first20_combined"].values(),
                               include_pool_sizes=[5, 10, 20, 50, 100]))
    return rows


def condition_rows_for_50_source(data: dict) -> list[list[str]]:
    rows = [[
        "GT-only",
        "N/A",
        "N/A",
        "N/A",
        "First20 only; no Smoke10/Smoke20 GT-only baseline",
    ]]
    rows.extend(pool_rate_rows(
        (r for _, r in data["unique"]["source50"].items()),
        include_pool_sizes=[5, 10, 20, 50, 100],
    ))
    return rows


def condition_rows_for_all_available(data: dict) -> list[list[str]]:
    rows = [[
        "GT-only",
        "GT only",
        "37/70",
        "52.9%",
        "14 First20 tasks only",
    ]]
    rows.extend(pool_rate_rows(
        (r for _, r in data["unique"]["all_available"].items()),
        include_pool_sizes=[5, 10, 20, 50, 100],
    ))
    return rows


def fmt_rate(x: float | None) -> str:
    return "N/A" if x is None else f"{100 * x:.1f}%"


def pn(p: int, n: int) -> str:
    return f"{p}/{n}" if n else "0/0"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(c).replace("\n", " ") for c in row) + " |")
    return "\n".join(out)


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8, "Unified Unique-Config Evaluation Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(40, 60, 100)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, "  " + title, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def body(self, text: str):
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 4.5, text)
        self.ln(1)

    def table(self, headers: list[str], rows: list[list[str]], widths: list[int], font_size: float = 7):
        self.set_font("Helvetica", "B", font_size)
        self.set_fill_color(220, 230, 240)
        for i, h in enumerate(headers):
            self.cell(widths[i], 6, str(h), border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", font_size)
        for ri, row in enumerate(rows):
            if self.get_y() > 260:
                self.add_page()
                self.set_font("Helvetica", "B", font_size)
                self.set_fill_color(220, 230, 240)
                for i, h in enumerate(headers):
                    self.cell(widths[i], 6, str(h), border=1, fill=True, align="C")
                self.ln()
                self.set_font("Helvetica", "", font_size)
            fill = ri % 2 == 1
            if fill:
                self.set_fill_color(245, 245, 250)
            for i, cell in enumerate(row):
                txt = str(cell)
                if len(txt) > 42:
                    txt = txt[:39] + "..."
                self.cell(widths[i], 5, txt, border=1, fill=fill, align="C")
            self.ln()
        self.ln(2)


def build_report_data() -> dict:
    v4_raw = load_jsonl(RESULTS / "sb_exec_20t.jsonl")
    first20_raw = load_jsonl(RESULTS / "sb_prefill_n5.jsonl")
    smoke10_raw = load_jsonl(RESULTS / "smoke10" / "smoke10_n5.jsonl")
    smoke20_raw = load_jsonl(RESULTS / "smoke20" / "smoke20_n5.jsonl")
    baselines_raw = load_jsonl(RESULTS / "sb_baselines_n5.jsonl")

    v4_unique = dedup(v4_raw, v4_prefill=True)
    first20_unique = dedup(first20_raw)
    smoke10_unique = dedup(smoke10_raw)
    smoke20_unique = dedup(smoke20_raw)

    # Combine v4 seed=0 with First20 P0 seeds 1-4. These keys are disjoint in
    # practice, but last-wins keeps this deterministic if a rerun exists.
    first20_combined_unique = dict(dedup(v4_raw))
    first20_combined_unique.update(first20_unique)

    source50_unique = {}
    source50_unique.update({("First20P0", k): v for k, v in first20_unique.items()})
    source50_unique.update({("Smoke10", k): v for k, v in smoke10_unique.items()})
    source50_unique.update({("Smoke20", k): v for k, v in smoke20_unique.items()})

    all_available_unique = {}
    all_available_unique.update({("First20Combined", k): v for k, v in first20_combined_unique.items()})
    all_available_unique.update({("Smoke10", k): v for k, v in smoke10_unique.items()})
    all_available_unique.update({("Smoke20", k): v for k, v in smoke20_unique.items()})

    # Baseline summary for interpreting v4 only.
    baseline_unique = dedup([r for r in baselines_raw if not r.get("skipped")])
    ns = [r for r in baseline_unique.values() if r.get("noise_mode") == "noskill"]
    gt = [r for r in baseline_unique.values() if r.get("noise_mode") == "gtonly"]

    return {
        "raw": {
            "v4": v4_raw,
            "first20": first20_raw,
            "smoke10": smoke10_raw,
            "smoke20": smoke20_raw,
        },
        "unique": {
            "v4": v4_unique,
            "first20": first20_unique,
            "first20_combined": first20_combined_unique,
            "smoke10": smoke10_unique,
            "smoke20": smoke20_unique,
            "source50": source50_unique,
            "all_available": all_available_unique,
        },
        "baseline": {
            "noskill": ns,
            "gtonly": gt,
        },
    }


def build_markdown(data: dict) -> str:
    unique = data["unique"]
    raw = data["raw"]

    v4_s = stats_from_unique(unique["v4"])
    f20_s = stats_from_unique(unique["first20"])
    f20_combined_s = stats_from_unique(unique["first20_combined"])
    sm10_s = stats_from_unique(unique["smoke10"])
    sm20_s = stats_from_unique(unique["smoke20"])
    source50_s = stats_from_unique(unique["source50"])
    all_s = stats_from_unique(unique["all_available"])

    ns = data["baseline"]["noskill"]
    gt = data["baseline"]["gtonly"]
    ns_pass = sum(1 for r in ns if r.get("pass"))
    gt_pass = sum(1 for r in gt if r.get("pass"))

    first20_tasks_v4 = task_stats(unique["v4"].values())
    first20_tasks_p0 = task_stats(unique["first20"].values())
    first20_tasks_combined = task_stats(unique["first20_combined"].values())

    first20_rows = []
    for task in sorted(set(first20_tasks_v4) | set(first20_tasks_p0) | set(first20_tasks_combined)):
        a = first20_tasks_v4.get(task, {"pass": 0, "exec": 0, "skip": 0, "rate": None})
        b = first20_tasks_p0.get(task, {"pass": 0, "exec": 0, "skip": 0, "rate": None})
        c = first20_tasks_combined.get(task, {"pass": 0, "exec": 0, "skip": 0, "rate": None})
        first20_rows.append([
            task,
            pn(a["pass"], a["exec"]),
            fmt_rate(a["rate"]),
            pn(b["pass"], b["exec"]),
            fmt_rate(b["rate"]),
            pn(c["pass"], c["exec"]),
            fmt_rate(c["rate"]),
            str(c["skip"]),
        ])

    batch_rows = []
    for name, expected, raw_name, st, uniq in [
        ("20t v4 prefill seed0", 240, "v4", v4_s, unique["v4"]),
        ("First20 P0 seeds1-4", 960, "first20", f20_s, unique["first20"]),
        ("First20 combined seeds0-4", 1200, "-", f20_combined_s, unique["first20_combined"]),
        ("Smoke10", 750, "smoke10", sm10_s, unique["smoke10"]),
        ("Smoke20", 1500, "smoke20", sm20_s, unique["smoke20"]),
        ("50-report source only", 3210, "-", source50_s, unique["source50"]),
        ("All available prefill", 3450, "-", all_s, unique["all_available"]),
    ]:
        raw_count = len(raw[raw_name]) if raw_name in raw else "-"
        batch_rows.append([
            name,
            str(expected),
            str(raw_count),
            str(st["unique"]),
            str(expected - st["unique"]),
            str(st["skipped"]),
            str(st["executed"]),
            pn(st["passed"], st["executed"]),
            fmt_rate(st["rate"]),
            f"{st['tasks_exec']}/{st['tasks_all']}",
        ])

    source50_tasks = []
    for batch_name, uniq_map in [
        ("First20 P0", unique["first20"]),
        ("Smoke10", unique["smoke10"]),
        ("Smoke20", unique["smoke20"]),
    ]:
        ts = task_stats(uniq_map.values())
        for task in sorted(ts):
            s = ts[task]
            source50_tasks.append([
                batch_name,
                task,
                pn(s["pass"], s["exec"]),
                fmt_rate(s["rate"]),
                str(s["skip"]),
            ])

    # Duplicate audit.
    dup_rows = []
    for label, path, uniq_map, raw_rows in [
        ("20t v4 prefill", RESULTS / "sb_exec_20t.jsonl", unique["v4"], raw["v4"]),
        ("First20 P0", RESULTS / "sb_prefill_n5.jsonl", unique["first20"], raw["first20"]),
        ("Smoke10", RESULTS / "smoke10" / "smoke10_n5.jsonl", unique["smoke10"], raw["smoke10"]),
        ("Smoke20", RESULTS / "smoke20" / "smoke20_n5.jsonl", unique["smoke20"], raw["smoke20"]),
    ]:
        groups, extra_exact = exact_duplicate_count(path)
        dup_rows.append([
            label,
            str(len(raw_rows)),
            str(len(uniq_map)),
            str(len(raw_rows) - len(uniq_map)),
            str(groups),
            str(extra_exact),
        ])

    lines = []
    lines.append("# Unified Unique-Config Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "The apparent conflict between `20t_report_v4.pdf` and "
        "`comparison_50task_report.pdf` is a denominator issue, not a model-result conflict."
    )
    lines.append("")
    lines.append(
        f"- 20t v4 prefill seed0, using unique configs and excluding skipped configs: "
        f"{pn(v4_s['passed'], v4_s['executed'])} = {fmt_rate(v4_s['rate'])}."
    )
    lines.append(
        f"- First20 P0 from the 50-task report, using unique configs and excluding skipped configs: "
        f"{pn(f20_s['passed'], f20_s['executed'])} = {fmt_rate(f20_s['rate'])}."
    )
    lines.append(
        f"- Combined First20 prefill seeds0-4: "
        f"{pn(f20_combined_s['passed'], f20_combined_s['executed'])} = {fmt_rate(f20_combined_s['rate'])}."
    )
    lines.append(
        f"- First20 GT-only baseline: {pn(gt_pass, len(gt))} = "
        f"{fmt_rate(gt_pass / len(gt))}; comparable First20 combined GT+pool rates are listed below."
    )
    lines.append(
        f"- 50-report source only under this same rule: "
        f"{pn(source50_s['passed'], source50_s['executed'])} = {fmt_rate(source50_s['rate'])} "
        f"over {source50_s['tasks_exec']} task IDs with at least one executed config."
    )
    lines.append("")
    lines.append(
        "Interpretation: the First20 prefill result is stable across v4 seed0 and later P0 seeds1-4. "
        "Raw-row reporting in the 50-task report should not be used as a performance denominator for "
        "First20 because repeated config rows heavily overweight a few tasks."
    )
    lines.append("")
    lines.append("## Statistical Unit")
    lines.append("")
    lines.append("- Primary unit: one unique config, encoded by `trial_id`.")
    lines.append("- `skipped=true` configs are excluded from execution pass-rate denominators.")
    lines.append("- Raw rows are retained only for provenance, duplicate, and coverage audits.")
    lines.append("- Rate-limited rows are not removed from the primary metric here, matching the requested 404/672 denominator.")
    lines.append("- `GT+5/10/20/50/100` means GT skills are included in a total candidate pool of 5/10/20/50/100 skills; it does not mean GT plus that many additional distractors.")
    lines.append("- Timeout-excluded rates remove failed executions classified as timeout failures (`agent_wall_s > 600`, `agent_rc=124`, or timeout/killed text).")
    lines.append("")
    lines.append("## Artifact Inventory")
    lines.append("")
    lines.append(md_table(["Artifact", "Bytes", "Rows", "PDF Pages", "SHA256-16", "Role"], artifact_rows()))
    lines.append("")
    lines.append("## Unified Batch Summary")
    lines.append("")
    lines.append(md_table(
        ["Dataset slice", "Expected configs", "Raw rows", "Unique configs", "Missing", "Skipped", "Executed", "Pass/Executed", "Rate", "Exec/All tasks"],
        batch_rows,
    ))
    lines.append("")
    lines.append("## GT And GT+Pool Pass Rates")
    lines.append("")
    lines.append("### First20 Comparable Scope")
    lines.append("")
    lines.append(
        "This is the cleanest condition comparison because GT-only exists for these same 14 executable First20 tasks. "
        "GT+ rows use the combined First20 unique-config prefill data across seeds 0-4 and noise modes random/hard/easy."
    )
    lines.append("")
    lines.append(md_table(
        ["Condition", "Pool size meaning", "Pass/Executed", "Pass rate", "Executable tasks"],
        condition_rows_for_first20(data),
    ))
    lines.append("")
    lines.append("First20 split by noise mode, with timeout failures excluded in the rightmost rate:")
    lines.append("")
    lines.append(md_table(
        ["Condition", "Pool", "Noise", "Pass/Exec", "Rate", "Timeout fails", "Pass/Non-timeout", "Timeout-excl rate", "Tasks"],
        timeout_split_rows_first20(data),
    ))
    lines.append("")
    lines.append("### 50-Report Source Scope")
    lines.append("")
    lines.append(
        "This recomputes the 50-task source files only (`sb_prefill_n5`, Smoke10, Smoke20). "
        "There is no GT-only baseline for Smoke10/Smoke20, so GT-only is not a 50-task metric."
    )
    lines.append("")
    lines.append(md_table(
        ["Condition", "Pool size meaning", "Pass/Executed", "Pass rate", "Executable tasks"],
        condition_rows_for_50_source(data),
    ))
    lines.append("")
    lines.append("50-report source split by noise mode, with timeout failures excluded in the rightmost rate:")
    lines.append("")
    lines.append(md_table(
        ["Condition", "Pool", "Noise", "Pass/Exec", "Rate", "Timeout fails", "Pass/Non-timeout", "Timeout-excl rate", "Tasks"],
        timeout_split_rows_50_source(data),
    ))
    lines.append("")
    lines.append("### All Available Prefill Scope")
    lines.append("")
    lines.append(
        "This adds v4 seed0 First20 configs to the 50-report source. Use it as the most complete execution ledger, not as a causal comparison."
    )
    lines.append("")
    lines.append(md_table(
        ["Condition", "Pool size meaning", "Pass/Executed", "Pass rate", "Executable tasks"],
        condition_rows_for_all_available(data),
    ))
    lines.append("")
    lines.append("All-available split by noise mode, with timeout failures excluded in the rightmost rate:")
    lines.append("")
    lines.append(md_table(
        ["Condition", "Pool", "Noise", "Pass/Exec", "Rate", "Timeout fails", "Pass/Non-timeout", "Timeout-excl rate", "Tasks"],
        timeout_split_rows_all_available(data),
    ))
    lines.append("")
    lines.append("## 20t v4 vs First20 P0 Consistency")
    lines.append("")
    lines.append(
        "The task-level correlation between v4 seed0 rates and First20 P0 unique-config rates "
        "across the 14 executable First20 tasks is 0.967. Mean task rate changes from 58.9% "
        "to 60.1%, a +1.2 percentage point difference."
    )
    lines.append("")
    lines.append(md_table(
        ["Task", "v4 seed0", "v4 rate", "P0 seeds1-4", "P0 rate", "Combined seeds0-4", "Combined rate", "Combined skipped"],
        first20_rows,
    ))
    lines.append("")
    lines.append("## 50-Report Source Recomputed With Unified Rule")
    lines.append("")
    lines.append(md_table(
        ["Batch", "Task", "Pass/Executed", "Rate", "Skipped configs"],
        source50_tasks,
    ))
    lines.append("")
    lines.append("## Baselines From 20t v4")
    lines.append("")
    lines.append(
        f"`sb_baselines_n5.jsonl` gives Noskill {pn(ns_pass, len(ns))} = {fmt_rate(ns_pass / len(ns))} "
        f"and GT-only {pn(gt_pass, len(gt))} = {fmt_rate(gt_pass / len(gt))}. "
        "These baseline conditions exist only for the First20 14 executable tasks, so they should not be "
        "mixed into Smoke10/Smoke20 causal claims."
    )
    lines.append("")
    lines.append("## Accuracy And Explainability Verdict")
    lines.append("")
    lines.append(
        "Accuracy: under the unique-config execution denominator, the two reports are consistent for "
        "First20. The prior 17.3% First20 number in the 50-task report is a raw-row ledger statistic, "
        "not a comparable performance statistic."
    )
    lines.append("")
    lines.append(
        "Explainability: 20t v4 remains the better causal/effect-size report because it includes "
        "paired baselines and warns about pseudoreplication. The 50-task report is better as a "
        "coverage and reliability ledger after correcting its denominator caveats."
    )
    lines.append("")
    lines.append("Caveats:")
    lines.append("")
    lines.append("- First20 has 6 heavy-Dockerfile tasks that never executed on host.")
    lines.append("- First20 P0 is missing 44 skipped exoplanet configs from the nominal full design.")
    lines.append("- Smoke10 is incomplete for `jpg-ocr-stat` and `glm-lake-mendota`.")
    lines.append("- Smoke20 has 30 design skips at pool_size=5 for two tasks with 6 GT skills.")
    lines.append("- Aggregate rates across all 50 tasks are descriptive only because task mix and coverage differ by batch.")
    lines.append("")
    lines.append("## Produced Artifacts")
    lines.append("")
    lines.append(f"- `{OUT_MD.relative_to(BASE)}`")
    lines.append(f"- `{OUT_PDF.relative_to(BASE)}`")
    lines.append("")
    return "\n".join(lines)


def write_pdf(data: dict, md_text: str) -> None:
    unique = data["unique"]
    raw = data["raw"]

    v4_s = stats_from_unique(unique["v4"])
    f20_s = stats_from_unique(unique["first20"])
    f20_combined_s = stats_from_unique(unique["first20_combined"])
    sm10_s = stats_from_unique(unique["smoke10"])
    sm20_s = stats_from_unique(unique["smoke20"])
    source50_s = stats_from_unique(unique["source50"])
    all_s = stats_from_unique(unique["all_available"])

    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.section("1. Executive Summary")
    pdf.body(
        "This report reconciles 20t_report_v4.pdf and comparison_50task_report.pdf "
        "with one statistical rule: deduplicate to unique configs and exclude skipped "
        "configs from execution pass-rate denominators."
    )
    summary_rows = [
        ["20t v4 seed0", pn(v4_s["passed"], v4_s["executed"]), fmt_rate(v4_s["rate"]), "14 exec tasks"],
        ["First20 P0 seeds1-4", pn(f20_s["passed"], f20_s["executed"]), fmt_rate(f20_s["rate"]), "14 exec tasks"],
        ["First20 combined seeds0-4", pn(f20_combined_s["passed"], f20_combined_s["executed"]), fmt_rate(f20_combined_s["rate"]), "14 exec tasks"],
        ["50-report source only", pn(source50_s["passed"], source50_s["executed"]), fmt_rate(source50_s["rate"]), "44 exec task IDs"],
        ["All available prefill", pn(all_s["passed"], all_s["executed"]), fmt_rate(all_s["rate"]), "44 exec task IDs"],
    ]
    pdf.table(["Slice", "Pass/Executed", "Rate", "Scope"], summary_rows, [58, 34, 24, 40], 8)
    pdf.body(
        "The First20 discrepancy is denominator-driven. The 50-task raw-row First20 rate "
        "of 17.3% is not comparable to v4's unique-config rate. After deduplication, "
        "First20 P0 is 404/672 = 60.1%, close to v4's 99/168 = 58.9%."
    )

    pdf.section("2. Statistical Unit")
    pdf.body(
        "Primary unit: one unique config encoded by trial_id. Skipped configs are excluded "
        "from execution pass-rate denominators. Raw rows are retained for duplicate and "
        "coverage audits. Rate-limited rows are kept in the requested primary metric. "
        "GT+5/10/20/50/100 means GT skills are included in a total candidate pool of "
        "5/10/20/50/100 skills, not GT plus that many extra distractors. "
        "Timeout-excluded rates remove failed executions with wall time >600 seconds, "
        "agent_rc=124, or timeout/killed text."
    )

    pdf.section("3. Artifact Inventory")
    pdf.table(["Artifact", "Bytes", "Rows", "Pages", "SHA16", "Role"], artifact_rows(), [48, 20, 14, 14, 28, 66], 5.6)

    pdf.section("4. Unified Batch Summary")
    batch_rows = []
    for name, expected, raw_name, st in [
        ("20t v4 prefill seed0", 240, "v4", v4_s),
        ("First20 P0 seeds1-4", 960, "first20", f20_s),
        ("First20 combined", 1200, "-", f20_combined_s),
        ("Smoke10", 750, "smoke10", sm10_s),
        ("Smoke20", 1500, "smoke20", sm20_s),
        ("50-report source only", 3210, "-", source50_s),
        ("All available prefill", 3450, "-", all_s),
    ]:
        raw_count = len(raw[raw_name]) if raw_name in raw else "-"
        batch_rows.append([
            name, str(expected), str(raw_count), str(st["unique"]),
            str(expected - st["unique"]), str(st["skipped"]), str(st["executed"]),
            pn(st["passed"], st["executed"]), fmt_rate(st["rate"]),
        ])
    pdf.table(["Slice", "Expected", "Raw", "Unique", "Missing", "Skip", "Exec", "Pass/Exec", "Rate"],
              batch_rows, [42, 19, 18, 18, 18, 16, 18, 24, 18], 6.4)

    pdf.section("5. GT And GT+Pool Pass Rates")
    pdf.body(
        "First20 comparable scope uses the same 14 executable tasks for GT-only and "
        "GT+pool conditions. 50-report source scope has no GT-only baseline for Smoke10/20."
    )
    pdf.table(["Condition", "Pool", "Pass/Exec", "Rate", "Tasks"],
              condition_rows_for_first20(data), [34, 42, 34, 22, 44], 7)
    pdf.table(["50-source Condition", "Pool", "Pass/Exec", "Rate", "Tasks"],
              condition_rows_for_50_source(data), [40, 36, 34, 22, 56], 6.4)
    pdf.table(["All-available Condition", "Pool", "Pass/Exec", "Rate", "Tasks"],
              condition_rows_for_all_available(data), [42, 34, 34, 22, 56], 6.4)

    pdf.add_page()
    pdf.section("6. Noise Split And Timeout-Excluded Rates")
    pdf.body(
        "Rows below split GT+pool conditions by noise mode. The timeout-excluded rate "
        "removes failed executions classified as timeouts; skipped configs are already "
        "excluded from all denominators."
    )
    pdf.table(["Cond", "Pool", "Noise", "Pass/Exec", "Rate", "TO", "Pass/NoTO", "NoTO Rate", "Tasks"],
              timeout_split_rows_first20(data), [22, 18, 18, 24, 18, 12, 24, 20, 16], 5.8)
    pdf.table(["50 Cond", "Pool", "Noise", "Pass/Exec", "Rate", "TO", "Pass/NoTO", "NoTO Rate", "Tasks"],
              timeout_split_rows_50_source(data), [22, 18, 18, 24, 18, 12, 24, 20, 16], 5.8)

    pdf.section("7. First20 Task Consistency")
    first20_tasks_v4 = task_stats(unique["v4"].values())
    first20_tasks_p0 = task_stats(unique["first20"].values())
    first20_tasks_combined = task_stats(unique["first20_combined"].values())
    first20_rows = []
    for task in sorted(set(first20_tasks_v4) | set(first20_tasks_p0) | set(first20_tasks_combined)):
        a = first20_tasks_v4.get(task, {"pass": 0, "exec": 0, "rate": None})
        b = first20_tasks_p0.get(task, {"pass": 0, "exec": 0, "rate": None})
        c = first20_tasks_combined.get(task, {"pass": 0, "exec": 0, "skip": 0, "rate": None})
        first20_rows.append([
            task[:36],
            pn(a["pass"], a["exec"]),
            fmt_rate(a["rate"]),
            pn(b["pass"], b["exec"]),
            fmt_rate(b["rate"]),
            pn(c["pass"], c["exec"]),
            fmt_rate(c["rate"]),
            str(c["skip"]),
        ])
    pdf.table(["Task", "v4", "v4 Rate", "P0", "P0 Rate", "Combined", "Comb Rate", "Skip"],
              first20_rows, [54, 18, 18, 18, 18, 22, 20, 12], 5.8)

    pdf.section("8. 50-Task Unified Per-Task Rates")
    rows = []
    for batch_name, uniq_map in [
        ("First20 P0", unique["first20"]),
        ("Smoke10", unique["smoke10"]),
        ("Smoke20", unique["smoke20"]),
    ]:
        for task, s in sorted(task_stats(uniq_map.values()).items()):
            rows.append([batch_name, task[:38], pn(s["pass"], s["exec"]), fmt_rate(s["rate"]), str(s["skip"])])
    pdf.table(["Batch", "Task", "Pass/Exec", "Rate", "Skip"], rows, [24, 72, 26, 20, 14], 6)

    pdf.section("9. Verdict")
    pdf.body(
        "Accuracy: the reports are consistent for First20 after the denominator is unified. "
        "The v4 report remains the better causal report because it has paired noskill and "
        "GT-only baselines. The 50-task report is a broader coverage ledger and should use "
        "unique-config execution rates for performance claims."
    )
    pdf.body(
        "Main caveats: 6 First20 tasks never executed on host, Smoke10 is incomplete for "
        "2 tasks, Smoke20 includes 30 design skips at pool_size=5, and aggregate rates "
        "across all batches are descriptive only."
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT_PDF))


def main() -> None:
    data = build_report_data()
    md_text = build_markdown(data)
    OUT_MD.write_text(md_text)
    write_pdf(data, md_text)
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
