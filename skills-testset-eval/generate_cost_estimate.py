#!/usr/bin/env python3
"""Estimate API cost of running the full SkillsBench eval grid on claude-sonnet-4-6.

Inputs: existing trial JSONLs (system_prompt_chars, agent_wall_s).
Output: results/api_cost_estimate.pdf
"""
import json, os, statistics
from collections import defaultdict
from pathlib import Path
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
OUT_PDF = ROOT / "results" / "api_cost_estimate.pdf"

# ---------- Load empirical trial data ----------
FILES = [
    "results/sb_exec_20t.jsonl", "results/sb_baselines_20t.jsonl",
    "results/sb_prefill_n5.jsonl", "results/sb_baselines_n5.jsonl",
    "results/sb_phase_b.jsonl",   "results/sb_exec.jsonl",
]
rows = []
for f in FILES:
    p = ROOT / f
    if not p.exists(): continue
    for line in p.read_text().splitlines():
        try: rows.append(json.loads(line))
        except: pass

# Filter: real runs only (wall >= 60s excludes rate-limited 1-sec failures)
real_by_pool = defaultdict(list)   # pool_size -> [(sp_chars, wall_s)]
rate_limited = 0
total_records = 0
for r in rows:
    if r.get("skipped") or "agent_wall_s" not in r: continue
    total_records += 1
    if r["agent_wall_s"] < 5:
        rate_limited += 1
        continue
    if r["agent_wall_s"] < 60: continue
    if r.get("system_prompt_chars") is None: continue
    real_by_pool[r["pool_size"]].append((r["system_prompt_chars"], r["agent_wall_s"]))

# ---------- Cost model ----------
# Pricing: Claude Sonnet 4 family (sonnet-4-6) public API rates.
P_IN     = 3.00 / 1e6
P_OUT    = 15.00 / 1e6
P_CWRITE = 3.75 / 1e6   # 5-min ephemeral cache
P_CREAD  = 0.30 / 1e6

CHARS_PER_TOK = 3.5      # English+code mix
USER_PROMPT_TOK = 600
TOOL_RESULT_TOK_PER_TURN = 1200
WALL_SEC_PER_TURN = 12   # observed Sonnet 4.6 turn cadence with tool exec
OUTPUT_TOK_PER_TURN = 500

def cost_per_trial(sp_chars: float, wall_s: float):
    """Single `claude -p` invocation with growing cache prefix."""
    S = sp_chars / CHARS_PER_TOK
    U = USER_PROMPT_TOK
    N = max(1, int(round(wall_s / WALL_SEC_PER_TURN)))
    T = TOOL_RESULT_TOK_PER_TURN
    cache_written = S + U
    cache_read    = 0.0
    output_tok    = OUTPUT_TOK_PER_TURN
    for i in range(2, N + 1):
        prev = S + U + (i - 2) * T
        cache_read    += prev
        cache_written += T
        output_tok    += OUTPUT_TOK_PER_TURN
    cost = cache_written * P_CWRITE + cache_read * P_CREAD + output_tok * P_OUT
    in_tok = cache_written + cache_read
    return dict(cost=cost, N=N, in_tok=in_tok, out_tok=output_tok,
                cache_written=cache_written, cache_read=cache_read, S=S)

# Compute per-pool reference cost (median over real trials)
per_pool = {}
for pool, vals in real_by_pool.items():
    sp_med   = statistics.median(v[0] for v in vals)
    wall_med = statistics.median(v[1] for v in vals)
    wall_mean = statistics.mean(v[1] for v in vals)
    sp_mean = statistics.mean(v[0] for v in vals)
    m = cost_per_trial(sp_med, wall_med)
    per_pool[pool] = dict(n=len(vals), sp_med=sp_med, sp_mean=sp_mean,
                          wall_med=wall_med, wall_mean=wall_mean, **m)

# ---------- Full-grid plan ----------
# run.sh --full: 84 tasks × (pool_sizes=[5,10,20,50]) × (noise=[random,hard,easy]) × seed=0
TASKS = 84
POOL_NOISE = {5: 3, 10: 3, 20: 3, 50: 3}
# Baselines (separate runs): 84 tasks × (noskill, gtonly), pool=0
BASELINE_MODES = 2

# Some trials skipped because |GT skills| > pool_size; for small pool, ~5-10% skip rate.
# We'll be conservative and not subtract these (overestimates slightly).

per_pool_cost = {}
for pool, n_noise in POOL_NOISE.items():
    if pool in per_pool:
        c = per_pool[pool]["cost"]
    else:
        c = cost_per_trial(pool * 850, 240)["cost"]  # extrapolate
    per_pool_cost[pool] = (c, n_noise, TASKS * n_noise, TASKS * n_noise * c)

baseline_c = per_pool[0]["cost"] if 0 in per_pool else cost_per_trial(300, 240)["cost"]
baseline_cost = TASKS * BASELINE_MODES * baseline_c

main_grid_cost = sum(v[3] for v in per_pool_cost.values())
total_baseline_cost = main_grid_cost + baseline_cost

# Sensitivity: vary turn count and output verbosity
def grid_total(turn_mult: float, out_mult: float) -> float:
    t = 0.0
    for pool, n_noise in POOL_NOISE.items():
        sp = per_pool.get(pool, dict(sp_med=pool*850))["sp_med"]
        wall = per_pool.get(pool, dict(wall_med=240))["wall_med"] * turn_mult
        m = cost_per_trial(sp, wall)
        # Scale output portion separately
        out_extra = (out_mult - 1.0) * m["out_tok"] * P_OUT
        t += TASKS * n_noise * (m["cost"] + out_extra)
    sp0   = per_pool.get(0, dict(sp_med=300))["sp_med"]
    wall0 = per_pool.get(0, dict(wall_med=240))["wall_med"] * turn_mult
    m0 = cost_per_trial(sp0, wall0)
    out_extra0 = (out_mult - 1.0) * m0["out_tok"] * P_OUT
    t += TASKS * BASELINE_MODES * (m0["cost"] + out_extra0)
    return t

scenarios = [
    ("Low  (fewer turns, terse output)", 0.6, 0.7,
     "Most tasks solved in <10 turns; agent emits compact summaries."),
    ("Baseline (median observed)",       1.0, 1.0,
     "Per-pool median wall-time; 12s/turn; 500 out-tok/turn."),
    ("High (long retries, verbose)",     1.5, 1.4,
     "Heavy debugging; agent verbose; cache miss on retries."),
]

# ---------- Build PDF ----------
class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 7, "SkillsBench Full-Run API Cost Estimate (claude-sonnet-4-6)",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")

pdf = Report()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

def H(s, sz=12):
    pdf.set_font("Helvetica", "B", sz); pdf.cell(0, 6, s, new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
def P(s, sz=9):
    pdf.set_font("Helvetica", "", sz); pdf.multi_cell(0, 4.5, s); pdf.ln(1)
def MONO(s, sz=8):
    pdf.set_font("Courier", "", sz); pdf.multi_cell(0, 3.8, s); pdf.ln(1)

# Headline
H("Headline", 13)
P(f"Estimated cost to run the full SkillsBench grid via the public Anthropic API "
  f"on claude-sonnet-4-6 (with prompt caching): "
  f"~${total_baseline_cost:,.0f} USD baseline; ${grid_total(0.6,0.7):,.0f}-${grid_total(1.5,1.4):,.0f} sensitivity range.")
P(f"Main grid only (no baselines): ${main_grid_cost:,.0f}. "
  f"Baselines only (noskill+gtonly): ${baseline_cost:,.0f}.")

# Scope
H("Run scope (from run.sh --full + run_baselines.sh)")
P(f"- Main grid: {TASKS} SkillsBench tasks x 4 pool sizes (5/10/20/50) x 3 noise modes (random/hard/easy) x 1 seed "
  f"= {TASKS*12} trials.")
P(f"- Baselines: {TASKS} tasks x 2 modes (noskill, gtonly), pool=0 = {TASKS*BASELINE_MODES} trials.")
P(f"- Grand total: {TASKS*12 + TASKS*BASELINE_MODES} agent invocations (minus tasks skipped because |GT skills| > pool_size, ~5%).")

# Empirical inputs
H("Empirical inputs (from already-completed runs)")
P(f"Examined {len(rows):,} trial records across 6 result files. "
  f"After filtering out {rate_limited:,} 1-second 'You've hit your limit' rate-limited entries "
  f"(Max-plan throttling, not real runs), {sum(len(v) for v in real_by_pool.values()):,} real runs remain. "
  f"Per pool size, the median observed values:")

# Table 1: empirical per-pool stats
pdf.ln(1); pdf.set_font("Courier", "B", 8)
pdf.cell(0,4,"  pool   n   sp_chars_med   wall_med_s   est_turns", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Courier","",8)
for pool in sorted(per_pool):
    d = per_pool[pool]
    pdf.cell(0,4,f"  {pool:>4} {d['n']:>4} {int(d['sp_med']):>13,} {d['wall_med']:>11.0f}   {d['N']:>5}",
             new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)

# Cost model
H("Cost model")
P("Pricing (claude-sonnet-4-6, public API, USD/M tokens):")
MONO("  input          $3.00     output         $15.00\n"
     "  cache write    $3.75     cache read     $0.30")
P("Per-trial billing for one `claude -p` invocation with a growing cache prefix:")
MONO(f"  S  = system_prompt_chars / {CHARS_PER_TOK} tokens\n"
     f"  U  = {USER_PROMPT_TOK} tokens (task instruction)\n"
     f"  T  = {TOOL_RESULT_TOK_PER_TURN} tokens (avg Bash/Read tool result per turn)\n"
     f"  O  = {OUTPUT_TOK_PER_TURN} tokens (avg assistant output per turn)\n"
     f"  N  = wall_s / {WALL_SEC_PER_TURN} (turn count, derived from wall time)\n"
     f"\n"
     f"  Turn 1: cache_write += S+U,  output += O\n"
     f"  Turn i>=2: cache_read += S+U+(i-2)*T,  cache_write += T,  output += O\n"
     f"\n"
     f"  cost = cache_write * $3.75/M + cache_read * $0.30/M + output * $15/M")

# Per-condition cost
pdf.add_page()
H("Per-condition cost")
pdf.set_font("Courier","B",8)
pdf.cell(0,4,"  pool noise_modes  n_trials   $/trial    $/condition_group", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Courier","",8)
for pool in sorted(POOL_NOISE):
    c, nn, nt, total = per_pool_cost[pool]
    pdf.cell(0,4,f"  {pool:>4}     {nn:>3}        {nt:>6}     ${c:>6.4f}        ${total:>8.2f}",
             new_x="LMARGIN", new_y="NEXT")
pdf.cell(0,4,f"   0       2        {TASKS*BASELINE_MODES:>6}     ${baseline_c:>6.4f}        ${baseline_cost:>8.2f}",
         new_x="LMARGIN", new_y="NEXT")
pdf.cell(0,4,"  " + "-"*60, new_x="LMARGIN", new_y="NEXT")
pdf.cell(0,4,f"  TOTAL                {TASKS*12 + TASKS*BASELINE_MODES:>6}                  ${total_baseline_cost:>8.2f}",
         new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)

# Token totals
H("Aggregate token volume (baseline scenario)")
total_in = 0.0; total_out = 0.0; total_cw = 0.0; total_cr = 0.0
for pool, n_noise in POOL_NOISE.items():
    m = per_pool[pool] if pool in per_pool else None
    if not m: continue
    n = TASKS * n_noise
    total_cw += m["cache_written"] * n
    total_cr += m["cache_read"] * n
    total_out += m["out_tok"] * n
if 0 in per_pool:
    m = per_pool[0]; n = TASKS * BASELINE_MODES
    total_cw += m["cache_written"] * n
    total_cr += m["cache_read"] * n
    total_out += m["out_tok"] * n
total_in = total_cw + total_cr

MONO(f"  cache writes : {total_cw/1e6:>8.2f} M tokens   (${total_cw*P_CWRITE:>7.2f})\n"
     f"  cache reads  : {total_cr/1e6:>8.2f} M tokens   (${total_cr*P_CREAD:>7.2f})\n"
     f"  output       : {total_out/1e6:>8.2f} M tokens   (${total_out*P_OUT:>7.2f})\n"
     f"  -----------------------------------------------\n"
     f"  total input  : {total_in/1e6:>8.2f} M tokens\n"
     f"  total output : {total_out/1e6:>8.2f} M tokens\n"
     f"  TOTAL COST   : ${total_baseline_cost:>7.2f}")

# Sensitivity
H("Sensitivity")
pdf.set_font("Courier","B",8)
pdf.cell(0,4,"  scenario                            $total", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Courier","",8)
for label, tm, om, _ in scenarios:
    pdf.cell(0,4,f"  {label:<35} ${grid_total(tm, om):>7.2f}", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)
for label, tm, om, note in scenarios:
    P(f"   {label}: {note}", 8)

# Caveats
H("Caveats and how to anchor this number")
P("1) Trial harness uses `claude -p --no-session-persistence --output-format text`, so no token counts were saved with the 967 real runs that already happened. Numbers above are model-based, not measured.")
P("2) Real exact cost can be obtained cheaply by adding `--output-format json` to call_claude() in testsets/exec_eval_prefill/run_trial.py, then running ~10 representative trials. The JSON envelope includes usage.input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens per trial. Multiply averages by the planned trial count for a measured estimate.")
P("3) Pricing assumes claude-sonnet-4-6 public API rates with 5-min ephemeral prompt caching. Switching to 1-hour cache (6.00/M write) would raise cost ~15%; opting out of caching entirely would roughly 5-10x the input bill on long ReAct sessions.")
P("4) Skipped trials (|GT skills| > pool_size) are not subtracted; that overstates cost by ~3-5%.")
P("5) The friend's hint 'should be less than what you calculated' likely refers to: (a) prompt caching saves more than naive math implies on multi-turn ReAct, (b) median successful runs finish well before the agent_wall_s tail. Both factors are folded into the median-based baseline above; an even tighter estimate is in the Low scenario column.")

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
pdf.output(str(OUT_PDF))
print(f"\nWrote {OUT_PDF}")
print(f"Baseline total: ${total_baseline_cost:,.2f}  "
      f"(low ${grid_total(0.6,0.7):,.0f}, high ${grid_total(1.5,1.4):,.0f})")
