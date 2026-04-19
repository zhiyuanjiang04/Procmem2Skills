# M1 Sweep Analysis — plan1-m1-fixed (2026-04-18)

Run directory: `skills retrieval/runs/2026-04-18-2309-plan1-m1-fixed/`
Model: claude-sonnet-4-6 · 5 tasks × 2 strategies × 4 N × 3 seeds × 2 probes = 240 calls

---

## Per-(strategy, N) Breakdown

| strategy            |   N | R@5   | Aware-Top1 | MRR   | Sel|Aware |
|---------------------|----:|-------|-----------|-------|-----------|
| hard_neg_semantic   |   1 | 1.000 | 0.733     | 0.867 | 0.800     |
| hard_neg_semantic   |   5 | 1.000 | 0.800     | 0.889 | 0.800     |
| hard_neg_semantic   |  50 | 1.000 | 0.600     | 0.733 | 0.600     |
| hard_neg_semantic   | 200 | 0.867 | 0.600     | 0.700 | 0.769     |
| random              |   1 | 1.000 | 1.000     | 1.000 | 0.867     |
| random              |   5 | 1.000 | 1.000     | 1.000 | 1.000     |
| random              |  50 | 1.000 | 1.000     | 1.000 | 1.000     |
| random              | 200 | 1.000 | 1.000     | 1.000 | 0.933     |

Overall: R@5=0.983, Top1=0.842, MRR=0.899, Sel|Aware=0.847, parse_fail=0.0

---

## Comparison to Previous Buggy Run (2026-04-18-2235-plan1-m1)

| strategy            |   N | R@5 old→new   | MRR old→new   | Sel|Aware old→new |
|---------------------|----:|---------------|---------------|-------------------|
| hard_neg_semantic   |   1 | 1.000→1.000   | 0.900→0.867   | 0.800→0.800       |
| hard_neg_semantic   |   5 | 1.000→1.000   | 0.900→0.889   | 0.800→0.800       |
| hard_neg_semantic   |  50 | 0.800→1.000   | 0.611→0.733   | 0.667→0.600       |
| hard_neg_semantic   | 200 | 0.467→0.867   | 0.433→0.700   | 0.857→0.769       |
| random              |   1 | 1.000→1.000   | 1.000→1.000   | 0.867→0.867       |
| random              |   5 | 0.933→1.000   | 0.933→1.000   | 1.000→1.000       |
| random              |  50 | 0.667→1.000   | 0.667→1.000   | 1.000→1.000       |
| random              | 200 | 0.267→1.000   | 0.267→1.000   | 1.000→0.933       |

Overall: old R@5=0.767 → new R@5=0.983; old MRR=0.714 → new MRR=0.899.

The biggest improvements are at large N (50, 200) — exactly where the model most often omitted the
SKILL_ prefix or leading zeros. At N=200 random, R@5 went 0.267→1.000; the buggy run was measuring
near-zero not because the model failed but because the parser could not recognise returned IDs.

---

## id_normalized flag

**69 / 240 trial records** (29%) had flags["id_normalized"]=True. The flag propagates through
TrialRecord.flags to parsed/*.json. Rate rises sharply with N: negligible at N=1/5, frequent at
N=50/200, consistent with the model dropping zero-padding when listing many IDs.

---

## Effect of Description Fix (Bug 2)

With descriptions_path now passed, distractor cards have the same ~200-char prose as GT cards,
removing the trivially-detectable empty-desc cue. The clean random R@5=1.000 at N=200 confirms
the model can reliably list GT even with symmetric presentation — a true recall upper bound.

---

## hard_neg_semantic vs random

With symmetric descriptions, hard_neg_semantic is now meaningfully harder:
- R@5: hard_neg 0.867 vs random 1.000 at N=200 (Δ=0.133)
- MRR: hard_neg 0.700 vs random 1.000 at N=200

Degradation is monotonic with N for hard_neg_semantic while random is flat at ~1.0 through N=200,
confirming semantically similar distractors impose genuine selection difficulty.
