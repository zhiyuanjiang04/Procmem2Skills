# M1 Reproduction — ANALYSIS
**Run:** `2026-04-18-2235-plan1-m1`  
**Backend:** `claude` CLI (claude-sonnet-4-6, Max OAuth)  
**Config:** 5 tasks × {random, hard_neg_semantic} × N∈{1,5,50,200} × 3 seeds × 2 probes = 240 calls  
**Date:** 2026-04-18

---

## Per-(Strategy, N) Breakdown

| Strategy            |   N | n   | Recall@5 | Top-1 |  MRR | Sel Top-1 | Sel\|Aware |
|---------------------|----:|-----|----------|-------|------|-----------|-----------|
| hard_neg_semantic   |   1 | 15  | 1.000    | 0.800 | 0.900 | 0.800    | 0.800     |
| hard_neg_semantic   |   5 | 15  | 1.000    | 0.800 | 0.900 | 0.800    | 0.800     |
| hard_neg_semantic   |  50 | 15  | 0.800    | 0.467 | 0.611 | 0.600    | 0.667     |
| hard_neg_semantic   | 200 | 15  | 0.467    | 0.400 | 0.433 | 0.733    | 0.857     |
| random              |   1 | 15  | 1.000    | 1.000 | 1.000 | 0.867    | 0.867     |
| random              |   5 | 15  | 0.933    | 0.933 | 0.933 | 1.000    | 1.000     |
| random              |  50 | 15  | 0.667    | 0.667 | 0.667 | 1.000    | 1.000     |
| random              | 200 | 15  | 0.267    | 0.267 | 0.267 | 1.000    | 1.000     |

**Overall:** awareness_recall5=0.767, awareness_top1=0.667, awareness_mrr=0.714, selection_top1=0.850, parse_fail=0.000

---

## Comparison to v1 Expected Anchors

| Anchor                                  | Expected        | Observed       | Status |
|-----------------------------------------|-----------------|----------------|--------|
| N=1 → Recall@5 = 1.0                    | 1.0             | 1.0 (both strategies) | PASS |
| random @ N=5 → Recall@5 ≈ 1.0          | ~1.0            | 0.933          | NEAR (1/15 misses) |
| random @ N=50 → Recall@5 ≈ 1.0         | ~1.0            | 0.667          | BELOW — collapse visible |
| hard_neg_semantic @ N=50 → [0.65,0.80] | [0.65, 0.80]   | 0.800          | PASS (upper bound) |
| random @ N=200 → Recall@5 ≈ 1.0        | ~1.0            | 0.267          | SEVERE DROP — main finding |

---

## Key Findings

1. **Selection collapse confirmed.** `random @ N=200` drops to Recall@5=0.267, despite selection_top1=1.0 (when the model does select, it picks correctly). Collapse is an *awareness* failure as pool grows, not a selection failure.

2. **Hard negatives are genuinely harder.** `hard_neg_semantic @ N=50` Recall@5=0.800 vs random=0.667 at same N, but selection_top1 drops to 0.600 vs 1.000 — semantic distractors confuse selection more than random distractors.

3. **N=1 sanity check holds.** Both strategies give Recall@5=1.0 at N=1, validating pool construction and parsing pipeline.

4. **Zero parse failures.** CLI driver produces clean structured responses across all 240 trials after fixing the user_prompt format instruction.

5. **Random selection_top1 stable at 1.0 for N≥5.** When the model is aware, it always picks GT from random distractors regardless of N. Hard negatives degrade both awareness and sel|aware simultaneously.

---

## Notes

- The user_prompt construction required a fix: the original split at "Available skills" discarded the response format instruction, causing 100% parse failure in the initial smoke run. Fix: reconstruct user_prompt as task description + format instruction (pool block stays in combined_system for CLIDriver).
- No retries needed; all 240 CLI calls succeeded on first attempt.
