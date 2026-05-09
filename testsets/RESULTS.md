# Skill-selection eval — results log

## 2026-05-03: SkillsBench full grid

Run: `testsets/runs/2026-05-03-grid-full.jsonl`
Config: 89 tasks × {5, 50, 500} pool size × {random, hard, easy, none} noise × seed 0 = **1068 trials, 0 errors**.
Backend: `claude -p` Max plan, model `claude-sonnet-4-5`, conc=6, ~48 min wall-clock.
Pool builder: `max_gt=1` (subsample to 1 GT/task), GT-alias-aware ClawHub exclusion, post-shuffle GT positions recorded.

| size | noise   | n  | Hit@1   | Recall  | MRR    | refusal |
|------|---------|----|---------|---------|--------|---------|
| 5    | random  | 89 | 80.90%  | 80.90%  | 0.809  | 17.98%  |
| 5    | hard    | 89 | 59.55%  | 60.67%  | 0.601  | 10.11%  |
| 5    | easy    | 89 | 85.39%  | 85.39%  | 0.854  | 14.61%  |
| 5    | none    | 89 | —       | —       | —      | **97.75%** |
| 50   | random  | 89 | 83.15%  | 83.15%  | 0.831  | 15.73%  |
| 50   | hard    | 89 | 55.06%  | 55.06%  | 0.551  | 12.36%  |
| 50   | easy    | 89 | 80.90%  | 80.90%  | 0.809  | 17.98%  |
| 50   | none    | 89 | —       | —       | —      | **95.51%** |
| 500  | random  | 89 | 76.40%  | 76.40%  | 0.764  | 20.22%  |
| 500  | hard    | 89 | **41.57%** | 41.57% | 0.416 | 12.36%  |
| 500  | easy    | 89 | 71.91%  | 71.91%  | 0.719  | 28.09%  |
| 500  | none    | 89 | —       | —       | —      | 82.02%  |

### Headline findings

1. **Selection collapse under hard noise.** Hard-neg distractors degrade Hit@1 from 59.6% (sz=5) → 41.6% (sz=500), an 18 pp drop. The model genuinely cannot distinguish GT from its tightest embedding neighbours at scale.
2. **Random / easy noise are mostly survivable.** ~10 pp drop from sz=5 to sz=500 — the model handles uncorrelated distraction.
3. **Refusal in no-GT condition is robust at small/medium sizes (≥95%) but breaks at sz=500 (82%).** The model "hallucinates" a pick in 18% of large no-GT pools — a clear retrieval-system safety concern.
4. **Easy noise at sz=5 gives the highest Hit@1 (85.4%)** — when distractors are blatantly wrong, GT pops out cleanly. This is the "ceiling" reference for the harness.

### Position-bias sanity (separate run, sz=500 random, n=9/bin)
Hit@1: early 100% / middle 78% / late 89%. Mild middle-position dip; no catastrophic attention collapse — confirms the 500-pool numbers are not a positional artifact.

### Caveats
- `max_gt=1` simplifies but ignores multi-GT tasks (some SkillsBench tasks legitimately need 3–5 skills).
- Single seed; variance not estimated. Re-run with seeds {0,1,2} for confidence intervals before paper.
- 25.5% of GT skills have a ClawHub alias at sim ≥ 0.9; the pool builder excludes the aliased corpus ID from noise, so misses in `hard` mode at the alias boundary represent the model flipping between embedding-equivalent skills (not a bug — that *is* the failure mode being measured).

---

## 2026-05-03: Terminal-Bench pilot (P4 scope: no execution, no validated GT)

Run: `testsets/runs/2026-05-03-tb-pilot.jsonl`
Config: 30 TB tasks × {5, 50, 500} pool size × {topk_only, random, hard, easy, none} noise × seed 0 = **450 trials, 0 errors**.
Pool: top-5 FAISS retrievals from ClawHub serve as the proxy candidate set; treated as "intended" rather than validated GT (P4 dropped harbor execution).

| size | noise      | n  | pick  | in_topk | top1=ret | refusal |
|------|-----------|----|-------|---------|----------|---------|
| 5    | topk_only | 30 | 33.3% | 33.3%   | 16.7%    | 66.7%   |
| 5    | random    | 30 | 33.3% | 33.3%   | 20.0%    | 66.7%   |
| 5    | hard      | 30 | 26.7% | 26.7%   | 20.0%    | 73.3%   |
| 5    | easy      | 30 | 33.3% | 33.3%   | 20.0%    | 66.7%   |
| 5    | none      | 30 | 0.0%  | —       | —        | 100.0%  |
| 50   | random    | 30 | 26.7% | 26.7%   | 20.0%    | 73.3%   |
| 50   | hard      | 30 | 40.0% | 23.3%   | 16.7%    | 60.0%   |
| 50   | easy      | 30 | 23.3% | 23.3%   | 16.7%    | 76.7%   |
| 50   | topk_only | 30 | 33.3% | 30.0%   | 13.3%    | 66.7%   |
| 50   | none      | 30 | 3.3%  | —       | —        | 96.7%   |
| 500  | random    | 30 | 20.0% | 13.3%   | 6.7%     | 80.0%   |
| 500  | hard      | 30 | 33.3% | 16.7%   | 6.7%     | 66.7%   |
| 500  | easy      | 30 | 13.3% | 13.3%   | 6.7%     | 86.7%   |
| 500  | topk_only | 30 | 30.0% | 30.0%   | 13.3%    | 70.0%   |
| 500  | none      | 30 | 16.7% | —       | —        | 83.3%   |

(`pick`=non-refusal rate; `in_topk`=top-1 picked is one of the FAISS top-5; `top1=ret`=top-1 picked is the FAISS top-1; `refusal`=model emitted NONE.)

### Headline findings

1. **Corpus–task mismatch is fundamental.** Even `topk_only` at sz=5 — a model handed exactly the FAISS-most-similar 5 ClawHub skills with no noise — refuses 67% of the time. ClawHub does not actually cover most TB task domains; the model knows it.
2. **`agree_top1` ≤ 20% across all conditions.** When the model does pick, it rarely picks the FAISS top-1; it has independent judgement about which retrieved candidate (if any) is right.
3. **Hard noise *increases* pick rate at sz=50 (40%) over topk_only (33%).** Hard-neg distractors look superficially more relevant than the actual FAISS top-K, so the model is more willing to commit. At sz=500 the same effect: hard 33% pick rate beats random 20% and easy 13%.
4. **No-GT hallucination at sz=500: 16.7%.** Same pattern as SkillsBench but lower magnitude (SB was 18%) — TB's task descriptions are more specific, helping refusal even in pure-noise pools.

### Implications

- The TB testset's "validated GT" is genuinely empty for most tasks; ~67% of TB tasks have no ClawHub skill the model would accept. This justifies P4's drop of incremental injection — it would have produced "no skill helps" labels for the same ~67% regardless of execution.
- For the controlled-study paper, TB serves best as a **negative control / OOD corpus**: the model's selection collapse on SkillsBench shows it can fail even when GT exists; on TB it shows the model correctly refuses when GT *doesn't* exist (mostly). Together that's a stronger story than either alone.

### Caveats
- 30 tasks (12.4% of TB) — full 241 should be run for paper-grade numbers.
- Single seed.
- "topk_only" pool ordering is rank-preserved (not shuffled); other modes shuffle. Need a `topk_only_shuffled` variant to verify ranking signal isn't just position bias.

---

## 2026-05-04: Terminal-Bench validated GT via Opus LLM-as-judge

Run: `testsets/runs/2026-05-04-tb-judge.jsonl` → `testsets/data/terminal_bench_validated.jsonl`
Method: for each TB task × top-5 ClawHub FAISS retrieval = 1205 (task,skill) pairs, ask Opus 4.7 (Max-plan `claude -p`, conc=4): "would loading this skill materially help an autonomous shell agent solve the task?". YES → ground-truth skill; tasks with all-NO are dropped.

Replaces the originally-spec'd "incremental skill injection via harbor" GT-construction loop. Harbor was ruled out because (a) harbor's terminus-2-skills agent calls models via LiteLLM/API only (no Max-plan support), so 241 × 6 task-runs would have cost ~$700–$1.7K; (b) per `feedback_retrieval_isolation.md`, agent-execution-as-GT was previously forbidden. LLM-as-judge keeps GT construction in pure forward-pass methodology while still labelling at the (task, skill) granularity.

| metric                              | value                            |
|-------------------------------------|----------------------------------|
| (task, skill) pairs judged          | 1205 (241 × 5)                   |
| YES verdicts                        | 98 (8.1%)                        |
| NO verdicts                         | 1106 (91.8%)                     |
| Unparseable (Anthropic AUP refusal) | 1 (treated as NO)                |
| Tasks kept (≥ 1 YES)                | **62 / 241 (25.7%)**             |
| Tasks dropped                       | 179 (74.3%)                      |
| Multi-GT tasks (≥ 2 YES)            | 22                               |
| GT-count distribution               | 1: 40, 2: 12, 3: 7, 4: 2, 5: 1   |
| YES distribution by FAISS rank      | r1: 33, r2: 21, r3: 14, r4: 18, r5: 12 |
| Wall-clock                          | ~25 min (Opus 4.7 via Max plan)  |
| Cost                                | $0 marginal (Max plan)           |

### Headline findings

1. **74% TB-task drop rate confirms the corpus–task mismatch.** ClawHub does not cover most terminal-bench task domains; even with strict top-5 retrieval, only 1 in 4 TB tasks has a single ClawHub skill that passes a strict-relevance bar. Consistent with the 67% refusal rate observed in the 2026-05-03 prompt-and-parse pilot — the same coverage gap surfaces under both methodologies.
2. **YES verdicts span all 5 ranks (33/21/14/18/12), not concentrated at rank 1.** FAISS embedding similarity has signal but is not a clean ranker — 65 of 98 GT skills are at rank ≥ 2. This is exactly the dataset shape needed to evaluate whether the eval-target model's selection beats raw retrieval ordering.
3. **Multi-GT structure (22 tasks have 2–5 GT skills)** preserves the recall-vs-precision trade-off; the eval harness's `validated_gt_slugs` field can be scored as a set, not just top-1.

### Caveats

- Single judge model (Opus 4.7). For a paper-grade label, run a second-model agreement check (Sonnet or non-Anthropic) and treat the intersection as high-precision GT, the union as recall-leaning GT.
- Strict prompt may over-reject niche but valid skills. Spot-checking 10 NO verdicts before scaling decisions.
- Judge sees only skill name + ~200-char description from the corpus, not the full SKILL.md (the corpus's `data/raw/skills/_listing.jsonl` only stores summaries). Adding full-text retrieval would likely raise the YES rate.
- 62-task validated TB testset is small. Combine with the 88-task SkillsBench validated set for a 150-task evaluation pool when running the prompt-and-parse harness end-to-end.

---

## 2026-05-04: Terminal-Bench validated-grid eval (point 4c/5)

Run: `testsets/runs/2026-05-04-tb-validated-grid.jsonl` (summary `.summary.json`)
Config: 62 validated TB tasks × {5, 50, 500} pool size × {random, hard, easy, topk_only, none} noise × seed 0 = **930 trials, 0 errors**.
Backend: `claude -p` Max plan, model `claude-sonnet-4-5`, conc 6 (initial) / 4 (after rate-limit resume), wall-clock ~80 min.
Pool builder: GT slug pulled from `validated_gt_slugs`; multi-GT scoring (Hit@1 = top-1 in GT set; Recall = |picked ∩ GT| / |GT|; MRR over picked-list).

| size | noise      | n  | Hit@1   | Recall | MRR   | refusal | pick   | in_topk |
|------|-----------|----|---------|--------|-------|---------|--------|---------|
| 5    | random    | 62 | 74.2%   | 58.1%  | 0.742 | 12.9%   | 87.1%  | 87.1%   |
| 5    | hard      | 62 | 69.4%   | 54.1%  | 0.694 | 17.7%   | 82.3%  | 82.3%   |
| 5    | easy      | 62 | 75.8%   | 58.9%  | 0.758 | 17.7%   | 82.3%  | 82.3%   |
| 5    | topk_only | 62 | 74.2%   | 57.6%  | 0.742 | 16.1%   | 83.9%  | 83.9%   |
| 5    | none      | 62 | —       | —      | —     | **100.0%** | 0.0% | 0.0%    |
| 50   | random    | 62 | 75.8%   | 59.5%  | 0.758 | 12.9%   | 87.1%  | 85.5%   |
| 50   | hard      | 62 | 54.8%   | 45.1%  | 0.548 | 19.4%   | 80.6%  | 59.7%   |
| 50   | easy      | 62 | 75.8%   | 57.0%  | 0.758 | 17.7%   | 82.3%  | 82.3%   |
| 50   | topk_only | 62 | **80.6%** | 64.3% | 0.806 | 11.3%   | 88.7%  | 88.7%   |
| 50   | none      | 62 | —       | —      | —     | 96.8%   | 3.2%   | 0.0%    |
| 500  | random    | 62 | 58.1%   | 42.8%  | 0.581 | 30.6%   | 69.4%  | 62.9%   |
| 500  | hard      | 62 | **41.9%** | 34.2% | 0.419 | 27.4%   | 72.6%  | 43.5%   |
| 500  | easy      | 62 | 48.4%   | 36.2%  | 0.484 | 40.3%   | 59.7%  | 56.5%   |
| 500  | topk_only | 62 | 79.0%   | 63.5%  | 0.798 | 17.7%   | 82.3%  | 82.3%   |
| 500  | none      | 62 | —       | —      | —     | 83.9%   | 16.1%  | 0.0%    |

### Key takeaway

**The model underperforms a trivial embedding-argmax baseline under hard distractors, despite the correct skill being among the top-5 candidates. This indicates a failure of fine-grained discrimination rather than retrieval.** See [`figures/collapse_analysis.pdf`](figures/collapse_analysis.pdf) for the embedding-neighborhood diagnostics (Figure 1).

| baseline / picker | Hit@1 |
|--|--|
| random-from-top-5 (no LLM, uniform) | 31.3% |
| **LLM @ sz=500 hard** | **41.9%** |
| LLM @ sz=500 easy | 48.4% |
| **embedding-argmax oracle** (no LLM, deterministic) | **53.2%** |
| LLM @ sz=50 topk_only (clean condition) | 80.6% |

Wrong picks concentrate near the task in embedding space: 24.7% are the corpus argmax itself, 71.9% are in the corpus top-10, median rank = 4 of 44,787. 49% of wrong picks satisfy `|sim(GT,task) − sim(picked,task)| < 0.05`; 28% of wrong picks have `sim(picked) > sim(GT)`.

### Headline findings

1. **Validated GT recovers a real signal**: Hit@1 in the small-pool / topk_only setting (74-81%) is now in the same ballpark as SkillsBench's GT-aware grid (76-85% at sz=5 random) — confirming the Opus-judge GT is a meaningful target, not noise.
2. **Selection-collapse pattern reproduces on TB**: hard-noise Hit@1 falls 69.4% (sz=5) → 54.8% (sz=50) → **41.9%** (sz=500) — a 27.5 pp drop, the same shape as SkillsBench (59.6% → 41.6%). The collapse-under-hard-noise phenomenon is now a *cross-dataset* finding.
3. **topk_only is the stability ceiling**: at sz=50 (80.6%) and sz=500 (79.0%) it nearly equals sz=5 (74.2%) — when the top-K ordering itself is preserved (no shuffled noise), model handles huge pools fine. Confirms the failure mode is *distractors mixed near GT*, not pool size per se.
4. **No-GT refusal degrades with pool size**: 100% (sz=5) → 96.8% (sz=50) → 83.9% (sz=500). Identical to SkillsBench (97.8% / 95.5% / 82.0%). The 16% sz=500 hallucination rate is robust across both datasets — a real retrieval-system safety concern.
5. **Recall mostly tracks Hit@1** because most validated tasks are single-GT (40 of 62); multi-GT tasks (22) are where Recall lags Hit@1 (e.g., sz=50 topk_only: 80.6% Hit@1 but 64.3% Recall).

### Caveats

- Single seed; need {0,1,2} replication for variance bars.
- 62 tasks limits statistical power vs. SkillsBench's 89.
- "topk_only" pool ordering is rank-preserved (not shuffled). Position bias not isolated; could be a follow-up.
- Initial run hit Max-plan rate limit at 747/930 (5:40pm Indianapolis); resumed at conc=4 with `--resume` and finished cleanly.

### Dataset construction note (read before citing any numbers)

Ground-truth skills are selected via an LLM judge (Opus 4.7) from the top-5 embedding-retrieved candidates per task. **All results are therefore conditioned on the correct skill being highly retrievable** — i.e., within the top-5 by embedding similarity to the task description. Concretely, of the 62 validated tasks, 33 have `GT-rank-in-corpus = 1` and 29 have `GT-rank-in-corpus ∈ [2, 5]`; none lie deeper than rank 5 by construction.

Findings should be interpreted as: **even under this favorable retrievability regime, the model fails to reliably select the correct skill under hard distractors.** The 179 tasks dropped at the GT-construction step are out of scope for the validated-grid claims; they are characterised separately in the 2026-05-03 TB-pilot findings as a coverage-gap / refusal-calibration story.


