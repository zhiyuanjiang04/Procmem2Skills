# Awareness Collapse — Controlled Study of Agent Skill Retrieval (v2 Design)

**Date:** 2026-04-18
**Owner:** Rui
**Project:** ProcMem2Skills — Skill Retrieval workstream
**Supersedes:** `skills retrieval/design.md` (v1, sketch)
**Output root:** `skills retrieval/`

---

## 0. Context

A v1 pilot (5 SkillsBench tasks) and v1 scale-up (47 tasks, 3 seeds) were run between 2026-04-04 and 2026-04-06 against `claude-haiku-4-5` via Claude Code subagent dispatch. Results are at `data/noisy_retrieval/{PILOT_ANALYSIS,SCALE_UP_ANALYSIS}.md`. Two findings force a v2 re-design:

1. **Random distractors do not induce collapse on SkillsBench.** GT skill names are too keyword-distinctive; Recall@5 stays at ~100% up to N=1000. Pool-size scaling alone is a null result.
2. **Hard negatives cause collapse in the awareness probe, not the selection probe.** At N=50 hard-neg, 40% of tasks fail ≥1 of 3 trials; 20% never recover GT. The hypothesised failure mode ("model recognises GT but doesn't use it") is inverted.

v1 also had two methodology defects: `gt_display_ids` leaked into prompt JSON (contaminated selection numbers), and subagent dispatch returned only ~50% trustworthy responses. v2 fixes both and formalises the experiment for publication.

---

## 1. Goal & research questions

**Goal.** Quantify when a Claude agent can select the ground-truth skill from a pool containing semantically similar distractors, and when that ability collapses. Produce a paper-ready benchmark (metadata pushed to the project GitHub) plus an ablation of three variables: pool size, distractor strategy, skill representation.

**Rename.** "Selection Collapse" → **"Awareness Collapse"**, reflecting the actual failure mode.

**Research questions.**

- **RQ1.** Under hard-negative distractors, how does awareness accuracy scale with pool size `N ∈ {5, 50, 200, 1000, 2000, 5000}`?
- **RQ2.** Does skill representation (`card` vs. `full SKILL.md`) change the collapse threshold?
- **RQ3.** Does Phase A retrieval accuracy predict Phase B end-to-end task success on Terminal-Bench?
- **RQ4.** (stretch) Does extended-thinking alter the collapse curve?

---

## 2. Phase structure

### Phase A — Retrieval-only eval on SkillsBench (primary volume)

Single-turn prompt: agent receives `{task_description, skills_pool}`, emits a `<skill>…</skill>` or `<skills>…</skills>` tag, parser scores against GT. No agent execution, no harbor.

Two probes per `(task, pool, seed)`:

- **Awareness probe** — agent returns top-5 most relevant skills.
- **Selection probe** — agent returns the one skill it would use to solve the task.

GT: SkillsBench-native mapping, already materialised at `data/selection_collapse/skillsbench/tasks.jsonl` (single-GT per task; extend to multi-GT only if labelling ambiguity shows up).

### Phase B — End-to-end spot-check on Terminal-Bench (validation)

Terminal-Bench does not ship with GT skills. Phase B has two sub-stages:

- **B.1 GT labelling.** For each Terminal-Bench task, embed the description (Qwen-embedding, already used in `data/embeddings/`), retrieve top-20 candidate clawhub skills, run a baseline (no-skill) harbor eval, then re-run with each candidate skill injected. Candidates that strictly improve pass-rate (tiebreak: fewer steps) are recorded as GT. Tasks with no improving candidate are dropped. Uses `Procmem2Skills-ref/src/procmem2skills/integrations/harbor_terminal_experiment.py`.
- **B.2 Spot-check eval.** For a labelled subset, run full harbor tasks under controlled pools and correlate task success with Phase A's per-task awareness accuracy. Goal: validate Phase A as a cheap proxy for downstream success.

Phase B.1 blocks on harbor compute and Hanwen's coordination; Phase A is independent.

---

## 3. Experimental variables

| Variable | Levels | Notes |
|---|---|---|
| Pool size `N` | 5, 50, 200, 1000, 2000, 5000 | Capped by context window at `full` representation. |
| Distractor strategy | `random`, `hard_neg`, `easy_neg` | `hard_neg` = top-k by cosine of task-embedding to corpus; `easy_neg` = bottom-k; `random` = uniform. All three apply ε-dedup against GT. |
| Representation | `card`, `full` | `card` = name + one-line description from frontmatter. `full` = full `SKILL.md` body. `full` only feasible at small N. |
| Probe | `awareness`, `selection` | Separate prompts, separate calls. |
| Seeds | 3 per condition | Shuffle order + distractor sample. |
| Model | `claude-sonnet-4-6` (primary), `claude-haiku-4-5` (sweep), `claude-opus-4-7` (small replication) | |

**ε-dedup.** Drop distractors whose cosine similarity to any GT skill embedding exceeds ε. Start ε=0.85; tune on a sanity pass (20-sample human spot-check confirming filtered distractors are functionally distinct from GT).

---

## 4. Prompt & output protocol

**Shared skeleton.**

```
You are a retrieval subject in a controlled study.

Task:
<task_description>

Available skills (<N>):
<pool_block>

Respond with EXACTLY ONE of:
  <skill>SKILL_ID</skill>              # selection probe
  <skills>ID_1,ID_2,ID_3,ID_4,ID_5</skills>  # awareness probe, ranked

No other text.
```

**GT-leakage fixes.**

- `gt_display_ids` is **removed** from v2 pool JSON files.
- Each trial re-randomises skill IDs as `SKILL_000 … SKILL_{N-1}`. Ordinal position carries no signal; the SkillsBench canonical name (e.g., `mesh-analysis`) is preserved only inside the card body for the agent to reason over.
- Mapping from `SKILL_k` → canonical id is stored in `skills retrieval/runs/<run>/pools/<pool_id>.map.json` and loaded at scoring time, not prompt time.

**Parser.** Strict regex `^\s*<skill[s]?>([^<]+)</skill[s]?>\s*$`. Unparseable responses score 0 and are logged to `runs/<run>/parsed/unparseable.jsonl` for post-hoc inspection.

---

## 5. Metrics

Per `(task, pool, seed, probe)`:

- **Awareness Recall@5** — GT appears in returned top-5.
- **Selection Top-1** — returned skill == GT.
- **Selection | Aware** — selection accuracy conditional on awareness pass (diagnostic).
- **Parse-fail rate** — denominator hygiene.

Aggregated:

- **Per-task pass rate** (k/3 trials) → buckets `full-pass` (3/3), `partial-fail` (1–2/3), `full-fail` (0/3), as in v1 scale-up.
- **Bootstrap SE** across tasks × trials.
- **Collapse curve** — metric vs. `N`, faceted by distractor strategy and representation.

Paper figures:

1. Awareness Recall@5 vs. `N`, split by distractor strategy, with 95% bootstrap CI.
2. Representation ablation (`card` vs. `full`) at N=50 hard-neg.
3. Per-task collapse heatmap (tasks × N, cell = pass rate).
4. Phase A vs. Phase B correlation scatter (Terminal-Bench task success vs. Phase A awareness).

---

## 6. Infrastructure & budget

### 6.1 Runner

New driver at `skills retrieval/src/driver.py` (Arch-2: keep work local, reuse upstream data assets, do not touch `Procmem2Skills-ref` in this workstream):

- Direct `anthropic` SDK calls. No subagent dispatch, no harbor for Phase A.
- Async with bounded semaphore (8–16 concurrent requests).
- **Prompt caching** on the pool block — each pool is reused 6× (3 seeds × 2 probes); at N=5000 this saves ~200k cached tokens on 5 of 6 calls.
- Deterministic: `temperature=0`, seeded pool shuffle.

### 6.2 Data assets (reused as-is)

- `data/embeddings/skill_embeddings.npy` + `skill_metadata.jsonl` — 44,787 clawhub skills embedded.
- `data/selection_collapse/hard_negatives/sb_*_hard_negatives.json` — per-task hard-neg lists.
- `data/selection_collapse/skillsbench/tasks.jsonl` — task → GT mapping.
- `data/processed/skill_corpus.jsonl` — full corpus.

v2 pool files regenerate under `skills retrieval/pools/` (GT field stripped, IDs re-randomised).

### 6.3 Output layout

```
skills retrieval/
├── design-v2.md                          # this file
├── src/                                  # v2 driver code
│   ├── driver.py
│   ├── pool_builder.py
│   ├── prompt.py
│   ├── parser.py
│   └── metrics.py
├── pools/                                # v2 regenerated pools (no GT leak)
│   └── <task>_<strategy>_<N>_<seed>.json
├── runs/
│   ├── INDEX.jsonl                       # append-only run log
│   └── <YYYY-MM-DD-HHMM>-<label>/
│       ├── config.json
│       ├── raw/                          # api responses
│       ├── parsed/                       # parsed skill picks + unparseables
│       ├── metrics/                      # aggregated csv/json
│       └── figures/
├── benchmark/                            # published metadata (→ GitHub)
│   ├── skillsbench_gt.jsonl
│   └── terminal_bench_gt.jsonl          # from Phase B.1
└── analysis/                             # notebooks + write-up
```

### 6.4 Budget

After feasibility pruning (`full` representation infeasible at N>200; skip random × small-N cells that v1 already confirmed at ~100%): ~8k API calls. Cost dominated by N=2000 and N=5000 sweeps. Estimate ≤ \$400 on Sonnet 4.6 with caching; revisit after the M1 reproduction run.

---

## 7. Phase B — Terminal-Bench details

### 7.1 GT labelling pipeline

For each Terminal-Bench task `t`:

1. Embed `t.description` with Qwen-embedding.
2. `cands = top_k(corpus, t.description, k=20)`.
3. `baseline_result = harbor.run(t, skills=[])`.
4. For `s in cands`: `result = harbor.run(t, skills=[s])`; record pass, steps, tool calls.
5. `gt(t) = {s : result[s].pass ∧ ¬baseline_result.pass}` ∪ `{s : result[s].pass ∧ baseline_result.pass ∧ result[s].steps < baseline_result.steps}`.
6. If `gt(t) = ∅`, drop `t`.
7. Write `{task_id, gt_skills, candidate_evidence}` to `skills retrieval/benchmark/terminal_bench_gt.jsonl`.

"Improvement" definition default: pass-rate strict; tiebreak by step count. Record alternate definitions as ablation.

### 7.2 Spot-check eval

K ≈ 15 labelled tasks × 3 cells of interest (`hard_neg@50`, `hard_neg@1000`, `random@1000`) × 3 seeds. Run full harbor with the controlled pool, record task pass + which skill the agent invoked. Correlate against Phase A's per-task awareness accuracy.

### 7.3 Collaboration

Coordinate with Hanwen on harbor configuration and compute allocation. Open point: alternative "improvement" definitions (aggregate pass-rate over multiple seeds vs. single-run). Default to single-run pass + step tiebreak; document if revised.

---

## 8. Milestones

| # | Deliverable | Depends on | Est. |
|---|---|---|---|
| M1 | v2 driver + GT-stripped prompt; reproduce v1 pilot numbers on 5 tasks | — | 3 d |
| M2 | SkillsBench sweep `N ∈ {5,50,200,1000}` × `{random, hard_neg, easy_neg}` × `card` × 3 seeds | M1 | 2 d |
| M3 | Scale sweep `N ∈ {2000, 5000}`; representation ablation (`full` vs. `card`) at small N | M2 | 2 d |
| M4 | Phase A paper figures + metric tables | M3 | 1 d |
| M5 | Terminal-Bench GT labelling pipeline (coord w/ Hanwen) | M1 | 5 d |
| M6 | Phase B spot-check eval + correlation analysis | M4, M5 | 3 d |
| M7 | Benchmark metadata (`skillsbench_gt.jsonl`, `terminal_bench_gt.jsonl`) pushed to project GitHub | M5 | 0.5 d |

---

## 9. Risks

- **Context-window clip at N=5000.** Measure token usage of `card` representation at N=5000 before committing; fall back to N=3000 if needed.
- **ε-dedup tuning.** Too aggressive → removes legitimate distractors; too loose → distractors are near-duplicates of GT and "wrong" picks are defensible. Human spot-check on 20 samples is the gate.
- **Phase B compute.** GT labelling for Terminal-Bench requires up to 20 × task_count harbor runs. If compute is the bottleneck, restrict to a curated subset (e.g., 50 tasks) and note as scope limitation.
- **Multi-GT ambiguity surfacing during labelling.** Plan was single-GT v1; revisit if >15% of tasks have genuinely multiple valid skills.
- **Model drift.** Sonnet 4.6 may be deprecated mid-experiment. Lock model version in `runs/INDEX.jsonl` and re-run affected conditions if replaced.

---

## 10. Open questions

- Terminal-Bench "improvement" criterion — strict pass only, or pass + step count? (default: pass, tiebreak steps).
- Whether to include extended-thinking condition (RQ4 stretch) in v2 or defer.
- Whether to upstream the v2 driver into `Procmem2Skills-ref/src/procmem2skills/research/` after the paper is drafted.

---

## References

- v1 pilot: `data/noisy_retrieval/PILOT_ANALYSIS.md`
- v1 scale-up: `data/noisy_retrieval/SCALE_UP_ANALYSIS.md`
- Shared infra: `Procmem2Skills-ref/src/procmem2skills/{runtime,integrations,adapters}`
- Project blueprint: `Procmem2Skills-ref/docs/project-blueprint.md`
- SkillsBench: `/anvil/projects/x-cis260386/william/procmem2skills/skillsbench_repo/`
