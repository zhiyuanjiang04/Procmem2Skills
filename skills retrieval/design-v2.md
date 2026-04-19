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

**Goal.** Characterise when a Claude agent can **recognise** the correct skill amid semantically similar distractors (Phase A), and whether recognition ability predicts **downstream usage** under true retrieval conditions (Phase B). Produce a paper-ready benchmark (metadata on GitHub) and ablations across distractor type, pool size, representation, and content cues (name vs. description).

**Framing (per review).** Phase A is **recognition under distractor interference**, not open-set retrieval — the pool is given explicitly and the agent must identify the correct member. Phase B is **true retrieval + usage** in a real agent loop. The v1 failure mode we observed is a *recognition* failure (GT not surfaced under semantic interference), which connects to interference-resolution findings in cognitive-style LLM evaluation literature.

**Rename.** "Selection Collapse" → **"Recognition-under-Interference Collapse"** (short form: "Awareness Collapse" retained as informal label).

**Research questions.**

- **RQ1.** Under increasing *semantic* and *functional* distractor interference, at what pool size does correct skill recognition fail, and how does the degradation curve look (gradual vs. cliff)?
- **RQ2.** Does skill representation (`card` vs. `full SKILL.md`) and its content cues (name vs. description) change the collapse threshold? How much of recognition is lexical vs. semantic?
- **RQ3.** Does Phase A recognition accuracy (Recall@5, MRR, Top-1 Awareness) predict Phase B end-to-end task success on Terminal-Bench?
- **RQ4.** When the agent recognises the correct skill, does it reliably *use* it (Selection | Aware)? Do recognition and usage diverge as pool interference grows?
- **RQ5.** (stretch) Does extended-thinking alter the collapse curve?

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
| Distractor strategy | `random`, `easy_neg`, `hard_neg_semantic`, `hard_neg_functional`, `adversarial` | Four-tier distractor taxonomy — see §3.1. |
| Representation | `card`, `full`, `name_only`, `desc_only` | `card` = name + one-line description; `full` = full `SKILL.md`; `name_only` / `desc_only` are ablation cells — see §3.2. |
| Format perturbation | `canonical`, `shuffled` | 80% canonical, 20% shuffled (field order + phrasing paraphrase) — see §3.3. |
| Content budget (at fixed N) | `card_budget`, `full_budget_truncated` | Confound control: match token budget when comparing `card@N` vs. `full@N`. See §3.4. |
| Probe | `awareness`, `selection` | Separate prompts, separate calls. |
| Seeds | 3 per condition | Shuffle order + distractor sample. |
| Model | `claude-sonnet-4-6` (primary), `claude-haiku-4-5` (sweep), `claude-opus-4-7` (small replication) | |

### 3.1 Distractor taxonomy

- **`random`** — uniform from 44k corpus, ε-dedup against GT.
- **`easy_neg`** — bottom-k by cosine of task-embedding to skill embedding (unrelated skills).
- **`hard_neg_semantic`** — top-k by cosine similarity; linguistically/topically close to the task description. (v1's only hard-neg strategy.)
- **`hard_neg_functional`** — sampled from the same skill cluster as GT (HDBSCAN or k-means on the 44k embedding space; see §3.5), but excluding skills in the GT's immediate functional subtree. These solve *similar task types* but differ in key constraints, tools, or outputs — closer to "confusable competitor" than "lexically similar noise". Expected to be stronger than semantic hard-neg.
- **`adversarial`** — LLM-generated skill descriptions produced by prompting Claude Opus 4.7 with: *"Given this task, generate a skill description that looks applicable but is subtly incorrect (wrong tool / wrong output / wrong constraint)."* Applied only at N=50 as a mini stress-test; quality-filtered by a second model pass.

### 3.2 Representation ablation

At N=50 × `hard_neg_semantic`, compare four representations to disentangle lexical vs. semantic recognition:

- `name_only` — skill name string only.
- `desc_only` — one-line description only, name stripped.
- `card` — name + description (v2 default).
- `full` — full `SKILL.md` body (feasibility-permitting).

Expected insight: quantifies how much of recognition is lexical name-matching vs. reading the description.

### 3.3 Format perturbation

To rule out "style leakage" (models picking up on consistent phrasing/naming patterns), 20% of runs per condition apply:

- Random field-order shuffle within each card.
- Light paraphrase of the description (single Claude pass, preserving semantics).
- Bullet ↔ sentence reformatting.

Report accuracy delta between canonical and shuffled; large deltas flag stylistic leakage.

### 3.4 Representation × N confound control

Raw `card@N=2000` vs. `full@N=200` confounds interference (more distractors) with information loss (less detail per distractor). To isolate:

- At N=200, run `full` at its natural token budget AND a truncated `full` matched to `card`'s per-skill token budget.
- Alternately: run `card@N=2000` alongside a `compressed_full@N=2000` (full-body summarised to card-length).
- Target: show whether accuracy drops are driven by interference or by information loss. Reviewer-proofing.

### 3.5 ε-dedup and clustering

- **ε-dedup.** Drop distractors whose cosine similarity to any GT skill embedding exceeds ε. Start ε=0.85; tune on a sanity pass (20-sample human spot-check confirming filtered distractors are functionally distinct from GT).
- **Clustering for `hard_neg_functional`.** Run HDBSCAN on the 44k skill embeddings (metric=cosine). Each GT is mapped to its cluster; functional hard-negs sample from the same cluster minus a small "functional subtree" around GT (nearest neighbours within ε2=0.75). Record cluster id + cluster density with each pool for per-task difficulty analysis (§5).

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
- Each trial re-randomises skill IDs as `SKILL_000 … SKILL_{N-1}`. Ordinal position carries no signal; the SkillsBench canonical name (e.g., `mesh-analysis`) is preserved only inside the card body for the agent to reason over (except in `desc_only` where names are stripped).
- Mapping from `SKILL_k` → canonical id is stored in `skills retrieval/runs/<run>/pools/<pool_id>.map.json` and loaded at scoring time, not prompt time.

**Pool-block rendering** depends on representation (§3.2):

- `card` — `SKILL_k: <name> — <one-line description>`
- `name_only` — `SKILL_k: <name>`
- `desc_only` — `SKILL_k: <one-line description>` (name stripped; parser still maps `SKILL_k` → canonical id server-side)
- `full` — `SKILL_k:\n<full SKILL.md body>` separated by `---`
- `full_truncated` — `full` body truncated to token budget matching `card` (for §3.4 confound control)
- `compressed_full` — one-pass Claude summary of full body, target length = card; cached

**Format perturbation** (§3.3): when a trial is flagged `shuffled`, for each skill card independently:

- Randomly permute the order of (name, description) — e.g., "desc — name" vs. "name — description".
- Paraphrase the description with Claude Haiku (temp=0.3) preserving semantics.
- Optionally reformat bullet points to prose or vice versa.
- Record the perturbation seed in the pool map for reproducibility.

**Parser.** Strict regex `^\s*<skill[s]?>([^<]+)</skill[s]?>\s*$`. Unparseable responses score 0 and are logged to `runs/<run>/parsed/unparseable.jsonl` for post-hoc inspection.

---

## 5. Metrics

Per `(task, pool, seed, probe)`:

- **Awareness Top-1** — GT at rank 1. (New, primary; catches cases Recall@5 misses.)
- **Awareness MRR** — Mean Reciprocal Rank of GT in the returned ranked list. (New, primary.)
- **Awareness Recall@5** — GT appears in returned top-5. (Retained for comparability with v1; known saturated.)
- **Rank distribution** — histogram of GT rank across trials; reveals whether GT is barely-in-top-5 vs. solidly ranked.
- **Selection Top-1** — returned skill == GT (selection probe).
- **Selection | Aware** — selection accuracy conditional on awareness pass. **Primary result**, not diagnostic: a divergence between unconditional Selection and Selection|Aware indicates a real pipeline breakdown (agent saw the right skill but didn't use it).
- **Parse-fail rate** — denominator hygiene.
- **Format-robustness delta** — Δaccuracy between canonical and shuffled pools (§3.3).

Aggregated:

- **Per-task pass rate** (k/3 trials) → buckets `full-pass` (3/3), `partial-fail` (1–2/3), `full-fail` (0/3), as in v1 scale-up.
- **Bootstrap SE** across tasks × trials.
- **Collapse curve** — metric vs. `N`, faceted by distractor strategy and representation.

### 5.1 Per-task difficulty features

To move from *descriptive* (collapse curves) to *predictive* (when does collapse happen?), compute per-task features and correlate with failure rate:

- **Top-k similarity entropy** — entropy of the top-k cosine-similarity distribution between task embedding and pool. High entropy → ambiguous.
- **GT–2nd-best gap** — cosine gap between GT and the next-closest skill in the pool. Small gap → harder task.
- **Cluster density** — size + intra-cluster similarity of GT's HDBSCAN cluster.
- **GT name-lexical-overlap with task description** — proxy for "how keyword-matchable is this task".

Regress task-level failure rate on these features; report feature importances.

### 5.2 Paper figures

1. **Main collapse curve** — Awareness MRR + Top-1 vs. `N`, split by distractor strategy (including `hard_neg_functional` and `adversarial`), with 95% bootstrap CI. Recall@5 shown as faded reference line.
2. **Selection vs. Selection | Aware divergence** — both curves on one axis, vs. `N`, at `hard_neg_semantic`. Divergence marks the awareness–usage breakdown.
3. **Representation ablation** — accuracy at N=50 hard-neg for `name_only` / `desc_only` / `card` / `full`. Quantifies lexical-vs-semantic contribution.
4. **Representation × N confound control** — `card@N` vs. token-matched `full@N` and `compressed_full@N`. Isolates interference from information loss.
5. **Per-task collapse heatmap** — tasks × N, cell = pass rate; tasks sorted by GT–2nd-best gap.
6. **Difficulty regression** — feature importance bar chart from §5.1.
7. **Phase A → Phase B correlation** — Terminal-Bench task success vs. Phase A Awareness Top-1, per (N, strategy).

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
3. `baseline_result = harbor.run(t, skills=[])` — run 3 seeds to stabilise.
4. For `s in cands`: `result = harbor.run(t, skills=[s])`; record pass, steps, tool calls.
5. Assign each candidate one of three **graded GT labels**:
   - **`strong_gt`** — candidate passes consistently (≥2/3 seeds) when baseline fails (≤1/3).
   - **`weak_gt`** — candidate improves on baseline marginally (e.g., passes on 2/3 seeds when baseline passes 1/3, or reduces steps ≥30% on a passing task).
   - **`equivalent_gt`** — candidate passes at the same rate as baseline but uses a materially different strategy (different tools/commands). Identified by a cheap LLM judge comparing transcripts.
6. If no candidate has `strong_gt` or `weak_gt`, drop `t`.
7. Write `{task_id, candidates: [{skill_id, label, evidence}], dropped?}` to `skills retrieval/benchmark/terminal_bench_gt.jsonl`.

**Bias acknowledgements and mitigations** (per review):

- Single-skill injection only — cannot detect compositional GTs (skill-A ∘ skill-B). Noted as limitation; a follow-up experiment could sweep pairs at bounded cost.
- Labels favour skills that exploit benchmark weaknesses. Log `strong_gt` vs. `weak_gt` separately so Phase B analyses can condition on label strength.
- Record full candidate evidence (not just the chosen GT) so the graded labels can be re-aggregated post-hoc.

"Improvement" definition default: strict pass-rate with 3-seed aggregation; tiebreak by step count. Alternate definitions (pass-rate only; steps-only on passing tasks) recorded as ablation.

### 7.2 Spot-check eval

K ≈ 15 labelled tasks × 3 cells of interest (`hard_neg@50`, `hard_neg@1000`, `random@1000`) × 3 seeds. Run full harbor with the controlled pool, record task pass + which skill the agent invoked. Correlate against Phase A's per-task awareness accuracy.

### 7.3 Collaboration

Coordinate with Hanwen on harbor configuration and compute allocation. Open point: alternative "improvement" definitions (aggregate pass-rate over multiple seeds vs. single-run). Default to single-run pass + step tiebreak; document if revised.

---

## 8. Milestones

| # | Deliverable | Depends on | Est. |
|---|---|---|---|
| M1 | v2 driver + GT-stripped prompt + ID re-randomisation + format-shuffle; reproduce v1 pilot numbers on 5 tasks; add MRR/Top-1 metrics | — | 3 d |
| M1.5 | HDBSCAN clustering of 44k corpus; `hard_neg_functional` pool builder; ε-dedup sanity spot-check | M1 | 1.5 d |
| M2 | SkillsBench sweep `N ∈ {5,50,200,1000}` × `{random, easy_neg, hard_neg_semantic, hard_neg_functional}` × `card` × 3 seeds | M1.5 | 2 d |
| M2.5 | Selection probe (with fixed GT-leak) across same grid; compute Selection \| Aware curves | M2 | 1 d |
| M3 | Scale sweep `N ∈ {2000, 5000}`; representation ablation (`name_only`/`desc_only`/`card`/`full`) at N=50 hard-neg; confound control (token-matched `full`) at N=200 | M2 | 2 d |
| M3.5 | Adversarial-distractor mini-experiment at N=50 (LLM-generated distractors) | M1.5 | 1 d |
| M4 | Phase A paper figures + metric tables + per-task difficulty regression | M3, M3.5 | 1.5 d |
| M5 | Terminal-Bench GT labelling pipeline with graded labels (coord w/ Hanwen) | M1 | 5 d |
| M6 | Phase B spot-check eval + Phase A→B correlation analysis | M4, M5 | 3 d |
| M7 | Benchmark metadata (`skillsbench_gt.jsonl`, `terminal_bench_gt.jsonl` with graded labels) pushed to project GitHub | M5 | 0.5 d |

---

## 9. Risks

- **Context-window clip at N=5000.** Measure token usage of `card` representation at N=5000 before committing; fall back to N=3000 if needed.
- **ε-dedup tuning.** Too aggressive → removes legitimate distractors; too loose → distractors are near-duplicates of GT and "wrong" picks are defensible. Human spot-check on 20 samples is the gate.
- **HDBSCAN clustering instability.** Functional hard-negs depend on cluster quality. Sanity-check: inspect 5 clusters manually; if clusters are incoherent, fall back to k-means or GMM and note.
- **Adversarial distractor quality.** LLM-generated distractors may be obviously wrong (insufficiently adversarial) or accidentally correct. Second-pass LLM judge filters; keep a human-annotated subset to calibrate.
- **Phase B compute.** GT labelling for Terminal-Bench requires up to 20 × 3 seeds × task_count harbor runs. If compute is the bottleneck, restrict to a curated subset (e.g., 50 tasks) and note as scope limitation.
- **Graded-GT label noise.** `equivalent_gt` depends on an LLM judge of strategy difference — brittle. Report inter-judge agreement if feasible; otherwise demote to post-hoc analysis only.
- **Multi-GT ambiguity surfacing during labelling.** Plan was single-GT v1; graded labels (§7.1) partially absorb this. Revisit if >15% of tasks have multiple `strong_gt` candidates.
- **Model drift.** Sonnet 4.6 may be deprecated mid-experiment. Lock model version in `runs/INDEX.jsonl` and re-run affected conditions if replaced.

---

## 10. Open questions

- Terminal-Bench "improvement" criterion — strict pass only, or pass + step count? (default: pass, tiebreak steps; graded labels mitigate).
- Whether to include extended-thinking condition (RQ5 stretch) in v2 or defer.
- Whether to upstream the v2 driver into `Procmem2Skills-ref/src/procmem2skills/research/` after the paper is drafted.
- Whether to run a compositional (skill-pair) GT labelling follow-up for Terminal-Bench.
- Whether to expand format-perturbation beyond 20% if stylistic leakage deltas are non-trivial.

---

## References

- v1 pilot: `data/noisy_retrieval/PILOT_ANALYSIS.md`
- v1 scale-up: `data/noisy_retrieval/SCALE_UP_ANALYSIS.md`
- Shared infra: `Procmem2Skills-ref/src/procmem2skills/{runtime,integrations,adapters}`
- Project blueprint: `Procmem2Skills-ref/docs/project-blueprint.md`
- SkillsBench: `/anvil/projects/x-cis260386/william/procmem2skills/skillsbench_repo/`
