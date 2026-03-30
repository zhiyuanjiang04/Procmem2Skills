# Workflow-to-Skill Strategy Comparison Research Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare two atomic-skill induction strategies and decide a production direction for procmem2skills with evidence from real benchmark traces.

**Architecture:** Keep rollout and workflow induction unchanged; branch only at workflow->skill stage into Strategy A (step-level clustering) and Strategy B (workflow-cluster-first), then evaluate both with identical retrieval/injection settings.

**Tech Stack:** Python, Harbor, Qwen3-Embedding-0.6B, DBSCAN, codex-model API inference, terminal-bench/skillsbench traces.

---

## Scope and Question

We compare:
1. **Strategy A (Step-first):** split each workflow into steps -> cluster steps -> induce atomic skills from step clusters.
2. **Strategy B (Workflow-first):** cluster full workflows -> generate skills per workflow cluster -> dedup generated skills.

Primary decision question:
- Which strategy gives better trade-off among information retention, dedup quality, transfer performance, and operational stability?

## File Structure (Planned)

**Files to Create:**
- `docs/superpowers/specs/2026-03-28-workflow-to-skill-research-spec.md` (optional spec refinement if needed)
- `docs/superpowers/plans/2026-03-28-workflow-to-skill-research-plan.md` (this plan)
- `configs/research/workflow_to_skill_strategy_a.yaml`
- `configs/research/workflow_to_skill_strategy_b.yaml`
- `scripts/server/run_workflow_to_skill_research.py`
- `src/procmem2skills/research/workflow_to_skill_strategy.py`
- `src/procmem2skills/research/workflow_to_skill_metrics.py`
- `src/procmem2skills/research/workflow_to_skill_reporting.py`

**Files to Modify:**
- `src/procmem2skills/evaluation/pipeline.py` (add strategy switchpoints and shared interfaces)
- `src/procmem2skills/packager/materialize.py` (strict codex-model generation constraints)
- `src/procmem2skills/packager/llm_skill_creator.py` (skill generation prompt/format hard constraints)
- `src/procmem2skills/packager/skill_writer.py` (minimal reusable artifact format)

## Feasibility Analysis

### Strategy A: Step-first clustering

**What it is:**
- Flatten all workflows into step objects.
- Cluster semantically similar steps globally (or per-task in ablation).
- Generate atomic skills from each step cluster.

**Pros:**
- High atomicity; naturally aligned with reusable micro-skills.
- Better cross-task reuse when different tasks share identical operations.
- Fine-grained control for progressive disclosure.

**Risks:**
- Loses inter-step dependency unless explicitly preserved.
- More sensitive to parser variance and step boundary quality.
- Can create too many tiny clusters (fragmentation) and increase retrieval noise.

**Best fit benchmarks:**
- SkillsBench style tasks with repetitive tool primitives and short workflows.

### Strategy B: Workflow-first clustering

**What it is:**
- Cluster full workflows first (global DBSCAN + embedding).
- Generate skills per cluster from grouped trajectory evidence.
- Dedup skills after generation.

**Pros:**
- Better preserves procedural context and ordering.
- More robust when step granularity differs across trajectories.
- Usually lower cluster count and simpler management.

**Risks:**
- Generated skills may be less atomic and more task-specific.
- Dedup after generation may be harder if writing style differs.

**Best fit benchmarks:**
- Terminal-Bench where long-horizon procedure coherence matters.

### Practical conclusion before execution

- Both are feasible.
- **Recommended first execution order:** Strategy B -> Strategy A.
  - Reason: B has lower engineering and evaluation variance and gives a stable baseline for later A comparison.

## Unified Evaluation Protocol

Keep identical across A/B:
- Same trajectory pool and trial seeds.
- Same embedding backend/model for clustering (`Qwen/Qwen3-Embedding-0.6B`).
- Same retrieval modes and pool sweeps (`50/500/5000`).
- Same injection policy and timeout settings.
- Same agent/model for downstream evaluation (codex model via API key).

## Metrics

### Intrinsic metrics
- `compression_ratio`: raw workflows -> generated skills.
- `dedup_ratio`: pre/post dedup skill count.
- `coverage_step`: fraction of actionable steps covered by generated skills.
- `coverage_workflow`: fraction of workflows with at least one mapped skill.
- `redundancy_index`: mean nearest-neighbor similarity among skills.

### Extrinsic metrics
- Task success rate delta vs no-skills baseline.
- Timeout/error rate delta.
- Avg steps to success.
- Failure attribution shift:
  - unable-retrieve
  - pick-wrong
  - pick-related-but-fail
  - agent-misuse
  - skill-error
  - misled-by-noise

## Experimental Matrix

- Strategy: `A(step-first)` vs `B(workflow-first)`.
- Skill synthesis unit: `per-task` vs `per-cluster`.
- Memory source: `success-only` vs `all` (ablation).
- Retrieval: `page-index / context-injection / embedding-based`.
- Pool size: `50/500/5000`.
- Split: `in-task` vs `cross-task-holdout`.

## Execution Plan (No immediate run)

### Task 1: Define strategy interfaces and configs

**Files:**
- Create: `configs/research/workflow_to_skill_strategy_a.yaml`
- Create: `configs/research/workflow_to_skill_strategy_b.yaml`
- Create: `src/procmem2skills/research/workflow_to_skill_strategy.py`

- [ ] Step 1: Define shared schema for strategy input/output.
- [ ] Step 2: Encode Strategy A config.
- [ ] Step 3: Encode Strategy B config.
- [ ] Step 4: Add validator for reproducibility fields (seed/model/pool settings).

### Task 2: Implement strategy executors

**Files:**
- Modify: `src/procmem2skills/evaluation/pipeline.py`
- Create: `src/procmem2skills/research/workflow_to_skill_strategy.py`

- [ ] Step 1: Implement A executor (step flatten -> step cluster -> skill induction).
- [ ] Step 2: Implement B executor (workflow cluster -> cluster skill generation -> dedup).
- [ ] Step 3: Add per-task/per-cluster synthesis switch at generation stage.
- [ ] Step 4: Ensure full provenance linkage (skill -> source workflows/steps/episodes).

### Task 3: Strict codex-model generation guardrails

**Files:**
- Modify: `src/procmem2skills/packager/llm_skill_creator.py`
- Modify: `src/procmem2skills/packager/materialize.py`
- Modify: `src/procmem2skills/packager/skill_writer.py`

- [ ] Step 1: Enforce codex model family via API-key flow (no server-native codex binary dependency).
- [ ] Step 2: Add strict system prompt constraints (format, completeness, no evidence omission).
- [ ] Step 3: Output minimal reusable artifact format (`SKILL.md` + compact metadata).
- [ ] Step 4: Add strict failure behavior when generation format/coverage is invalid.

### Task 4: Metrics and reporting

**Files:**
- Create: `src/procmem2skills/research/workflow_to_skill_metrics.py`
- Create: `src/procmem2skills/research/workflow_to_skill_reporting.py`
- Create: `scripts/server/run_workflow_to_skill_research.py`

- [ ] Step 1: Implement intrinsic metrics.
- [ ] Step 2: Implement extrinsic metrics adapters.
- [ ] Step 3: Emit comparable JSON reports and markdown summary.
- [ ] Step 4: Add benchmark->recommended-analysis summary from observed traces.

## Risks and Controls

- **Risk:** Step-first fragmentation inflates skill count.
  - Control: min-cluster-size + post-merge threshold + redundancy cap.
- **Risk:** Workflow-first overfits and loses atomicity.
  - Control: enforce max steps per skill and split oversized skills.
- **Risk:** LLM variance in skill writing.
  - Control: deterministic prompts + strict schema + retry with validator.
- **Risk:** Retrieval noise confounds comparison.
  - Control: fixed retrieval budget and same noise sweep across A/B.

## Expected Deliverables

- One comparable report with A/B matrix and recommendation.
- Reproducible configs and launcher script.
- Final recommendation table by benchmark:
  - `benchmark -> preferred strategy -> preferred synthesis unit -> expected risk`.

## Recommendation Before Running

- Start with `Strategy B + per-cluster` as baseline.
- Then run `Strategy A + per-cluster` and `Strategy A + per-task`.
- Keep `Strategy B + per-task` as sanity ablation.
- Make final decision from cross-task-holdout first, then in-task.
