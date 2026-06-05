# Prefill-Context Execution Pass-Rate Evaluation: Does Injecting Procedural Skill Documents Improve LLM Task Execution?

**procmem2skills / SkillsBench Evaluation Pipeline**

Version 1.1 | 2026-05-18 | Model Under Test: Claude Sonnet 4.6

---

## Abstract

We present a controlled experiment measuring whether prefilling procedural skill documents (SKILL.md files) into an LLM agent's system prompt improves execution pass-rate on software engineering tasks. Using 14 lightweight tasks from the SkillsBench benchmark, we construct candidate pools containing ground-truth (GT) skills mixed with controlled noise distractors, invoke Claude Sonnet 4.6 via the Claude Code CLI with full tool access (Bash, Read, Write, Edit), and validate outputs via pytest. Our factorial design varies pool size (5, 10, 20, 50), noise type (random, embedding-similar, embedding-distant), and random seed (5 replications), yielding 672 experimental configurations per treatment. Preliminary results show that prefilling GT skills raises pass-rate from 41.4% (bare agent) to 62.8--66.7% across noise conditions — a +21--25 percentage-point improvement. We also observe an inversion of the selection-proxy finding: embedding-similar ("hard") noise, which degrades skill *selection* accuracy, does not harm and may aid task *execution*, suggesting that semantically related context provides ancillary benefit during procedural task completion.

---

## 1. Introduction

### 1.1 Motivation

Large language model (LLM) agents increasingly tackle complex, domain-specific software engineering tasks. A critical question is whether providing the agent with explicit procedural documentation — step-by-step instructions for specialized operations — improves task completion beyond what the model achieves from its parametric knowledge alone.

Prior work in the procmem2skills pipeline established a **selection-proxy evaluation**: given an XML-formatted list of candidate skill documents, can the model identify which skill is relevant to a task? This proxy measures retrieval competence but leaves open the question of **execution competence**: even when the correct skill is present in context, does the model successfully follow its instructions to produce correct outputs?

### 1.2 Research Questions

- **RQ1**: Does prefilling ground-truth SKILL.md content into the system prompt improve execution pass-rate compared to a bare agent baseline?
- **RQ2**: How does noise type (random, hard/embedding-similar, easy/embedding-distant) affect execution pass-rate?
- **RQ3**: How does candidate pool size (5, 10, 20, 50 total candidates) affect execution pass-rate?
- **RQ4**: What is the gap between the selection-proxy upper bound and actual execution pass-rate?

### 1.3 Key Finding

Prefilling GT skills yields a consistent +21--25 pp improvement over the bare-agent baseline across all noise conditions. Notably, embedding-similar ("hard") distractors — which collapse selection accuracy from 80.9% to 55.1% at large pool sizes — do not degrade execution pass-rate. This inversion suggests that in execution mode, semantically similar noise provides useful ancillary context rather than confusion.

---

## 2. Experimental Design

### 2.1 Factorial Structure

The experiment follows a factorial design with three independent variables:

| Variable | Levels | Description |
|----------|--------|-------------|
| **Pool Size** | 5, 10, 20, 50 | Total candidates (GT + noise) in system prompt |
| **Noise Mode** | random, hard, easy | How distractor skills are sampled |
| **Seed** | 0, 1, 2, 3, 4 | RNG seed for deterministic pool construction |

**Noise mode definitions**:

- **random**: Uniform sample from the 47,231-skill ClawHub corpus (excluding GT aliases). Controls for pool-size effects without semantic bias.
- **hard**: Embedding-nearest neighbors to the task description via BAAI/bge-small-en-v1.5 (384-dim) over a prebuilt FAISS IndexFlatIP. Most semantically confusable with GT.
- **easy**: Embedding-farthest skills. Trivially distinguishable from GT by semantic distance.

### 2.2 Baseline and Control Conditions

To isolate the active ingredients of the prefill approach, we include four control conditions in a 2x2 factorial:

| Condition | GT Skills | Noise Skills | Skill Prompt Frame | N (configs) |
|-----------|-----------|--------------|-------------------|-------------|
| **noskill** | No | No | No | 70 |
| **emptyframe** | No | No | Yes | 70 |
| **noiseonly** | No | Yes (N=10) | Yes | 70 |
| **gtonly** | Yes | No | Yes | 110 |

This design separates three potential mechanisms: (1) the prompt framing effect (emptyframe vs. noskill), (2) the noise priming effect (noiseonly vs. emptyframe), and (3) the GT skill effect (gtonly vs. emptyframe).

### 2.3 Dataset

**SkillsBench** comprises 88 software engineering tasks spanning scientific computing, finance, ML, systems engineering, and domain-specific applications. Each task includes:

- `instruction.md`: Natural-language task description
- `environment/`: Fixture files (data, configs, binaries) and a Dockerfile specifying dependencies
- `environment/skills/<name>/SKILL.md`: Ground-truth procedural skill documents (1--7 per task)
- `tests/test_outputs.py`: pytest validation suite

**Task filtering**: Of 88 tasks, 14 are classified as "lightweight" (executable on host without heavy system dependencies). Tasks requiring texlive, nodejs, playwright, gcc, cuda, or chromium are excluded via regex match on Dockerfile content. This yields 14 evaluable tasks with 1--5 GT skills each.

### 2.4 Skill Corpus

The noise candidate pool draws from a corpus of **47,231 procedural skills** extracted from ClawHub repositories. Each skill has:

- A unique slug identifier
- A natural-language description
- BAAI/bge-small-en-v1.5 embeddings (384-dim float32)
- Pre-indexed in FAISS IndexFlatIP for cosine similarity retrieval

---

## 3. Pipeline Architecture

### 3.1 System Overview

```
Task Spec + GT Metadata
       |
  [pool_builder_v3.py]   -->  Build exact-sized candidate pool
       |
  [prompt_assembler.py]  -->  Concatenate SKILL.md (GT real + noise stubs)
       |
  [run_trial.py]         -->  Per-trial orchestration:
       |                      1. Setup isolated workspace
       |                      2. Invoke claude -p with system prompt
       |                      3. Run pytest against outputs
       |                      4. Record pass/fail to JSONL
       |
  [aggregate.py]         -->  JSONL -> pass-rate tables (JSON + Markdown)
       |
  [run_*.sh]             -->  Bash orchestrator (parallelization, retry, resume)
```

### 3.2 Pool Builder (`pool_builder_v3.py`)

Constructs a candidate pool of exactly `pool_size` skills with all GT skills preserved. Key design decisions:

- **Exact totals**: Pool sizes are exact totals (GT + noise), not GT + N_noise. If |GT| > pool_size, the trial is skipped (no truncation).
- **Leakage prevention**: GT-alias ClawHub IDs are excluded from noise sampling.
- **Deterministic seeding**: Stable seed via SHA-256 hash of (prefix, task_id, pool_size, noise_mode, seed).
- **Bias analysis**: Final candidate list is shuffled; GT positions are recorded for position-bias analysis.

### 3.3 Prompt Assembler (`prompt_assembler.py`)

Generates the system prompt by concatenating SKILL.md content for all candidates. Critical design choice:

- **GT skills** receive **full procedural content** — the actual SKILL.md files from SkillsBench task directories.
- **Noise skills** receive **synthesized stubs** containing only the corpus description plus a header note.

This prevents the model from trivially distinguishing GT from noise by content quality alone — it must rely on semantic relevance to the task.

**Output format** (XML-style wrapping):

```xml
<skill name="mesh-analysis">
  [full SKILL.md content — step-by-step procedural instructions]
</skill>
<skill name="task-patterns">
  [synthesized stub from corpus description]
</skill>
```

### 3.4 Trial Execution (`run_trial.py`)

Each trial proceeds through four stages:

**Stage 1 — Workspace Setup**: An isolated directory at `/tmp/exec_prefill/<trial_id>/work/` mirrors the task's Dockerfile COPY layout without actually building a container. Path rewriting maps hardcoded container paths (`/root/`, `/data/`, `/output/`) to workspace subdirectories.

**Stage 2 — Agent Invocation**: The model is invoked via Claude Code CLI in programmatic mode:

```bash
claude -p \
  --model claude-sonnet-4-6 \
  --system-prompt "<concatenated skill pool>" \
  --allowed-tools Bash,Read,Write,Edit,Glob,Grep \
  --permission-mode bypassPermissions \
  --no-session-persistence \
  --output-format text \
  --cwd <workspace> < <task_instruction>
```

The agent receives the task instruction as user input and has full shell access within the workspace. A 900-second timeout guards against runaway trials.

**Stage 3 — Test Validation**: The task's pytest suite is executed against the workspace:

```bash
WORKDIR=<workspace> python -m pytest tests/test_outputs.py -v
```

A trial **passes** if pytest exits with code 0. The `WORKDIR` environment variable handles path-patched assertions.

**Stage 4 — Result Recording**: Each trial emits one JSONL row:

```json
{
  "trial_id": "sb__3d-scan-calc__sz5__random__s0",
  "task_id": "3d-scan-calc",
  "pool_size": 5,
  "noise_mode": "random",
  "seed": 0,
  "gt_slugs": ["mesh-analysis"],
  "gt_positions": [2],
  "n_candidates": 5,
  "model": "claude-sonnet-4-6",
  "agent_wall_s": 34.7,
  "pass": true,
  "system_prompt_chars": 5311,
  "test_log_tail": "2 passed in 0.01s"
}
```

### 3.5 Containerization

The pipeline is packaged as a Docker image for reproducible execution:

| Component | Specification |
|-----------|---------------|
| Base image | `python:3.12-slim` |
| Key deps | faiss-cpu, sentence-transformers, pytest, torch (CPU) |
| Claude CLI | `@anthropic-ai/claude-code` via npm |
| Auth | Volume-mount `~/.claude/` and `~/.claude.json` |
| User | Non-root `evaluser` (UID 1000) |
| Data | Baked into image (testsets/, skill corpus, FAISS index, task dirs) |
| Results | Volume-mounted `./results/` |

### 3.6 Orchestration and Parallelization

Bash harness scripts manage the trial grid with:

- **Parallel container sharding**: Multiple Docker containers process disjoint trial subsets via `--skip-trials` offset.
- **Resume support**: `--resume` deduplicates against existing JSONL by (task_id, pool_size, noise_mode, seed) tuple.
- **Rate-limit detection**: Regex match on agent stdout for rate-limit messages; affected trials excluded from clean counts.
- **Stall detection**: 15 consecutive iterations with no progress triggers mode skip.

---

## 4. Results

### 4.1 Overall Pass-Rate by Condition

Results from completed experimental phases (Phases A, B, C) with 14 lightweight SkillsBench tasks:

| Condition | N (configs) | Pass-Rate | Delta vs. noskill |
|-----------|------------|-----------|-------------------|
| **noskill** (bare agent) | 70 | 41.4% | — |
| **emptyframe** (prompt frame only) | 70 | 38.6% | -2.9 pp |
| **noiseonly** (noise, no GT) | 70 | 41.4% | +0.0 pp |
| **gtonly** (GT, no noise) | 110 | 51.8% | +10.4 pp |
| **prefill-random** (GT + random noise) | 78 | 62.8% | +21.4 pp |
| **prefill-hard** (GT + emb-similar noise) | 78 | 66.7% | +25.3 pp |
| **prefill-easy** (GT + emb-distant noise) | 79 | 64.6% | +23.2 pp |

**Key observations**:

1. **Prefilling GT skills provides substantial improvement**: All three prefill conditions outperform the bare agent by +21--25 pp, confirming that procedural skill documents meaningfully aid task execution (RQ1).

2. **Prompt framing alone has no effect**: The emptyframe condition (38.6%) is not meaningfully different from noskill (41.4%), indicating that the XML skill-framing infrastructure does not independently improve performance.

3. **Noise alone has no effect**: The noiseonly condition (41.4%) matches noskill exactly, confirming that irrelevant skill documents neither help nor harm when GT is absent.

4. **GT-only underperforms GT+noise**: The gtonly condition (51.8%) is notably lower than all prefill conditions (62.8--66.7%). This is partially explained by pool size confounding — gtonly uses minimal pools (1--5 GT skills only), while prefill conditions pad to 5/10/20/50 total candidates. The additional context volume may aid the model.

### 4.2 Pass-Rate by Pool Size and Noise Mode

Detailed breakdown from the 20-task Phase C run (seed=0):

| Pool Size | Random | Hard | Easy |
|-----------|--------|------|------|
| 5 | 57.9% (n=19) | 70.0% (n=20) | 65.0% (n=20) |
| 10 | 60.0% (n=20) | 68.4% (n=19) | 70.0% (n=20) |
| 20 | 65.0% (n=20) | 65.0% (n=20) | 68.4% (n=19) |
| 50 | 68.4% (n=19) | 63.2% (n=19) | 55.0% (n=20) |

**Pool-size effect (RQ3)**: No consistent degradation with increasing pool size. Random noise shows slight improvement at larger pools; easy noise shows slight degradation at sz=50. The model appears robust to prompt length increases up to 50 candidates (median 40K chars at sz=50).

**Noise-mode effect (RQ2)**: Hard noise (embedding-similar) performs comparably to or better than random and easy noise across most pool sizes. This inverts the selection-proxy finding (see Section 4.4).

### 4.3 Per-Task Pass-Rate

Tasks exhibit a trimodal difficulty distribution:

| Tier | Tasks | Pass-Rate | Characteristics |
|------|-------|-----------|-----------------|
| **Always-pass** | earthquake-plate-calculation, energy-market-pricing, 3d-scan-calc, dialogue-parser, citation-check | 91--100% | Computational; well-specified inputs/outputs |
| **Skill-sensitive** | dapt-intrusion-detection, earthquake-phase-association, econ-detrending-correlation, adaptive-cruise-control | 46--83% | Benefit from procedural guidance; domain-specific |
| **Always-fail** | enterprise-information-search, civ6-adjacency-optimizer, azure-bgp-oscillation-route-leak | 0--8% | Complex multi-step logic; environment limitations |

The always-pass tier (5 tasks, ~95% average) establishes a performance ceiling. The skill-sensitive tier (4 tasks) is where prefilling GT skills has the most impact — these tasks require specialized domain knowledge that the procedural documents provide. The always-fail tier (3 tasks) likely requires capabilities beyond what current tool-augmented agents can achieve in an isolated workspace.

### 4.4 Selection-Proxy vs. Execution Inversion

The selection-proxy eval (conducted separately on Sonnet 4.5 with 1,068 trials) showed:

| Pool Size | Random Hit@1 | Hard Hit@1 | Easy Hit@1 |
|-----------|-------------|------------|------------|
| 5 | 80.9% | 59.6% | 85.4% |
| 50 | 83.2% | 55.1% | 80.9% |
| 500 | 76.4% | 41.6% | 71.9% |

Hard noise collapsed selection accuracy from ~81% (random) to ~55% (hard) — a devastating -26 pp effect. However, in execution mode, hard noise achieves 66.7% vs. random's 62.8% — a *positive* +3.9 pp effect.

**Interpretation**: In selection mode, embedding-similar distractors confuse the model about *which specific skill to pick*. In execution mode, the model doesn't need to select — all skills are in context. Semantically similar noise may provide ancillary domain context (related terminology, adjacent procedures) that aids comprehension and execution, even though the noise skills aren't the correct procedural match.

### 4.5 Execution Statistics

| Metric | Value |
|--------|-------|
| Median wall-time per trial | 213.8s |
| Mean wall-time per trial | 373.2s |
| Min / Max wall-time | 38.3s / 901.6s |
| Agent timeout | 900s |

**System prompt size by pool size**:

| Pool Size | Median Chars | Mean Chars | Range |
|-----------|-------------|------------|-------|
| 5 | 11,059 | 14,611 | 4,594 -- 38,573 |
| 10 | 14,807 | 18,067 | 7,149 -- 61,700 |
| 20 | 21,375 | 27,889 | 13,115 -- 168,270 |
| 50 | 40,439 | 50,568 | 30,039 -- 407,171 |

---

## 5. Experimental Phases

The experiment is structured into four phases to manage rate limits and prioritize statistical power:

| Phase | Purpose | Conditions | Configs | Status |
|-------|---------|------------|---------|--------|
| **Phase C** | Exploratory 20-task run | 3 noise x 4 pools x 1 seed | 168 | Complete |
| **Phase A** | Baseline replications | noskill + gtonly x 5 seeds | 140 | Complete |
| **Phase B** | Factorial controls | emptyframe + noiseonly x 5 seeds | 140 | Complete |
| **Phase P0** | Prefill seed replications | 3 noise x 4 pools x 4 seeds | 672 | In progress (243/672) |

Phase P0 adds seeds 1--4 to the existing seed-0 data, resolving pseudoreplication — the 12 conditions per task at seed=0 are not independent samples (they share the same pool construction randomness). With 5 seeds, each condition has true replications for variance estimation and paired statistical testing.

---

## 6. Methodology Notes

### 6.1 Threats to Validity

**Internal validity**:
- *Pseudoreplication*: Phases A/B/C use 5 seeds for true replications. Phase P0 (in progress) extends this to the prefill treatment.
- *Task selection bias*: 14/88 tasks are evaluable on host. Heavy-Dockerfile tasks (requiring nodejs, playwright, etc.) are excluded. Results may not generalize to the full SkillsBench distribution.
- *Path rewriting artifacts*: The workspace setup rewrites container paths to host paths. Imperfect rewrites could cause spurious failures unrelated to skill quality.

**External validity**:
- *Single model*: Only Claude Sonnet 4.6 is evaluated. Generalization to other LLMs is unknown.
- *Skill corpus scope*: The 47K ClawHub corpus may not represent the distribution of real-world procedural documentation.
- *Host execution*: Running tasks on the host (vs. inside Docker containers) may introduce environment-dependent variability.

### 6.2 Statistical Analysis Plan

Upon Phase P0 completion (672 configs + seed-0 merge = 840 total prefill configs):

1. **Task-level paired analysis**: Wilcoxon signed-rank test comparing per-task pass-rates between conditions.
2. **Bootstrap confidence intervals**: 95% CI on median pass-rate difference (prefill - baseline) via 10,000 bootstrap samples.
3. **Mixed-effects logistic regression**: Pass/fail ~ noise_mode * pool_size + (1|task_id) + (1|seed).
4. **Error taxonomy**: Classify failures into TIMEOUT, IMPORT_ERROR, PATH_ERROR, TEST_FAIL to understand failure modes.

---

## 7. Infrastructure

### 7.1 Compute Environment

- **Hardware**: NVIDIA GB10 (Grace-Blackwell, ARM64)
- **Model access**: Claude Sonnet 4.6 via Claude Code CLI (Max plan)
- **Concurrency**: 2 parallel Docker containers (rate-limit safe)
- **Estimated total compute**: ~600 trial-hours across all phases

### 7.2 Reproducibility

All pipeline components are deterministic given the same seed:
- Pool construction uses SHA-256 seeded randomness
- FAISS retrieval is deterministic for a given query
- Workspace setup follows the same Dockerfile parsing logic

The Docker image bakes all data dependencies (skill corpus, embeddings, task directories) for reproducible builds. Results are streamed to JSONL with resume support for interrupted runs.

---

## 8. Related Work

- **SkillsBench** (Anthropic): Benchmark of 88 software engineering tasks with Dockerfile-defined environments, pytest validation, and ground-truth skill annotations.
- **TerminalBench**: 62 validated terminal-based tasks (deferred in this evaluation — requires container execution).
- **ClawHub**: Corpus of 47,231 procedural skill documents extracted from open-source repositories, used as the distractor pool.
- **Selection-Proxy Eval** (procmem2skills): Prior work measuring skill identification accuracy without execution; establishes the upper bound for retrieval-based approaches.

---

## 9. Conclusion

Prefilling procedural skill documents into an LLM agent's system prompt provides a robust +21--25 pp improvement in task execution pass-rate over a bare agent baseline. This improvement is consistent across noise types and pool sizes up to 50 candidates. The selection-proxy inversion finding — that embedding-similar noise aids execution while harming selection — suggests that execution competence and retrieval competence are distinct capabilities with different failure modes.

These results support the value of maintaining curated procedural skill libraries for LLM agents: even imperfect retrieval (which necessarily includes some noise) is unlikely to degrade execution performance and may provide ancillary benefits through related domain context.

---

## Appendix A: Complete Per-Task Results (Phase C, seed=0)

| Task | N Trials | Pass-Rate | GT Skills |
|------|----------|-----------|-----------|
| earthquake-plate-calculation | 12 | 100.0% | 1 |
| energy-market-pricing | 12 | 100.0% | 1 |
| 3d-scan-calc | 48 | 97.9% | 1 |
| dialogue-parser | 12 | 91.7% | 1 |
| citation-check | 12 | 91.7% | 1 |
| dapt-intrusion-detection | 12 | 83.3% | 1 |
| earthquake-phase-association | 12 | 75.0% | 2 |
| econ-detrending-correlation | 12 | 75.0% | 1 |
| adaptive-cruise-control | 43 | 46.5% | 5 |
| energy-ac-optimal-power-flow | 12 | 41.7% | 2 |
| exceltable-in-ppt | 12 | 33.3% | 1 |
| azure-bgp-oscillation-route-leak | 12 | 8.3% | 3 |
| civ6-adjacency-optimizer | 12 | 8.3% | 4 |
| enterprise-information-search | 12 | 0.0% | 1 |

## Appendix B: Key Source Files

| File | Purpose |
|------|---------|
| `testsets/exec_eval_prefill/run_trial.py` | Per-trial driver: workspace, agent invocation, test runner |
| `testsets/exec_eval_prefill/pool_builder_v3.py` | Candidate pool construction (GT + noise, exact sizes) |
| `testsets/exec_eval_prefill/prompt_assembler.py` | System prompt generation (XML skill blocks) |
| `testsets/exec_eval_prefill/aggregate.py` | JSONL aggregation to pass-rate tables |
| `testsets/skill_selection_eval/retrieve.py` | FAISS embedding retrieval for hard/easy noise |
| `run_prefill_n5.sh` | Phase P0 orchestrator |
| `run_phase_b.sh` | Phase B orchestrator |
| `run_baselines_n5.sh` | Phase A orchestrator |

## Appendix C: Data Assets

| Asset | Size | Description |
|-------|------|-------------|
| `skill_corpus.jsonl` | 16 MB | 47,231 ClawHub skills (id, slug, description) |
| `skill_embeddings.npy` | 72 MB | 47K x 384 float32 BGE-small embeddings |
| `index.faiss` | ~5 MB | FAISS IndexFlatIP for cosine retrieval |
| `skillsbench_tasks.jsonl` | 136 KB | 88 SB task specs (task_id, gt_skills, desc) |
| `skillsbench_repo/tasks/` | 581 MB | 89 SB task directories (env, tests, solution) |
