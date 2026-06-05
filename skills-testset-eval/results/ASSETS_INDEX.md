# SkillsBench Eval Pipeline -- Assets Index

**Model**: Claude Sonnet 4.6 | **Date**: 2026-05-26 | **Total trials**: ~5,200+

---

## 1. Raw Execution Traces (JSONL)

Each row = one agent execution trial. Fields: `trial_id`, `task_id`, `pool_size`, `noise_mode`, `seed`, `gt_slugs`, `pass`, `agent_wall_s`, `agent_rc`, `agent_stdout_tail`, `agent_stderr_tail`, `test_log_tail`, `model`, `system_prompt_chars`.

### Phase A -- Baselines (Noskill + GT-only)

| File | Description | Rows | Tasks | Seeds | Conditions | Pass |
|------|-------------|------|-------|-------|------------|------|
| `results/sb_baselines.jsonl` | v1 pilot: noskill+gtonly | 10 | 5 | [0] | noskill, gtonly | 6/10 |
| `results/sb_baselines_v1.jsonl` | v1 copy (different run) | 10 | 5 | [0] | noskill, gtonly | 3/10 |
| `results/sb_baselines_20t.jsonl` | 20-task expansion, seed=0 | 40 | 20 | [0] | noskill, gtonly | 8/40 |
| `results/sb_baselines_n5.jsonl` | **Primary**: 20 tasks, N=5 seeds | 298 | 20 | [0-4] | noskill, gtonly | 86/298 (28.9%) |

### Phase B -- Factorial Controls

| File | Description | Rows | Tasks | Seeds | Conditions | Pass |
|------|-------------|------|-------|-------|------------|------|
| `results/sb_phase_b.jsonl` | Emptyframe + noiseonly | 200 | 20 | [0-4] | emptyframe, noiseonly | 56/200 (28.0%) |

### Phase C -- Prefill Execution (seed=0)

| File | Description | Rows | Tasks | Seeds | Pools | Pass |
|------|-------------|------|-------|-------|-------|------|
| `results/sb_exec.jsonl` | v1 pilot: 5 tasks | 60 | 5 | [0] | 5,10,20,50 | 34/60 |
| `results/sb_exec_v1.jsonl` | v1 copy | 60 | 5 | [0] | 5,10,20,50 | 26/60 |
| `results/sb_exec_20t.jsonl` | **Primary**: 20 tasks, seed=0 | 307 | 20 | [0] | 5,10,20,50 | 152/307 (49.5%) |

### Phase P0 -- Prefill Seed Replications (seeds 1-4)

| File | Description | Rows | Tasks | Seeds | Pools | Pass |
|------|-------------|------|-------|-------|-------|------|
| `results/sb_prefill_n5.jsonl` | **Primary**: 20 tasks, seeds 1-4 | 2333 | 20 | [1-4] | 5,10,20,50 | 404/2333 (17.3%) |

Note: Combine with `sb_exec_20t.jsonl` (seed=0) for full N=5 coverage of First20.

### Smoke10 -- 10 New Tasks

| File | Description | Rows | Tasks | Seeds | Pools | Pass |
|------|-------------|------|-------|-------|-------|------|
| `results/smoke10/smoke10_n5.jsonl` | **Primary** | 720 | 10 | [0-4] | 5,10,20,50,100 | 300/720 (41.7%) |
| `results/smoke10/canary.jsonl` | Single-task test | 1 | 1 | [0] | 5 | 1/1 |

Incomplete: jpg-ocr-stat (52/75), glm-lake-mendota (68/75).

### Smoke20 -- 20 New Tasks

| File | Description | Rows | Tasks | Seeds | Pools | Pass |
|------|-------------|------|-------|-------|-------|------|
| `results/smoke20/smoke20_n5.jsonl` | **Primary** | 1500 | 20 | [0-4] | 5,10,20,50,100 | 321/1500 (21.4%) |

30 design skips at pool=5 for python-scala-translation and travel-planning (|GT|=6 > pool_size).

---

## 2. Selection Eval Traces

Each row = one prompt-and-parse trial. Fields: `task_id`, `n_noise`, `pool_size`, `n_gt`, `seed`, `gt_names`, `hit1`, `recall`, `refusal`, `response`.

| File | Description | Rows | Tasks |
|------|-------------|------|-------|
| `testsets/eval_passrate_sonnet46/sb_passrate.jsonl` | **Primary**: 88 SB tasks, 6 noise levels | 528 | 88 |
| `testsets/eval_passrate_sonnet46/sb_passrate_smoke.jsonl` | Smoke: 5 SB tasks | 30 | 5 |
| `testsets/eval_passrate_sonnet46/tb_passrate.jsonl` | 62 Terminal-Bench validated tasks | 372 | 62 |
| `testsets/eval_passrate_sonnet46/tb_passrate_smoke.jsonl` | Smoke: 5 TB tasks | 30 | 5 |
| `testsets/eval_exec_prefill_sonnet46/sb_exec_smoke.jsonl` | Exec pilot: 4 tasks | 8 | 4 |

---

## 3. Run Logs

| File | Description | Size |
|------|-------------|------|
| `logs/smoke10.log` | Smoke10 orchestration output (10 tasks, 750 trials) | 267K |
| `logs/smoke20.log` | Smoke20 orchestration output (20 tasks, 1500 trials) | 636K |
| `results/baselines_20t.log` | Phase A 20-task baselines | 9K |
| `results/baselines_n5.log` | Phase A N=5 baselines | 43K |
| `results/parallel_20t.log` | Phase C parallel 20-task run | 83K |
| `results/rerun_20t.log` | Phase C re-run log | 28K |
| `testsets/eval_passrate_sonnet46/run.log` | Selection eval run | 76K |

---

## 4. Task Definitions

| File | Description | Tasks |
|------|-------------|-------|
| `testsets/data/skillsbench_tasks.jsonl` | Full SkillsBench task corpus | 88 |
| `testsets/data/terminal_bench_tasks.jsonl` | Full Terminal-Bench task corpus | 241 |
| `testsets/data/terminal_bench_validated.jsonl` | TB tasks with validated GT (Opus judge) | 62 |
| `testsets/data/smoke10.jsonl` | Smoke10 task definitions | 10 |
| `testsets/data/smoke20_next.jsonl` | Smoke20 task definitions | 20 |
| `testsets/data/smoke8_main.jsonl` | Smoke8 subset (predecessor to Smoke10) | 8 |
| `testsets/data/smoke2_zero.jsonl` | 2-task canary set | 2 |
| `testsets/data/pilot_tasks.jsonl` | Initial pilot task set | 20 |
| `testsets/data/canary_1.jsonl` | Single-task canary | 1 |

---

## 5. Embedding / Retrieval Data

| File | Description |
|------|-------------|
| `testsets/embeddings/skillsbench_gt_metadata.jsonl` | GT skill metadata for 88 SB tasks |
| `testsets/embeddings/task_description_embeddings_keys.jsonl` | Task description embedding keys |
| `data/embeddings/skill_metadata.jsonl` | Skill corpus metadata |
| `data/processed/skill_corpus.jsonl` | Processed skill corpus |

---

## 6. Reports (PDF)

### Primary Reports
| File | Description | Pages |
|------|-------------|-------|
| `results/final_analysis_report.pdf` | **Final**: Selection vs Exec, Pool/Noise grid, Case studies, API check | 6 |
| `results/grid_analysis_timeout_excluded.pdf` | Conditional non-timeout correctness, 4-metric diagnostic | 8 |
| `results/comparison_50task_report.pdf` | 50-task comparison across 3 batches, data integrity | 12 |
| `results/20t_report_v4.pdf` | **Canonical v4**: 14-task paired analysis, error taxonomy | 6 |
| `results/20t_report_v5.pdf` | v5: adds Phase B factorial controls | 6 |

### Earlier/Supplementary Reports
| File | Description |
|------|-------------|
| `results/20t_report.pdf` | First 20-task report |
| `results/20t_report_v3.pdf` | v3 iteration |
| `results/analysis_report.pdf` | Initial analysis |
| `results/analysis_report_v2.pdf` | Updated analysis |
| `results/final_report_v2.pdf` | Earlier "final" report |
| `results/api_cost_estimate.pdf` | Cost estimation |
| `results/prompt_vs_walltime.pdf` | Prompt length vs wall time analysis |
| `results/unified_unique_config_report.pdf` | Unique config analysis |
| `design_document.pdf` | Original experiment design |
| `results/design_document_v2.pdf` | Updated design |

---

## 7. Figures

| File | Description |
|------|-------------|
| `results/experiment_design_diagram.png` | Experiment pipeline diagram (for NotebookLM) |
| `results/prompt_vs_walltime.png` | Scatter: prompt chars vs wall time |
| `results/122.png` | Reference diagram (RQ1-RQ2 style) |
| `results/results.pptx` | PowerPoint results deck |

---

## 8. Analysis Scripts

### Report Generators
| File | Produces |
|------|----------|
| `generate_final_analysis.py` | `final_analysis_report.pdf` |
| `generate_grid_analysis.py` | `grid_analysis_timeout_excluded.pdf` |
| `generate_comparison_report.py` | `comparison_50task_report.pdf` |
| `generate_20t_report_v5.py` | `20t_report_v5.pdf` |
| `generate_experiment_diagram.py` | `experiment_design_diagram.png` |
| `generate_cost_estimate.py` | `api_cost_estimate.pdf` |

### Analysis Code
| File | Purpose |
|------|---------|
| `analysis_error_taxonomy.py` | Classify failures: TIMEOUT/TEST_FAIL/IMPORT_ERROR/PATH_ERROR/UNKNOWN |
| `analysis_paired.py` | Paired non-parametric analysis (bootstrap CI, Wilcoxon) |

---

## 9. Orchestration Scripts

| File | Phase | Description |
|------|-------|-------------|
| `run_baselines_n5.sh` | Phase A | Noskill + GT-only, 20 tasks, seeds 0-4 |
| `run_phase_b.sh` | Phase B | Emptyframe + noiseonly |
| `run_prefill_n5.sh` | Phase P0 | Prefill exec, seeds 1-4 |
| `run_baselines_20t.sh` | Phase A/C | 20-task baselines and prefill |
| `launch_smoke20_next.sh` | Smoke20 | 20 new tasks, seeds 0-4, pools 5-100 |

---

## 10. Documentation

| File | Description |
|------|-------------|
| `README.md` | Project overview and usage |
| `MANIFEST.md` | Original file manifest |
| `SKILLSBENCH_EVAL_PAPER.md` | Publication-ready paper draft |
| `REMAINING_6_TASKS_DESIGN.md` | Design for remaining 6 tasks |
| `UNTOUCHED_69_TASKS_DESIGN.md` | Design for untouched 69 tasks |
| `testsets/RESULTS.md` | Selection eval results log (historical) |
| `testsets/README.md` | Testsets documentation |

---

## Quick Reference: Which files answer which question?

| Question | Primary Data | Report |
|----------|-------------|--------|
| Does skill selection accuracy translate to execution? | `sb_passrate.jsonl` + `smoke{10,20}_n5.jsonl` | `final_analysis_report.pdf` Sec 1 |
| Does pool size / noise mode affect pass rate? | `sb_exec_20t.jsonl` + `sb_prefill_n5.jsonl` + smokes | `final_analysis_report.pdf` Sec 2, `grid_analysis_timeout_excluded.pdf` Sec 4 |
| Are 0% tasks caused by API exhaustion? | `smoke{10,20}_n5.jsonl` (stderr/stdout fields) | `final_analysis_report.pdf` Sec 4 |
| What is the overall pass rate across all tasks? | All `*_n5.jsonl` files | `comparison_50task_report.pdf` |
| Is the prefill effect statistically significant? | `sb_exec_20t.jsonl` + `sb_baselines_n5.jsonl` | `20t_report_v4.pdf` Sec 4 (p=0.14, non-significant) |
| What are the failure modes? | `sb_exec_20t.jsonl` + smokes | `20t_report_v4.pdf` Sec 7, `grid_analysis_timeout_excluded.pdf` Sec 3 |
