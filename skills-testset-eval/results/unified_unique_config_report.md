# Unified Unique-Config Evaluation Report

Generated: 2026-05-25 08:02 UTC

## Executive Summary

The apparent conflict between `20t_report_v4.pdf` and `comparison_50task_report.pdf` is a denominator issue, not a model-result conflict.

- 20t v4 prefill seed0, using unique configs and excluding skipped configs: 99/168 = 58.9%.
- First20 P0 from the 50-task report, using unique configs and excluding skipped configs: 404/672 = 60.1%.
- Combined First20 prefill seeds0-4: 503/840 = 59.9%.
- First20 GT-only baseline: 37/70 = 52.9%; comparable First20 combined GT+pool rates are listed below.
- 50-report source only under this same rule: 1025/2862 = 35.8% over 44 task IDs with at least one executed config.

Interpretation: the First20 prefill result is stable across v4 seed0 and later P0 seeds1-4. Raw-row reporting in the 50-task report should not be used as a performance denominator for First20 because repeated config rows heavily overweight a few tasks.

## Statistical Unit

- Primary unit: one unique config, encoded by `trial_id`.
- `skipped=true` configs are excluded from execution pass-rate denominators.
- Raw rows are retained only for provenance, duplicate, and coverage audits.
- Rate-limited rows are not removed from the primary metric here, matching the requested 404/672 denominator.
- `GT+5/10/20/50/100` means GT skills are included in a total candidate pool of 5/10/20/50/100 skills; it does not mean GT plus that many additional distractors.
- Timeout-excluded rates remove failed executions classified as timeout failures (`agent_wall_s > 600`, `agent_rc=124`, or timeout/killed text).

## Artifact Inventory

| Artifact | Bytes | Rows | PDF Pages | SHA256-16 | Role |
| --- | --- | --- | --- | --- | --- |
| results/20t_report_v4.pdf | 13074 | - | 6 | be9205a392290ad9 | Source report: seed-0 20-task paired analysis; mtime 2026-05-16 08:51 UTC |
| results/comparison_50task_report.pdf | 27596 | - | 12 | 79ede6f53ffa0180 | Source report: 50-task comparison; mtime 2026-05-25 06:49 UTC |
| generate_20t_report.py | 23381 | - | - | b3f8e5b6ec50e4db | Generator for 20t_report_v4.pdf; mtime 2026-05-17 08:03 UTC |
| generate_comparison_report.py | 26396 | - | - | 0de0351022f22d95 | Generator for comparison_50task_report.pdf; mtime 2026-05-25 06:49 UTC |
| results/sb_exec_20t.jsonl | 701075 | 307 | - | 23a5dbacf3be93be | 20t v4 prefill, seed 0; mtime 2026-05-16 02:05 UTC |
| results/sb_baselines_n5.jsonl | 727461 | 298 | - | 6b24edbb6b280be8 | 20t v4 noskill and GT-only baselines, seeds 0-4; mtime 2026-05-16 07:00 UTC |
| results/sb_prefill_n5.jsonl | 5698722 | 2333 | - | 55d929f4ca87cafd | First20 Phase P0 prefill, seeds 1-4 plus reruns; mtime 2026-05-21 11:37 UTC |
| results/smoke10/smoke10_n5.jsonl | 2125460 | 720 | - | 3df086ecb92b2389 | Smoke10 prefill, seeds 0-4; mtime 2026-05-24 03:34 UTC |
| results/smoke20/smoke20_n5.jsonl | 4488415 | 1500 | - | 46b6a8f888d4dcf2 | Smoke20 prefill, seeds 0-4; mtime 2026-05-24 21:10 UTC |
| logs/smoke10.log | 273040 | - | - | 67d67127091a9f69 | Run log supporting Smoke10 resume/incomplete diagnosis; mtime 2026-05-24 03:34 UTC |
| logs/smoke20.log | 651589 | - | - | aa48f9d0608dd0dd | Run log supporting Smoke20 run provenance; mtime 2026-05-24 21:10 UTC |
| testsets/data/smoke10.jsonl | 13845 | 10 | - | 0856a12ae2ebaf2a | Smoke10 task list; mtime 2026-05-23 08:34 UTC |
| testsets/data/smoke20_next.jsonl | 27468 | 20 | - | 42d69e5e8228fd82 | Smoke20 task list; mtime 2026-05-24 01:53 UTC |

## Unified Batch Summary

| Dataset slice | Expected configs | Raw rows | Unique configs | Missing | Skipped | Executed | Pass/Executed | Rate | Exec/All tasks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20t v4 prefill seed0 | 240 | 307 | 240 | 0 | 72 | 168 | 99/168 | 58.9% | 14/20 |
| First20 P0 seeds1-4 | 960 | 2333 | 916 | 44 | 244 | 672 | 404/672 | 60.1% | 14/20 |
| First20 combined seeds0-4 | 1200 | - | 1156 | 44 | 316 | 840 | 503/840 | 59.9% | 14/20 |
| Smoke10 | 750 | 720 | 720 | 30 | 0 | 720 | 300/720 | 41.7% | 10/10 |
| Smoke20 | 1500 | 1500 | 1500 | 0 | 30 | 1470 | 321/1470 | 21.8% | 20/20 |
| 50-report source only | 3210 | - | 3136 | 74 | 274 | 2862 | 1025/2862 | 35.8% | 44/50 |
| All available prefill | 3450 | - | 3376 | 74 | 346 | 3030 | 1124/3030 | 37.1% | 44/50 |

## GT And GT+Pool Pass Rates

### First20 Comparable Scope

This is the cleanest condition comparison because GT-only exists for these same 14 executable First20 tasks. GT+ rows use the combined First20 unique-config prefill data across seeds 0-4 and noise modes random/hard/easy.

| Condition | Pool size meaning | Pass/Executed | Pass rate | Executable tasks |
| --- | --- | --- | --- | --- |
| GT-only | GT only | 37/70 | 52.9% | 14 |
| GT+5 | 5 | 124/210 | 59.0% | 14 |
| GT+10 | 10 | 117/210 | 55.7% | 14 |
| GT+20 | 20 | 126/210 | 60.0% | 14 |
| GT+50 | 50 | 136/210 | 64.8% | 14 |
| GT+100 | 100 | 0/0 | N/A | 0 |

First20 split by noise mode, with timeout failures excluded in the rightmost rate:

| Condition | Pool | Noise | Pass/Exec | Rate | Timeout fails | Pass/Non-timeout | Timeout-excl rate | Tasks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GT-only | GT only | - | 37/70 | 52.9% | 14 | 37/56 | 66.1% | 14 |
| GT+5 | 5 | random | 43/70 | 61.4% | 6 | 43/64 | 67.2% | 14 |
| GT+5 | 5 | easy | 40/70 | 57.1% | 12 | 40/58 | 69.0% | 14 |
| GT+5 | 5 | hard | 41/70 | 58.6% | 10 | 41/60 | 68.3% | 14 |
| GT+10 | 10 | random | 40/70 | 57.1% | 12 | 40/58 | 69.0% | 14 |
| GT+10 | 10 | easy | 36/70 | 51.4% | 10 | 36/60 | 60.0% | 14 |
| GT+10 | 10 | hard | 41/70 | 58.6% | 9 | 41/61 | 67.2% | 14 |
| GT+20 | 20 | random | 42/70 | 60.0% | 11 | 42/59 | 71.2% | 14 |
| GT+20 | 20 | easy | 42/70 | 60.0% | 10 | 42/60 | 70.0% | 14 |
| GT+20 | 20 | hard | 42/70 | 60.0% | 12 | 42/58 | 72.4% | 14 |
| GT+50 | 50 | random | 47/70 | 67.1% | 14 | 47/56 | 83.9% | 14 |
| GT+50 | 50 | easy | 41/70 | 58.6% | 9 | 41/61 | 67.2% | 14 |
| GT+50 | 50 | hard | 48/70 | 68.6% | 8 | 48/62 | 77.4% | 14 |
| GT+100 | 100 | random | 0/0 | N/A | 0 | 0/0 | N/A | 0 |
| GT+100 | 100 | easy | 0/0 | N/A | 0 | 0/0 | N/A | 0 |
| GT+100 | 100 | hard | 0/0 | N/A | 0 | 0/0 | N/A | 0 |

### 50-Report Source Scope

This recomputes the 50-task source files only (`sb_prefill_n5`, Smoke10, Smoke20). There is no GT-only baseline for Smoke10/Smoke20, so GT-only is not a 50-task metric.

| Condition | Pool size meaning | Pass/Executed | Pass rate | Executable tasks |
| --- | --- | --- | --- | --- |
| GT-only | N/A | N/A | N/A | First20 only; no Smoke10/Smoke20 GT-only baseline |
| GT+5 | 5 | 230/588 | 39.1% | 42 |
| GT+10 | 10 | 215/607 | 35.4% | 44 |
| GT+20 | 20 | 218/606 | 36.0% | 44 |
| GT+50 | 50 | 239/618 | 38.7% | 44 |
| GT+100 | 100 | 123/443 | 27.8% | 30 |

50-report source split by noise mode, with timeout failures excluded in the rightmost rate:

| Condition | Pool | Noise | Pass/Exec | Rate | Timeout fails | Pass/Non-timeout | Timeout-excl rate | Tasks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GT+5 | 5 | random | 73/196 | 37.2% | 49 | 73/147 | 49.7% | 42 |
| GT+5 | 5 | easy | 79/196 | 40.3% | 54 | 79/142 | 55.6% | 42 |
| GT+5 | 5 | hard | 78/196 | 39.8% | 56 | 78/140 | 55.7% | 42 |
| GT+10 | 10 | random | 75/205 | 36.6% | 61 | 75/144 | 52.1% | 44 |
| GT+10 | 10 | easy | 66/201 | 32.8% | 55 | 66/146 | 45.2% | 43 |
| GT+10 | 10 | hard | 74/201 | 36.8% | 59 | 74/142 | 52.1% | 43 |
| GT+20 | 20 | random | 73/201 | 36.3% | 45 | 73/156 | 46.8% | 43 |
| GT+20 | 20 | easy | 75/204 | 36.8% | 54 | 75/150 | 50.0% | 44 |
| GT+20 | 20 | hard | 70/201 | 34.8% | 60 | 70/141 | 49.6% | 43 |
| GT+50 | 50 | random | 80/206 | 38.8% | 58 | 80/148 | 54.1% | 44 |
| GT+50 | 50 | easy | 84/206 | 40.8% | 54 | 84/152 | 55.3% | 44 |
| GT+50 | 50 | hard | 75/206 | 36.4% | 53 | 75/153 | 49.0% | 44 |
| GT+100 | 100 | random | 42/150 | 28.0% | 52 | 42/98 | 42.9% | 30 |
| GT+100 | 100 | easy | 42/145 | 29.0% | 38 | 42/107 | 39.3% | 29 |
| GT+100 | 100 | hard | 39/148 | 26.4% | 39 | 39/109 | 35.8% | 30 |

### All Available Prefill Scope

This adds v4 seed0 First20 configs to the 50-report source. Use it as the most complete execution ledger, not as a causal comparison.

| Condition | Pool size meaning | Pass/Executed | Pass rate | Executable tasks |
| --- | --- | --- | --- | --- |
| GT-only | GT only | 37/70 | 52.9% | 14 First20 tasks only |
| GT+5 | 5 | 255/630 | 40.5% | 42 |
| GT+10 | 10 | 240/649 | 37.0% | 44 |
| GT+20 | 20 | 244/648 | 37.7% | 44 |
| GT+50 | 50 | 262/660 | 39.7% | 44 |
| GT+100 | 100 | 123/443 | 27.8% | 30 |

All-available split by noise mode, with timeout failures excluded in the rightmost rate:

| Condition | Pool | Noise | Pass/Exec | Rate | Timeout fails | Pass/Non-timeout | Timeout-excl rate | Tasks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GT-only | GT only | - | 37/70 | 52.9% | 14 | 37/56 | 66.1% | 14 |
| GT+5 | 5 | random | 81/210 | 38.6% | 50 | 81/160 | 50.6% | 42 |
| GT+5 | 5 | easy | 88/210 | 41.9% | 55 | 88/155 | 56.8% | 42 |
| GT+5 | 5 | hard | 86/210 | 41.0% | 59 | 86/151 | 57.0% | 42 |
| GT+10 | 10 | random | 83/219 | 37.9% | 64 | 83/155 | 53.5% | 44 |
| GT+10 | 10 | easy | 74/215 | 34.4% | 58 | 74/157 | 47.1% | 43 |
| GT+10 | 10 | hard | 83/215 | 38.6% | 62 | 83/153 | 54.2% | 43 |
| GT+20 | 20 | random | 81/215 | 37.7% | 47 | 81/168 | 48.2% | 43 |
| GT+20 | 20 | easy | 84/218 | 38.5% | 56 | 84/162 | 51.9% | 44 |
| GT+20 | 20 | hard | 79/215 | 36.7% | 61 | 79/154 | 51.3% | 43 |
| GT+50 | 50 | random | 89/220 | 40.5% | 62 | 89/158 | 56.3% | 44 |
| GT+50 | 50 | easy | 90/220 | 40.9% | 56 | 90/164 | 54.9% | 44 |
| GT+50 | 50 | hard | 83/220 | 37.7% | 55 | 83/165 | 50.3% | 44 |
| GT+100 | 100 | random | 42/150 | 28.0% | 52 | 42/98 | 42.9% | 30 |
| GT+100 | 100 | easy | 42/145 | 29.0% | 38 | 42/107 | 39.3% | 29 |
| GT+100 | 100 | hard | 39/148 | 26.4% | 39 | 39/109 | 35.8% | 30 |

## 20t v4 vs First20 P0 Consistency

The task-level correlation between v4 seed0 rates and First20 P0 unique-config rates across the 14 executable First20 tasks is 0.967. Mean task rate changes from 58.9% to 60.1%, a +1.2 percentage point difference.

| Task | v4 seed0 | v4 rate | P0 seeds1-4 | P0 rate | Combined seeds0-4 | Combined rate | Combined skipped |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3d-scan-calc | 12/12 | 100.0% | 47/48 | 97.9% | 59/60 | 98.3% | 0 |
| adaptive-cruise-control | 2/12 | 16.7% | 18/48 | 37.5% | 20/60 | 33.3% | 0 |
| azure-bgp-oscillation-route-leak | 1/12 | 8.3% | 0/48 | 0.0% | 1/60 | 1.7% | 0 |
| citation-check | 11/12 | 91.7% | 41/48 | 85.4% | 52/60 | 86.7% | 0 |
| civ6-adjacency-optimizer | 1/12 | 8.3% | 5/48 | 10.4% | 6/60 | 10.0% | 0 |
| court-form-filling | 0/0 | N/A | 0/0 | N/A | 0/0 | N/A | 60 |
| crystallographic-wyckoff-position-analysis | 0/0 | N/A | 0/0 | N/A | 0/0 | N/A | 60 |
| dapt-intrusion-detection | 10/12 | 83.3% | 47/48 | 97.9% | 57/60 | 95.0% | 0 |
| data-to-d3 | 0/0 | N/A | 0/0 | N/A | 0/0 | N/A | 60 |
| dialogue-parser | 11/12 | 91.7% | 47/48 | 97.9% | 58/60 | 96.7% | 0 |
| dynamic-object-aware-egomotion | 0/0 | N/A | 0/0 | N/A | 0/0 | N/A | 60 |
| earthquake-phase-association | 9/12 | 75.0% | 37/48 | 77.1% | 46/60 | 76.7% | 0 |
| earthquake-plate-calculation | 12/12 | 100.0% | 42/48 | 87.5% | 54/60 | 90.0% | 0 |
| econ-detrending-correlation | 9/12 | 75.0% | 41/48 | 85.4% | 50/60 | 83.3% | 0 |
| edit-pdf | 0/0 | N/A | 0/0 | N/A | 0/0 | N/A | 60 |
| energy-ac-optimal-power-flow | 5/12 | 41.7% | 14/48 | 29.2% | 19/60 | 31.7% | 0 |
| energy-market-pricing | 12/12 | 100.0% | 45/48 | 93.8% | 57/60 | 95.0% | 0 |
| enterprise-information-search | 0/12 | 0.0% | 3/48 | 6.2% | 3/60 | 5.0% | 0 |
| exceltable-in-ppt | 4/12 | 33.3% | 17/48 | 35.4% | 21/60 | 35.0% | 0 |
| exoplanet-detection-period | 0/0 | N/A | 0/0 | N/A | 0/0 | N/A | 16 |

## 50-Report Source Recomputed With Unified Rule

| Batch | Task | Pass/Executed | Rate | Skipped configs |
| --- | --- | --- | --- | --- |
| First20 P0 | 3d-scan-calc | 47/48 | 97.9% | 0 |
| First20 P0 | adaptive-cruise-control | 18/48 | 37.5% | 0 |
| First20 P0 | azure-bgp-oscillation-route-leak | 0/48 | 0.0% | 0 |
| First20 P0 | citation-check | 41/48 | 85.4% | 0 |
| First20 P0 | civ6-adjacency-optimizer | 5/48 | 10.4% | 0 |
| First20 P0 | court-form-filling | 0/0 | N/A | 48 |
| First20 P0 | crystallographic-wyckoff-position-analysis | 0/0 | N/A | 48 |
| First20 P0 | dapt-intrusion-detection | 47/48 | 97.9% | 0 |
| First20 P0 | data-to-d3 | 0/0 | N/A | 48 |
| First20 P0 | dialogue-parser | 47/48 | 97.9% | 0 |
| First20 P0 | dynamic-object-aware-egomotion | 0/0 | N/A | 48 |
| First20 P0 | earthquake-phase-association | 37/48 | 77.1% | 0 |
| First20 P0 | earthquake-plate-calculation | 42/48 | 87.5% | 0 |
| First20 P0 | econ-detrending-correlation | 41/48 | 85.4% | 0 |
| First20 P0 | edit-pdf | 0/0 | N/A | 48 |
| First20 P0 | energy-ac-optimal-power-flow | 14/48 | 29.2% | 0 |
| First20 P0 | energy-market-pricing | 45/48 | 93.8% | 0 |
| First20 P0 | enterprise-information-search | 3/48 | 6.2% | 0 |
| First20 P0 | exceltable-in-ppt | 17/48 | 35.4% | 0 |
| First20 P0 | exoplanet-detection-period | 0/0 | N/A | 4 |
| Smoke10 | financial-modeling-qa | 1/75 | 1.3% | 0 |
| Smoke10 | flood-risk-analysis | 38/75 | 50.7% | 0 |
| Smoke10 | glm-lake-mendota | 0/68 | 0.0% | 0 |
| Smoke10 | hvac-control | 68/75 | 90.7% | 0 |
| Smoke10 | jpg-ocr-stat | 2/52 | 3.8% | 0 |
| Smoke10 | mario-coin-counting | 43/75 | 57.3% | 0 |
| Smoke10 | pptx-reference-formatting | 38/75 | 50.7% | 0 |
| Smoke10 | r2r-mpc-control | 20/75 | 26.7% | 0 |
| Smoke10 | sec-financial-report | 32/75 | 42.7% | 0 |
| Smoke10 | software-dependency-audit | 58/75 | 77.3% | 0 |
| Smoke20 | fix-build-agentops | 0/75 | 0.0% | 0 |
| Smoke20 | fix-build-google-auto | 0/75 | 0.0% | 0 |
| Smoke20 | fix-druid-loophole-cve | 23/75 | 30.7% | 0 |
| Smoke20 | grid-dispatch-operator | 60/75 | 80.0% | 0 |
| Smoke20 | lake-warming-attribution | 43/75 | 57.3% | 0 |
| Smoke20 | offer-letter-generator | 69/75 | 92.0% | 0 |
| Smoke20 | pg-essay-to-audiobook | 0/75 | 0.0% | 0 |
| Smoke20 | powerlifting-coef-calc | 25/75 | 33.3% | 0 |
| Smoke20 | protein-expression-analysis | 14/75 | 18.7% | 0 |
| Smoke20 | python-scala-translation | 0/60 | 0.0% | 15 |
| Smoke20 | reserves-at-risk-calc | 3/75 | 4.0% | 0 |
| Smoke20 | sales-pivot-analysis | 68/75 | 90.7% | 0 |
| Smoke20 | setup-fuzzing-py | 0/75 | 0.0% | 0 |
| Smoke20 | shock-analysis-demand | 0/75 | 0.0% | 0 |
| Smoke20 | shock-analysis-supply | 0/75 | 0.0% | 0 |
| Smoke20 | simpo-code-reproduction | 0/75 | 0.0% | 0 |
| Smoke20 | travel-planning | 16/60 | 26.7% | 15 |
| Smoke20 | video-filler-word-remover | 0/75 | 0.0% | 0 |
| Smoke20 | weighted-gdp-calc | 0/75 | 0.0% | 0 |
| Smoke20 | xlsx-recover-data | 0/75 | 0.0% | 0 |

## Baselines From 20t v4

`sb_baselines_n5.jsonl` gives Noskill 29/70 = 41.4% and GT-only 37/70 = 52.9%. These baseline conditions exist only for the First20 14 executable tasks, so they should not be mixed into Smoke10/Smoke20 causal claims.

## Accuracy And Explainability Verdict

Accuracy: under the unique-config execution denominator, the two reports are consistent for First20. The prior 17.3% First20 number in the 50-task report is a raw-row ledger statistic, not a comparable performance statistic.

Explainability: 20t v4 remains the better causal/effect-size report because it includes paired baselines and warns about pseudoreplication. The 50-task report is better as a coverage and reliability ledger after correcting its denominator caveats.

Caveats:

- First20 has 6 heavy-Dockerfile tasks that never executed on host.
- First20 P0 is missing 44 skipped exoplanet configs from the nominal full design.
- Smoke10 is incomplete for `jpg-ocr-stat` and `glm-lake-mendota`.
- Smoke20 has 30 design skips at pool_size=5 for two tasks with 6 GT skills.
- Aggregate rates across all 50 tasks are descriptive only because task mix and coverage differ by batch.

## Produced Artifacts

- `results/unified_unique_config_report.md`
- `results/unified_unique_config_report.pdf`
