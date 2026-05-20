# Paired Failure-Mode Report (v1)

Analyzed 528 (task, setting) triples across raw / workflow / skill arms.

## 1. Overall success rate per arm

| arm | success | total | rate |
|---|---:|---:|---:|
| raw | 312 | 528 | 59.1% |
| workflow | 295 | 528 | 55.9% |
| skill | 327 | 528 | 61.9% |

## 2. Paired success-rate delta (bootstrap 95% CI, n_iter=1000)

| comparison | mean delta | 95% CI | n_pairs |
|---|---:|---|---:|
| workflow_vs_raw | -0.0322 | [-0.0814, +0.0208] | 528 |
| skill_vs_raw | +0.0284 | [-0.0227, +0.0795] | 528 |
| skill_vs_workflow | +0.0606 | [+0.0076, +0.1136] | 528 |

## 3. Net-effect distribution

How does the treatment compare to baseline overall?

| comparison | fixed | regressed | unchanged | mixed | not_comparable |
|---|---:|---:|---:|---:|---:|
| workflow_vs_raw | 100 | 130 | 261 | 37 | 0 |
| skill_vs_raw | 114 | 108 | 272 | 34 | 0 |
| skill_vs_workflow | 121 | 89 | 295 | 23 | 0 |

## 4. Mode frequency per arm (out of 528 triples)

| mode | raw | workflow | skill |
|---|---|---|---|
| skill_guided_success | 55 | 2 | 325 |
| workflow_guided_success | 0 | 288 | 0 |
| autonomous_clean_success | 257 | 4 | 1 |
| static_verification_without_runtime | 66 | 66 | 62 |
| algorithmic_logic_error | 44 | 58 | 39 |
| timeout_budget_exhaustion | 9 | 56 | 23 |
| output_format_schema_mismatch | 39 | 20 | 17 |
| skill_guidance_misapplied_or_ignored | 4 | 2 | 53 |
| environment_infrastructure_failure | 28 | 9 | 1 |
| background_service_lifecycle_failure | 14 | 13 | 4 |
| shell_code_corruption | 6 | 10 | 1 |
| capability_or_safety_limit | 6 | 0 | 2 |

## 5. Paired mode delta — skill vs raw (sorted by net help)

`fixed_by_treatment` = mode appeared in raw, gone in skill.
`introduced_by_treatment` = mode appeared in skill, not in raw.

| mode | fixed_by_treatment | introduced_by_treatment | net_delta |
|---|---|---|---|
| environment_infrastructure_failure | 27 | 0 | 27 |
| output_format_schema_mismatch | 30 | 8 | 22 |
| background_service_lifecycle_failure | 10 | 2 | 8 |
| shell_code_corruption | 7 | 1 | 6 |
| capability_or_safety_limit | 5 | 1 | 4 |
| algorithmic_logic_error | 31 | 28 | 3 |
| autonomous_clean_success | 2 | 0 | 2 |
| skill_guided_success | 0 | 1 | -1 |
| static_verification_without_runtime | 28 | 34 | -6 |
| timeout_budget_exhaustion | 2 | 20 | -18 |
| skill_guidance_misapplied_or_ignored | 4 | 52 | -48 |

## 6. Paired mode delta — workflow vs raw

| mode | fixed_by_treatment | introduced_by_treatment | net_delta |
|---|---|---|---|
| environment_infrastructure_failure | 25 | 7 | 18 |
| output_format_schema_mismatch | 21 | 5 | 16 |
| capability_or_safety_limit | 6 | 0 | 6 |
| skill_guidance_misapplied_or_ignored | 3 | 2 | 1 |
| background_service_lifecycle_failure | 8 | 7 | 1 |
| autonomous_clean_success | 1 | 0 | 1 |
| workflow_guided_success | 0 | 1 | -1 |
| shell_code_corruption | 5 | 9 | -4 |
| static_verification_without_runtime | 33 | 44 | -11 |
| algorithmic_logic_error | 29 | 43 | -14 |
| timeout_budget_exhaustion | 2 | 48 | -46 |

## 7. Paired mode delta — skill vs workflow

| mode | fixed_by_treatment | introduced_by_treatment | net_delta |
|---|---|---|---|
| timeout_budget_exhaustion | 33 | 6 | 27 |
| algorithmic_logic_error | 31 | 20 | 11 |
| shell_code_corruption | 11 | 1 | 10 |
| background_service_lifecycle_failure | 11 | 2 | 9 |
| environment_infrastructure_failure | 8 | 1 | 7 |
| output_format_schema_mismatch | 16 | 10 | 6 |
| static_verification_without_runtime | 28 | 25 | 3 |
| workflow_guided_success | 1 | 0 | 1 |
| capability_or_safety_limit | 0 | 2 | -2 |
| skill_guidance_misapplied_or_ignored | 2 | 40 | -38 |

## 8. Mechanism distribution

How is the treatment helping (or hurting)?

### `skill_mechanism`
- procedural_anchor: 347
- counterproductive: 87
- failure_warning: 64
- knowledge_injection: 24
- none: 6

### `workflow_mechanism`
- procedural_anchor: 381
- counterproductive: 65
- none: 41
- knowledge_injection: 31
- failure_warning: 10

## 9. Trends by setting (workflow-success mix)

Setting `5s0f` = 5 success workflows, 0 failure. `0s5f` = 5 failure, 0 success.

### `skill_vs_raw` net_effect by setting
| setting | fixed | regressed | unchanged | mixed |
|---|---:|---:|---:|---:|
| 5s0f | 19 | 12 | 51 | 6 |
| 4s1f | 22 | 16 | 46 | 4 |
| 3s2f | 22 | 21 | 38 | 7 |
| 2s3f | 20 | 15 | 51 | 2 |
| 1s4f | 18 | 16 | 48 | 6 |
| 0s5f | 13 | 28 | 38 | 9 |

### `workflow_vs_raw` net_effect by setting
| setting | fixed | regressed | unchanged | mixed |
|---|---:|---:|---:|---:|
| 5s0f | 17 | 18 | 45 | 8 |
| 4s1f | 18 | 20 | 45 | 5 |
| 3s2f | 18 | 17 | 46 | 7 |
| 2s3f | 12 | 21 | 50 | 5 |
| 1s4f | 17 | 28 | 37 | 6 |
| 0s5f | 18 | 26 | 38 | 6 |

## 10. By benchmark

### `skill_vs_raw` net_effect by benchmark
| benchmark | fixed | regressed | unchanged | mixed |
|---|---:|---:|---:|---:|
| skillsbench | 38 | 26 | 69 | 11 |
| terminalbench2 | 28 | 47 | 96 | 15 |
| terminalbenchpro | 48 | 35 | 107 | 8 |

## 11. Agent token / cost / duration per arm (from manifest)

### Per-arm means (median in parens)

| arm | input_tokens | output_tokens | duration_sec |
|---|---:|---:|---:|
| raw | 699,084 (252,490) | 15,389 (10,688) | 452 (259) |
| workflow | 445,177 (188,330) | 8,415 (7,316) | 341 (173) |
| skill | - | - | 322 (161) |

### Paired per-task deltas (treatment minus baseline)

| comparison | Δ input_tokens (mean) | Δ output_tokens (mean) | Δ duration_sec (mean) |
|---|---:|---:|---:|
| workflow_vs_raw | -253,908 | -6,975 | -111 |
| skill_vs_raw | - | - | -130 |
| skill_vs_workflow | - | - | -20 |
