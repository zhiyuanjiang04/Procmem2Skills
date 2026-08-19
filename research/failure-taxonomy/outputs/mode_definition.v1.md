# Canonical Failure/Success Mode Taxonomy (v1)

Generated from 239 LLM-labeled trajectories via claude-sonnet-4-6, split into 4 batches and merged.

Total unified modes: 12

## Modes

### `skill_guided_success` (n=54)

Agent successfully completed the task by reading and faithfully applying SKILL.md guidance. Skill content provided domain knowledge, procedural steps, API choices, or known-failure warnings that the agent leveraged to produce a correct, oracle-passing result.

*Merged from batch modes: skill_guided_clean_execution, skill_guided_success*

### `workflow_guided_success` (n=47)

Agent successfully completed the task by following workflow trace hints or prior-attempt artifacts that directed tool selection, command sequencing, or parameter strategy. The workflow artifact was substantively actionable and led directly to a passing oracle result.

*Merged from batch modes: workflow_guided_clean_execution, workflow_guided_success*

### `autonomous_clean_success` (n=16)

Agent produced a correct, oracle-passing result without skill or workflow augmentation, relying on its own domain knowledge, systematic exploration, self-directed verification, or pre-existing environment state discovered and reused opportunistically.

*Merged from batch modes: clean_autonomous_success, clean_implementation_success, environment_state_leveraged_success, raw_clean_success*

### `timeout_budget_exhaustion` (n=16)

Agent exhausted the wall-clock time or context budget before producing valid output, typically by spending it in open-ended search loops, exhaustive analysis, hyperparameter sweeps, or non-converging optimization without a fallback commit strategy.

*Merged from batch modes: environment_resource_timeout, exploration_loop_timeout, timeout_budget_exhaustion_failure, timeout_budget_exhausted*

### `environment_infrastructure_failure` (n=10)

Task failed due to a missing dependency, incompatible package, absent input file, out-of-memory kill, or environment state corrupted by the agent's own prior actions. The required execution context could not be established or was destroyed during the attempt.

*Merged from batch modes: offline_environment_dependency_failure, build_environment_oom_kill, environment_setup_failure, environment_corrupted_by_agent_actions*

### `static_verification_without_runtime` (n=21)

Agent validated correctness through static inspection, syntax checks, or self-scoped audits that could not detect behavioral failures. Oracle evaluation caught errors invisible to the agent's verification, including different test partitions, stricter thresholds, or coverage gaps bounded by the agent's own prior output.

*Merged from batch modes: self_referential_verification_gap, static_check_misses_runtime_failure, static_only_verification_failure, oracle_partition_mismatch, verification_mismatch_self_vs_oracle*

### `output_format_schema_mismatch` (n=17)

Implementation logic is functionally plausible but oracle rejects the submission due to wrong field names, incorrect precision or units, hardcoded status labels, schema structure divergence, or evaluation criteria the agent did not anticipate from internal testing alone.

*Merged from batch modes: oracle_evaluation_mismatch, output_format_schema_mismatch, oracle_format_mismatch_failure, output_format_or_value_mismatch*

### `algorithmic_logic_error` (n=29)

Agent selected a fundamentally incorrect algorithm, formula, API primitive, or extraction strategy. Covers mathematical model errors, wrong optimization direction, misclassification heuristics, incorrect spreadsheet formula placement, wrong regex semantics, and parser implementations that fail on structurally different inputs.

*Merged from batch modes: incorrect_mathematical_model, domain_classification_error, spreadsheet_formula_placement_error, algorithmic_formulation_error, wrong_api_or_algorithm_usage, parser_fragility_unseen_input_failure*

### `shell_code_corruption` (n=6)

Malformed shell constructs — double-escaped regex patterns in bash heredocs or inline Python embedded via heredoc quoting — produce silently incorrect field extractions or blocked execution that the agent does not detect before submission.

*Merged from batch modes: regex_escaping_corruption, heredoc_shell_corruption*

### `background_service_lifecycle_failure` (n=7)

Agent launched a required long-running process but failed to properly daemonize or detach it, causing the service to be dead or unreachable at oracle evaluation time due to a blocking foreground run, silent crash from a port conflict, or missing setsid.

*Merged from batch modes: server_persistence_failure, background_service_lifecycle_failure*

### `skill_guidance_misapplied_or_ignored` (n=10)

Agent had access to SKILL.md guidance but skipped required steps, applied an incorrect mitigation for a correctly identified issue, took a disallowed path, or followed guidance so partially that it backfired. Failure stems from procedural non-compliance or misinterpretation rather than a capability ceiling.

*Merged from batch modes: skill_guidance_misapplied, skill_misguidance, skill_guidance_ignored_or_misapplied, skill_guidance_ignored_or_diverged*

### `capability_or_safety_limit` (n=6)

Task failed because it required capabilities at or beyond the agent's current limits — exact PRNG reproduction, robust OCR under constrained fallback, precise credential identification under truncated visibility — or the agent declined on safety grounds.

*Merged from batch modes: capability_or_safety_limit, credential_identification_error_failure*
