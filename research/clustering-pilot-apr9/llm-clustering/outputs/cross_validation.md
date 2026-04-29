# LLM Clustering vs DBSCAN eps=0.40

LLM clusters: 95, unclustered: 110
DBSCAN multi-task clusters: 26

## Pairwise agreement

- Tasks compared (non-noise & non-unclustered in both): **46**
- Total pairs: 1035
- **Rand Index: 0.9256**
- Same cluster in both: 7
- Together in DBSCAN only: 76
- Together in LLM only: 1
- Apart in both: 951

## How each DBSCAN cluster maps into LLM clustering

`unity` = fraction of cluster members that ended up in the same LLM cluster.
1.0 means LLM kept them all together; lower means LLM split them.

| DBSCAN | size | unity | LLM distribution |
|--------|-----:|------:|------------------|
| D8 | 16 | 0.25 | _UNCLUST×4, B3_C3.2×2, B4_C12.1×2, B1_C15.1×1, B2_C13.1×1, B2_C1.2×1, B3_C13.1×1, B3_C1.1×1, B3_C1.3×1, B3_C1.2×1, B4_C6.2×1 |
| D0 | 3 | 0.333 | B1_C10.2×1, B1_C10.1×1, B3_C2.2×1 |
| D1 | 3 | 0.667 | B1_C7.1×2, B2_C11.1×1 |
| D6 | 3 | 1.0 | _UNCLUST×3 |
| D9 | 3 | 0.667 | _UNCLUST×2, B2_C3.2×1 |
| D2 | 2 | 0.5 | B1_C14.1×1, B2_C10.1×1 |
| D3 | 2 | 0.5 | B1_C3.1×1, B1_C3.2×1 |
| D4 | 2 | 0.5 | _UNCLUST×1, B1_C1.1×1 |
| D5 | 2 | 0.5 | B1_C1.1×1, _UNCLUST×1 |
| D7 | 2 | 1.0 | _UNCLUST×2 |
| D10 | 2 | 1.0 | _UNCLUST×2 |
| D11 | 2 | 0.5 | B1_C5.1×1, B1_C5.2×1 |
| D12 | 2 | 0.5 | B2_C2.1×1, B2_C2.2×1 |
| D13 | 2 | 1.0 | _UNCLUST×2 |
| D14 | 2 | 1.0 | _UNCLUST×2 |
| D15 | 2 | 1.0 | B2_C1.1×2 |
| D16 | 2 | 0.5 | B2_C9.1×1, B2_C9.2×1 |
| D17 | 2 | 1.0 | B2_C4.1×2 |
| D18 | 2 | 1.0 | B2_C5.1×2 |
| D19 | 2 | 0.5 | B3_C8.1×1, _UNCLUST×1 |
| D20 | 2 | 0.5 | _UNCLUST×1, B4_C10.2×1 |
| D21 | 2 | 0.5 | _UNCLUST×1, B4_C3.1×1 |
| D22 | 2 | 1.0 | B3_C4.1×2 |
| D23 | 2 | 1.0 | _UNCLUST×2 |
| D24 | 2 | 0.5 | B4_C4.1×1, B4_C4.2×1 |
| D25 | 2 | 0.5 | B4_C5.1×1, B4_C5.2×1 |

## LLM clusters DBSCAN missed (≥2 DBSCAN-noise members)

### LLM B1_C2.1 — "Diagnose and repair broken Python environments by resolving pip/setuptools failures and conda dependency conflicts."
  size=2, DBSCAN-noise members: 2 (100%)
  members: broken-python, conda-env-conflict-resolution

### LLM B1_C4.1 — "Implement or modify statistical algorithms in R—sampling methods and Bayesian networks—using domain-specific packages (e.g., bnlearn, ARS) with direct Rscript execution, modular code structure, and output validation."
  size=2, DBSCAN-noise members: 2 (100%)
  members: adaptive-rejection-sampler, bn-fit-modify

### LLM B1_C6.1 — "Enumerate combinatorial spaces (permutations, arithmetic expression trees) with backtracking or pruning, using itertools and fractions.Fraction for exact arithmetic, to find solutions satisfying hard constraints."
  size=2, DBSCAN-noise members: 2 (100%)
  members: assign-seats, countdown-game

### LLM B1_C8.1 — "Implement missing PyTorch neural network components or reinforcement learning training loops, and diagnose common PyTorch training failures such as gradient issues, loss plateaus, or divergence."
  size=3, DBSCAN-noise members: 3 (100%)
  members: attention-mil, cartpole-rl-training, classifier-debug

### LLM B2_C7.1 — "Generate a self-signed SSL certificate with openssl and configure a server (nginx or Jupyter) to terminate HTTPS by referencing the cert and key in the server's config file, then restart the service."
  size=2, DBSCAN-noise members: 2 (100%)
  members: home-server-https, jupyter-notebook-server

### LLM B3_C2.1 — "Composing and executing SQL queries (joins, window functions, aggregations, deduplication, type normalization) against structured data in DuckDB or PostgreSQL to produce a query-result dataset."
  size=2, DBSCAN-noise members: 2 (100%)
  members: pandas-sql-query, postgres-csv-clean

### LLM B3_C7.1 — "Configuring and starting network service daemons by editing config files, initializing any required backend state, starting processes, and verifying live endpoints via a protocol probe."
  size=2, DBSCAN-noise members: 2 (100%)
  members: mailman, nginx-request-logging

### LLM B3_C9.1 — "Migrate domain-specific scientific numerical code across frameworks or languages (MATLAB→NumPy, NEURON→Jaxley) by mapping source numerical semantics and API calls to modern Python equivalents, then validating agreement between reference and ported outputs."
  size=2, DBSCAN-noise members: 2 (100%)
  members: matlab-python-conversion, neuron-to-jaxley-conversion

### LLM B3_C15.1 — "Implementing a numerical iterative algorithm (fixed-step ODE integrator, entropic regularized transport) from a mathematical recurrence or optimization formulation in Python, managing floating-point step size or convergence tolerances, and verifying output against a quantitative accuracy threshold."
  size=2, DBSCAN-noise members: 2 (100%)
  members: ode-solver-rk4, optimal-transport

### LLM B4_C1.1 — "Reproduce a reported bug in an existing open-source Python library, locate the faulty code path with git/grep/find, apply a minimal targeted fix (one line to a few lines), and verify correctness against the project's existing test suite."
  size=4, DBSCAN-noise members: 4 (100%)
  members: swe-bench-astropy-1, swe-bench-astropy-2, swe-bench-fsspec, swe-bench-langcodes

### LLM B4_C2.1 — "Identify an HTTP-layer vulnerability (SQLi, SpEL/SSTI injection, unauthenticated endpoint exposure) from source or app resources, craft the exploit payload, and deliver it via curl or Python HTTP requests to achieve the stated objective."
  size=3, DBSCAN-noise members: 3 (100%)
  members: sql-injection-attack, spring-messaging-vul, security-vulhub-minio

### LLM B4_C2.3 — "Read vulnerable framework or application source code, apply a minimal targeted patch that eliminates the vulnerability class at the framework level, and rebuild the project so the fix takes effect at runtime."
  size=2, DBSCAN-noise members: 2 (100%)
  members: vul-flask, vul-flink

### LLM B1_C1.1 — "Configure, patch, and compile C/C++ or system-level software from source using make/cmake/gcc, resolving toolchain errors, missing dependencies, and platform-specific build issues."
  size=7, DBSCAN-noise members: 5 (71%)
  members: 3d-model-format-legacy, build-pmars, build-pov-ray, build-stp, caffe-cifar-10, build-tcc-qemu, build-linux-kernel-qemu

### LLM B3_C2.2 — "Reading heterogeneous file formats (CSV, JSON, Parquet) in Python, applying field-level transformations (regex extraction, type coercion, schema renaming, priority-based conflict resolution, nested structure building), and writing a structured output file."
  size=3, DBSCAN-noise members: 2 (67%)
  members: pandas-etl, multi-source-data-merger, organization-json-generator

### LLM B4_C3.1 — "Recover hidden information from obfuscated artifacts by statically identifying the encoding or transformation scheme (base64, gzip, bzip2, custom hash) and inverting it with standard decoding tools or mathematical inverse computation, without executing the target."
  size=3, DBSCAN-noise members: 2 (67%)
  members: reverse-engineering, shell-deobfuscation, recover-obfuscated-files
