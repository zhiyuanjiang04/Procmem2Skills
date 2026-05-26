# Untouched SkillsBench Tasks — Coverage Expansion Design

**Status:** supersedes the previous `REMAINING_6_TASKS_DESIGN.md` (which scoped only the 6 v4-skipped tasks). Per direction, the 6 v4-skipped tasks are **out of scope** here.

**Scope:** The untouched-SkillsBench expansion ledger for `20t_report_v4.pdf`: 68 GT-bearing tasks that the actual runner never selected, plus 1 malformed task that the runner silently dropped. This is the 69-task design object named in this file. Bring the evaluable portion under the prefill-context execution eval so a v6/v7 report can speak to the full 89-row SkillsBench source file — the dataset size the MANIFEST already advertises ("84 lightweight SB tasks × 4 sizes × 3 noise = 1008 trials").

**Counting note:** A raw source-file shorthand says "89 total rows − first 20 = 69 never loaded, plus PDF-Crosspage = 70." The code path is slightly different: `run_trial.py` first drops rows with empty `gt_skills`, so `PDF-Crosspage-Table-Normalization-to-Excel` is removed before `--limit 20`; `exoplanet-detection-period` then becomes the 20th GT-bearing selected task and appears in v4 only as a skipped heavy-Dockerfile row. Therefore the execution-reproducible inventory here is `68 never-selected GT tasks + 1 malformed PDF task = 69`, matching the Bucket counts below (`31 + 30 + 7 + 1`).

**Non-goals:** TerminalBench, Phase-B factorial controls, the 6 v4-skipped tasks. Each is its own initiative.

---

## 1. Why v4 missed these tasks

v4 was launched via `run_prefill_n5_hard_easy.sh` with `--limit 20`. `run_trial.py` loads `testsets/data/skillsbench_tasks.jsonl`, drops tasks whose `gt_skills` list is empty, and then stops after 20 GT-bearing entries. Source order is loosely alphabetic with capitals first (`3d-scan-calc`, `PDF-Crosspage…`, `adaptive-cruise-control`, …). Because `PDF-Crosspage…` has no GT skill, the 20 selected GT-bearing entries are `3d-scan-calc` through `exoplanet-detection-period`; the 68 GT-bearing tasks from `financial-modeling-qa` through `xlsx-recover-data` are never selected.

One additional task, `PDF-Crosspage-Table-Normalization-to-Excel` (source position #2), exists in the JSONL but has `gt_skills=[]`, so the runner silently filters it before trial generation and emits no JSONL line. Its directory also contains only `instruction.md` — no `environment/Dockerfile`, `tests/`, or `solution/` — so even a forced run would have no valid pass/fail oracle.

**Total in this design: 69 task records** — 68 never-selected GT-bearing tasks plus 1 malformed no-GT/no-test task.

---

## 2. Inventory by execution feasibility

Classification uses two regexes against `environment/Dockerfile`:

- *Old regex* (`run_trial.py:43`): `(playwright|nodejs|npm|texlive|poppler|gcc|build-essential|cuda|chromium|docker)` — whole-file substring; many false positives.
- *Proposed strict regex*: `^\s*(?:FROM|RUN)\s+.*\b(playwright|chromium|nodejs|npm|texlive|cuda)\b` — line-anchored to install commands; drops the noisy tokens.

| Bucket | Count | Treatment |
|--------|------:|-----------|
| **A. Run as-is** | 31 | Current runner accepts them after removing `--limit 20`. |
| **B. Old-regex false positives** | 30 | Triggered by comment-only `docker`, `gcc`, `build-essential`, or `poppler`; tighten regex and provision host dependencies. |
| **C. True heavy/browser tasks** | 7 | npm/Chromium/browser tests; route through the container branch. |
| **D. Malformed spec** | 1 | `PDF-Crosspage…` has no tests/Dockerfile/solution; file upstream issue and permanently exclude. |

### Bucket A — Run as-is (31 tasks)

Pass both regexes. Pipeline accepts them today; the only reason v4 didn't run them is `--limit 20`.

```
financial-modeling-qa            fix-build-agentops           fix-build-google-auto
fix-druid-loophole-cve           flood-risk-analysis          glm-lake-mendota
grid-dispatch-operator           hvac-control                 jpg-ocr-stat
lake-warming-attribution         mario-coin-counting          offer-letter-generator
pg-essay-to-audiobook            powerlifting-coef-calc       pptx-reference-formatting
protein-expression-analysis      python-scala-translation     r2r-mpc-control
reserves-at-risk-calc            sales-pivot-analysis         sec-financial-report
setup-fuzzing-py                 shock-analysis-demand        shock-analysis-supply
simpo-code-reproduction          software-dependency-audit    travel-planning
video-filler-word-remover        video-silence-remover        weighted-gdp-calc
xlsx-recover-data
```

Domain mix: 5 finance/econ, 4 video/audio, 4 build/dev-env, 3 manufacturing/operations, 3 data-quality, 3 document/office, 2 science, others. Healthy diversity — broadens v4's narrow domain coverage on its own.

### Bucket B — Blocked only by the leaky old regex (30 tasks)

These have light-to-medium real footprints; the old regex tripped on substrings (`docker` inside comments, `gcc` inside compiler version pins, `build-essential` for compiled wheels, `poppler` for PDF utilities already on most boxes). After the regex fix they're host-executable.

| Task | Old trigger | What's actually in the Dockerfile |
|------|-------------|-----------------------------------|
| `find-topk-similiar-chemicals` | `build-essential` | RDKit / scikit-learn wheels |
| `fix-erlang-ssh-cve` | `gcc` | Erlang/OTP source build |
| `flink-query` | `build-essential` | Apache Flink + PyFlink |
| `gh-repo-analytics` | `docker` (comment) | pure Python (`gh` CLI optional) |
| `gravitational-wave-detection` | `build-essential`, `gcc` | gwpy/lalsuite |
| `invoice-fraud-detection` | `build-essential`, `poppler` | PDF + sklearn |
| `jax-computing-basics` | `build-essential` | jax-cpu wheels |
| `lab-unit-harmonization` | `build-essential` | pint, pandas |
| `lean4-proof` | `build-essential` | lean4 toolchain (real, but elan auto-installs) |
| `manufacturing-codebook-normalization` | `Docker` (comment) | pandas |
| `manufacturing-equipment-maintenance` | `Docker` (comment) | pandas + lifelines |
| `manufacturing-fjsp-optimization` | `Docker` (comment) | OR-tools |
| `mars-clouds-clustering` | `build-essential`, `gcc` | astropy + sklearn |
| `mhc-layer-impl` | `Docker` (comment) | PyTorch CPU |
| `multilingual-video-dubbing` | `build-essential` | edge-tts + ffmpeg |
| `organize-messy-files` | `poppler` | PDF mime detection |
| `paper-anonymizer` | `poppler` | PDF redaction |
| `parallel-tfidf-search` | `build-essential` | scikit-learn |
| `pddl-tpp-planning` | `build-essential` | unified-planning |
| `pdf-excel-diff` | `poppler` | pdfplumber + openpyxl |
| `pedestrian-traffic-counting` | `poppler` | OpenCV (poppler unused at runtime) |
| `quantum-numerical-simulation` | `build-essential` | qutip / numpy |
| `seismic-phase-picking` | `Docker` (comment) | obspy + numpy |
| `speaker-diarization-subtitles` | `build-essential` | pyannote + ffmpeg |
| `spring-boot-jakarta-migration` | `Docker` (comment) | maven/jdk (real, host-installable) |
| `syzkaller-ppdev-syzlang` | `build-essential` | linux headers, fuzz harness |
| `taxonomy-tree-merge` | `build-essential` | pandas |
| `trend-anomaly-causal-inference` | `build-essential` | dowhy / scipy |
| `video-tutorial-indexer` | `nodejs` | yt-dlp + Python; nodejs is a stray (maybe used to build a JS index) — verify |
| `virtualhome-agent-planning` | `build-essential` | virtualhome simulator |

A small number of these (Lean4, Spring Boot/Maven, syzkaller, gwpy/LALSuite) do bring real toolchains. They're still tractable on host but warrant a per-task provision check before promotion.

### Bucket C — Genuinely browser/JS-bound (7 tasks)

Tests load assets in headless Chromium or assert on built JS bundles. Container path is the only sane way to run them.

```
fix-visual-stability         latex-formula-extraction     react-performance-debugging
scheduling-email-assistant   suricata-custom-exfil        threejs-structure-parser
threejs-to-obj
```

### Bucket D — Malformed spec (1 task)

```
PDF-Crosspage-Table-Normalization-to-Excel
```

Has `instruction.md` only. No tests means no `pass/fail` signal — un-evaluable until upstream provides `tests/test_outputs.py` + a Dockerfile. Drop from the grid; file an upstream issue against the SkillsBench task author.

---

## 3. Cost ceiling

The MANIFEST quote is `1008 trials ≈ 6–8h` for 84 tasks at full grid (4 pool × 3 noise = 12 configs/task, seed=0). For the 68 evaluable tasks in this design (Buckets A+B+C; Bucket D excluded):

```
68 tasks × 12 prefill configs = 816 trials (seed=0)
68 tasks × 5 noskill seeds    = 340 trials
68 tasks × 5 gt-only seeds    = 340 trials
                              ───────────────
                                1,496 trials  (~10–14h wall on the eval box, 4-way concurrency)
```

If we want Prefill at N=5 (the v4 §11 "P0: enables variance estimation"), add `68 × 12 × 4 = 3,264` more trials — *roughly a day per condition addition*. Run order in §5 is sized so we can stop after each phase and still publish meaningful results.

API spend (Sonnet-4.6, conservative ~25 K prompt + ~10 K output tokens per trial): ~$0.20/trial → ~$300 for the 1,496-trial sweep. Already covered by the existing budget that funded v4 (200 trials → $40).

Scientific payoff: v4 §4's paired analysis had only N=14 tasks, p≈0.14, and bootstrap CIs crossing zero. That should be reported as low power: non-significance does **not** prove "skills have no effect." Moving the paired non-parametric test to roughly N=68 evaluable new tasks is the main scientific lever in this expansion. v7 may either flip the conclusion to a statistically supported skills benefit, or give a much more credible negative result.

---

## 4. Implementation plan

### 4.1 Filter cleanup (prerequisite for Bucket B)

In `testsets/exec_eval_prefill/run_trial.py`:

```python
_HEAVY_DOCKERFILE_MARKERS = re.compile(
    r"^\s*(?:FROM|RUN)\s+.*\b(playwright|chromium|nodejs|npm|texlive|cuda)\b",
    re.IGNORECASE | re.MULTILINE,
)

_HOST_PROVISIONED_TASKS = frozenset({
    # Bucket B promoted after host provisioning lands
})

def is_lightweight(task_dir: Path) -> bool:
    df = task_dir / "environment" / "Dockerfile"
    if not df.exists():
        return False  # Bucket D — still skip
    if task_dir.name in _HOST_PROVISIONED_TASKS:
        return True
    return not _HEAVY_DOCKERFILE_MARKERS.search(df.read_text())
```

Drop `docker`, `gcc`, `build-essential`, `poppler` from the marker set entirely — they're noise. Line-anchor on `FROM`/`RUN` so comments stop triggering matches. Unit test asserts the 30 Bucket-B tasks pass and Bucket-C tasks still fail.

### 4.2 Provisioning script

`scripts/provision_eval_host_b.sh` covers the Bucket-B union:

```bash
sudo apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    poppler-utils ffmpeg libgomp1 \
    openjdk-17-jdk maven                     # spring-boot-jakarta-migration
# Lean4 via elan (auto-managed installer)
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
python3 -m pip install --upgrade pip
python3 -m pip install \
    rdkit-pypi scikit-learn jax[cpu] pint dowhy lifelines \
    obspy pyannote.audio edge-tts \
    gwpy "lalsuite==7.21" \
    pdfplumber pypdf openpyxl pandas numpy scipy \
    "ortools>=9" unified-planning qutip \
    pymatgen sympy
```

Pin versions to the Dockerfiles where they pin (otherwise we lose comparability). Record `pip freeze` to `results/eval_env_b.lock` for v6 provenance.

Special cases:
- `lean4-proof`, `spring-boot-jakarta-migration`, `syzkaller-ppdev-syzlang`: host install of the toolchain works but is heavy; if smoke shows install failure, demote these to container path (Bucket C-like).
- `video-tutorial-indexer`: confirm whether `nodejs` is build-time only or runtime — if runtime, demote.
- `gravitational-wave-detection`: LALSuite is large; check disk before committing.

### 4.3 Container path (Bucket C — reuse for v5+)

Use the same scaffold the prior design described (per-task image, content-hashed tag, `docker run --network=none --rm -v workspace:/root`). Skill prefill stays in the system prompt; the Dockerfile's `COPY skills /root/.claude/skills` is harmless background noise because the agent reads the prefill from `--system-prompt`.

Bucket C image sizes are ~600 MB–1.2 GB each (chromium dominates). Build once, cache in a local registry.

### 4.4 Stratified sampling vs full sweep

Two run modes:

- **`run_prefill_v6_smoke.sh`** — pick one task per SkillsBench category from Bucket A only. ~10 tasks × 12 configs × 1 seed = 120 trials, ~90 min. Validates pipeline end-to-end before committing to the full sweep.
- **`run_prefill_v6_full.sh`** — Bucket A ∪ provisioned Bucket B, all configs, seed=0. The 1,496-trial sweep from §3.

Smoke first, full after, container third.

### 4.5 Patch verification per task

`patch_tests()` and `_mirror_dockerfile_copies()` are the two host-mode rewrites that could fail silently. For each Bucket-A/B task, run an offline dry-run:

```bash
python -m exec_eval_prefill.run_trial --dataset sb --tasks <jsonl> \
  --task-id <id> --pool-sizes 5 --noise-modes noskill --seeds 0 \
  --dry-run-workspace   # new flag: build the workspace, print tree, exit before agent invocation
```

Add `--dry-run-workspace` so we can validate fixture mapping without paying for a full trial. Catches missing globs (e.g. `COPY data/* /root/data/` where `data/` has nested subdirs).

---

## 5. Phased rollout

| Phase | Scope | Artifact | Gating verification |
|-------|-------|----------|---------------------|
| **P0** | Tighten `_HEAVY_DOCKERFILE_MARKERS` + unit tests | PR #1 | `pytest test_is_lightweight.py` green; Bucket A/B/C classification reproducible |
| **P1 / v5** | Bucket A run (31 tasks) | v5 report | smoke-10 first; cumulative coverage becomes 45 tasks (`14 v4 + 31`) |
| **P2 / v6** | Bucket B provisioning + run (30 tasks) | v6 report | smoke 5/30 first; capture `results/eval_env_b.lock`; cumulative coverage becomes 75 tasks |
| **P3 / v7** | Bucket C container path + final large-N paired analysis | v7 report | container smoke 1 task, then full; cumulative coverage becomes 82 tasks and v4 §4 is recomputed at large N |

Each phase outputs its own JSONL append (`sb_exec.jsonl`) so we can re-aggregate at any cumulative cut. Prefill N=5 across all 68+ new evaluable tasks remains optional follow-up if v7 still needs tighter variance estimates; it is not part of the v5→v7 delivery.

PDF-Crosspage stays excluded across all phases; flag upstream.

---

## 6. Report changes

`generate_20t_report_v5.py` reads from `sb_exec.jsonl` and currently bakes the 20-task assumption into:

- Section 1 (Data Summary): hard-coded `"168 configs"`, `"14 tasks"`.
- Section 3 (Per-Task Results): table sized for ~14 rows.
- Section 4 (Paired Analysis): bootstrap CI computed from 14 paired points.

Updates:

- Replace hard-coded counts with `len(df.task_id.unique())` and friends.
- Per-task table paginates (one page per ~20 rows) — at 68 tasks that's 4 pages.
- Section 4 grows in power: paired test on N≈68 instead of N=14 may produce a real p < 0.05. The narrative needs to drop "non-significant due to low N" if that happens.
- Section 5 (Stratification): expand ESSENTIAL/REDUNDANT/HARD buckets — at this N the buckets become statistically meaningful.
- Section 8 (Noise & Pool): now backed by ~800 trials instead of 168 — descriptive trends should sharpen.
- Add a new Section 12: **Coverage**, listing which tasks each version covers (v4: 14, v5: +31 → 45, v6: +30 → 75, v7: +7 → 82, drop PDF-Crosspage and the 6 v4-skipped).

---

## 7. Risk register

| risk | likelihood | mitigation |
|------|-----------|------------|
| Bucket B host-install fails on a few tasks despite passing regex | medium | smoke each Bucket-B task individually before mass-promoting; demote stubborn ones to container |
| Cost overrun (Sonnet token spend) | low | $300 estimate already > 4× v4 ($40) but well within precedent; cap concurrency to control burst |
| Rate-limit collisions on Anvil-style shared hosts | medium | the existing `rl_pat` detector catches `hit your limit` / `resets HH:MM`; keep concurrency ≤ 4 |
| `lean4-proof` or `spring-boot-jakarta-migration` host install is too invasive | medium | demote to Bucket C and run in container; this is contemplated in §4.2 |
| New paired analysis still p > 0.05 even at N=68 | medium | scientifically valid finding — the eval gains negative-result credibility, narrative changes but report ships |
| Some Bucket-A tasks turn out to need fixtures not yet rewritten by `_mirror_dockerfile_copies()` | medium | `--dry-run-workspace` flag catches these before paying for trials |
| Aggregation script chokes on 1,500+ JSONL rows | low | already uses iterative JSONL read; check that pivot tables don't OOM |
| New category (e.g. `Data Visualization` from Bucket C) breaks the existing error taxonomy (TIMEOUT/TEST_FAIL/UNKNOWN) | low | add `INFRA_FAIL` bucket (browser / container errors) before P3 |

---

## 8. What we are explicitly NOT doing here

- Not running the 6 v4-skipped tasks (`court-form-filling`, `crystallographic-wyckoff-position-analysis`, `data-to-d3`, `dynamic-object-aware-egomotion`, `edit-pdf`, `exoplanet-detection-period`). Per direction.
- Not building Phase-B factorial controls (emptyframe / noiseonly). Orthogonal initiative.
- Not migrating the working 14 v4 tasks to a different execution path.
- Not changing the prompt assembler, skill pool builder, noise-mode logic, or rate-limit detector.
- Not touching TerminalBench.

---

## 9. Definition of done

- [ ] Regex tightened; `test_is_lightweight.py` green; Bucket A/B/C classification reproducible from a single helper.
- [ ] `provision_eval_host_b.sh` runs idempotently; `results/eval_env_b.lock` captured.
- [ ] `--dry-run-workspace` flag exists and is documented.
- [ ] Smoke phase (10 Bucket-A tasks × 12 configs × seed 0) produces ≥ 90% non-error trials.
- [ ] v5 report regenerated covering Bucket A (45 tasks total cumulative with v4's 14).
- [ ] v6 report covering Bucket B (75 tasks cumulative).
- [ ] v7 covers Bucket C (82 tasks cumulative).
- [ ] Per-task GT slug coverage audited (none `None` in the new JSONL rows).
- [ ] PDF-Crosspage filed as upstream issue and excluded from all SB grids in `skillsbench_tasks.jsonl` consumers.
