# Remaining 6 Skipped Tasks — Design Document

**Context:** `20t_report_v4.pdf` reports 14 tasks with all 3 conditions (Prefill / Noskill / GT-only). The other 6 of the planned 20 were filtered out by `is_lightweight()` in `testsets/exec_eval_prefill/run_trial.py:49` and emitted `"reason": "heavy Dockerfile (host execution unsafe)"`. This document describes (a) why each was skipped, (b) the strategy to bring each into the eval grid, and (c) the implementation work needed before kicking off a v5 run that covers all 20 tasks.

**Author scope:** unblocking the SkillsBench prefill-context execution eval, not the TerminalBench branch (still container-only).

---

## 1. Inventory of the 6 skipped tasks

Filter regex (`testsets/exec_eval_prefill/run_trial.py:43`):
```
(playwright|nodejs|npm|texlive|poppler|gcc|build-essential|cuda|chromium|docker)
```
Applied to the task's `environment/Dockerfile` text as a substring search.

| # | task_id | category | difficulty | trigger token | real heaviness |
|---|---------|----------|------------|---------------|----------------|
| 1 | `court-form-filling` | document-processing | easy | `poppler` | **light-medium** — only `poppler-utils` (CLI) + 5 pip packages |
| 2 | `crystallographic-wyckoff-position-analysis` | materials_science | medium | `build-essential`, `Docker` | **medium** — `pymatgen` wheels need toolchain only at install time |
| 3 | `data-to-d3` | data-visualization | medium | `nodejs npm playwright chromium` | **heavy** — Chromium download (~150 MB), required by tests |
| 4 | `dynamic-object-aware-egomotion` | video-analysis | medium | `Docker` (comment) | **light** — `ffmpeg` + `opencv-python-headless`, regex false-positive |
| 5 | `edit-pdf` | document | medium | `poppler` | **medium** — `poppler-utils` + `tesseract-ocr` |
| 6 | `exoplanet-detection-period` | astronomy | medium | `gcc build-essential` | **heavy at build** — `batman-package`, `transitleastsquares`, `numba`, `astropy` |

Notable: `dynamic-object-aware-egomotion` is matched only because the literal word "Docker" appears in a code comment (`# ... not during Docker build`). It is a **false positive** of the filter. `crystallographic-wyckoff-position-analysis` matches both `Docker` (comment) and `build-essential` (genuine).

---

## 2. Why fixing this matters for the eval

Two reasons:

1. **Stratification gap.** Section 5.3 of v4 (Hard Tasks split by root cause) is built only from the 14-task subset. The 6 missing tasks span four under-represented categories — document/PDF (×2), materials science, astronomy, web/visualization, video. Their inclusion will likely move both the KNOWLEDGE GAP and CAPABILITY LIMIT buckets, and is the only way to get fair domain coverage for the prefill-vs-baseline claim.
2. **Phase B factorial.** `run_phase_b.sh` orchestrates the 2×2 emptyframe/noiseonly N=5 controls from v4 §11. Running it on the same 14 tasks repeats the v4 selection bias. To make Phase B's variance estimates land on the same population the paper describes ("20 SB tasks, prefill grid"), the 6 missing tasks should be in scope before Phase B re-launches at full scale.

---

## 3. Strategies considered

### S1. Docker-per-trial execution (proper isolation)

Build each task's Docker image once via `environment/Dockerfile`, then launch a container per trial. The skills are injected as a system prompt (not into `/root/.claude/skills`), so the agent still sees the prefill-context skill block as v4 does — only the agent's bash work happens inside the container.

- **Pros:** Matches the task author's intended environment. Removes the regex filter entirely. Future-proofs TB (the README already notes TB needs containers).
- **Cons:** Largest engineering lift. Need a wrapper around `claude -p` that bind-mounts a workspace, runs the agent in container, and harvests `pytest` output. Image builds are 1–10 min each on cold cache; cumulative cost meaningful at 20 tasks × ~5 GB images.
- **Existing scaffold:** `testsets/exec_eval/orchestrate.py` + `run_exec_eval.sh` were started for exactly this. Need to confirm they cover the prefill prompt assembler path, not just the original `exec_eval` flow.

### S2. Host execution with extended dependency layer

Keep `run_trial.py`'s host-execution model. Pre-install the union of needed system + Python packages on the eval host once, and either drop the filter or narrow it to truly incompatible tasks. The `_mirror_dockerfile_copies()` and `_PATH_ROOTS` rewrites already handle filesystem layout.

- **Pros:** Smallest delta to working pipeline. Reuses every existing knob (`--concurrency`, `--resume`, `--max-trials`, prompt assembler).
- **Cons:** Host pollution. Some deps (Chromium, Tesseract) are big and OS-specific. `data-to-d3` *cannot* be host-executed safely because its test suite spawns a browser to load `index.html` — running headed Chromium from inside the eval pipeline is awkward and crashes on shared boxes.
- **Compatible with:** tasks 1, 2, 4, 5, 6.
- **Incompatible with:** task 3 (`data-to-d3`).

### S3. Hybrid: fix the filter + S2 for 5 tasks, S1 just for `data-to-d3`

Combines them. Land the filter bug-fix for false positives (task 4), broaden the host environment to cover tasks 1, 2, 5, 6, and isolate the browser-dependent task 3 into a container path.

- **Pros:** Smallest unblock for 5 of 6 tasks; isolates the one truly hostile case. No catch-all rewrite of `run_trial.py`.
- **Cons:** Two execution paths to maintain. Branching logic in the runner.

### Recommendation: **S3**

S1 is the "correct" long-term answer but is overkill for the immediate goal of completing v5 on the same 20 tasks. S2 alone fails on `data-to-d3`. S3 ships the next report fastest while planting the container scaffold for TB.

---

## 4. Implementation design (S3)

### 4.1 Filter cleanup

In `testsets/exec_eval_prefill/run_trial.py`:

- Tighten the regex so it inspects only `RUN` and `FROM` lines, not the whole file (drops the "Dockerfile in a comment" false positive).
- Drop `docker` from the marker set entirely — no task legitimately apt-installs Docker; the token only appears in comments.
- Add a per-task allowlist override hook, e.g. `_TASK_HOST_EXEC_OVERRIDES = {"court-form-filling", "crystallographic-wyckoff-position-analysis", "dynamic-object-aware-egomotion", "edit-pdf", "exoplanet-detection-period"}`. `is_lightweight()` returns `True` if the task is in the allowlist regardless of regex.
- Keep the regex as the default; only override what the host has been provisioned for.

Concrete shape:

```python
_HEAVY_DOCKERFILE_MARKERS = re.compile(
    r"^\s*(?:FROM|RUN)\s+.*\b(playwright|chromium|nodejs|npm|texlive|cuda)\b",
    re.IGNORECASE | re.MULTILINE,
)
_HOST_PROVISIONED_TASKS = frozenset({
    "court-form-filling", "crystallographic-wyckoff-position-analysis",
    "dynamic-object-aware-egomotion", "edit-pdf", "exoplanet-detection-period",
})

def is_lightweight(task_dir: Path) -> bool:
    if task_dir.name in _HOST_PROVISIONED_TASKS:
        return True
    df = task_dir / "environment" / "Dockerfile"
    return df.exists() and not _HEAVY_DOCKERFILE_MARKERS.search(df.read_text())
```

This removes `gcc`/`build-essential`/`poppler` from the marker list because the host will satisfy them. They were noisy proxies anyway — `gcc` matches `gcc++`, `g++`, `gccgo`, and various comments.

### 4.2 Host environment provisioning

Add `scripts/provision_eval_host.sh` (idempotent; safe to re-run):

```bash
#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    poppler-utils tesseract-ocr tesseract-ocr-eng \
    ffmpeg libgomp1
# Python deps — install into the eval venv, not system.
python3 -m pip install --upgrade pip
python3 -m pip install \
    "pypdf==5.1.0" "fillpdf==0.7.3" "pdfrw==0.4" "PyPDF2==3.0.1" "reportlab==4.2.5" \
    "PyMuPDF==1.24.10" "Pillow==10.4.0" \
    "pymatgen==2025.10.7" "sympy==1.14.0" \
    "opencv-python-headless==4.12.0.88" "numpy==1.26.4" \
    "astropy==6.0.1" "lightkurve==2.4.2" "transitleastsquares==1.32" \
    "batman-package==2.5.2" "numba==0.60.0" "llvmlite==0.43.0"
```

Pin to the same versions as the Dockerfiles to keep behavior comparable. Run once on the eval box before launching v5.

### 4.3 Container path for `data-to-d3`

Extend the runner with a `container_required` branch.

- Add a frozen set `_CONTAINER_REQUIRED = {"data-to-d3"}`.
- In `run_one_trial()`, after `is_lightweight()` succeeds, check container-required; if so, dispatch to `_run_trial_in_container()` instead of the host path.
- `_run_trial_in_container()`:
  - Builds image `skillsbench-{task_id}:eval` from `task_dir/environment/Dockerfile` if not cached. Image tag is content-hashed on the Dockerfile for idempotency.
  - Creates a host workspace (same `setup_workspace` / `patch_tests` flow).
  - Runs the agent with `docker run --rm --network=none -v $workspace:/root -v $skills_prompt_file:/tmp/system_prompt.md -e ANTHROPIC_API_KEY image ...`. The agent inside the container is launched with the same `claude -p --system-prompt @/tmp/system_prompt.md` we use on host.
  - Captures stdout/stderr/return code, then runs `pytest` inside the same container.
  - Cleans up via `--rm`.
- Skill **prefill content** is the same as host-mode — assembled by `prompt_assembler`, never written to `~/.claude/skills`. The `COPY skills /root/.claude/skills` lines in the Dockerfile are honored at image build (they install the *task author's* skills); the prefill experiment overrides via system prompt anyway, so the container's `~/.claude/skills` is benign noise.

### 4.4 Tests-side path patching

`patch_tests()` already handles `/root/...` rewrites for the host path. For the container branch nothing is patched — tests are run at `/root` exactly as the Dockerfile intends, so this is simpler, not harder.

Existing host-path patches need a sanity pass for the 5 host-provisioned tasks:

- `court-form-filling`: expects `/root/sc100-blank.pdf` and writes `/root/sc100-filled.pdf` → covered by current `_PATH_ROOTS`.
- `crystallographic-wyckoff-position-analysis`: input glob `*.cif` to `/root/cif_files/`, writes `/root/workspace/solution.py`. `_mirror_dockerfile_copies()` should handle the `COPY *.cif /root/cif_files/` line — confirm the glob path resolution works for the multi-cif case.
- `dynamic-object-aware-egomotion`: `/root/input.mp4` (single file COPY) and writes `/root/pred_instructions.json`, `/root/pred_dyn_masks.npz` — straightforward.
- `edit-pdf`: `/root/input/input.pdf`, `/root/input/input.txt`, output `/root/output/output.pdf`. Multi-level COPY mirror needed — confirm `_mirror_dockerfile_copies()` creates `input/` and `output/` subdirs.
- `exoplanet-detection-period`: `/root/data/tess_lc.txt`, output `/root/period.txt` — straightforward.

If `_mirror_dockerfile_copies()` doesn't recurse into globs, add a minimal fix instead of escalating to container path.

### 4.5 Runner-level concurrency considerations

- `data-to-d3` container path serializes Docker build/run; cap its concurrency at 1 to avoid Docker daemon contention. The existing per-task semaphore in `run_exec_prefill_sonnet46.sh` is configurable; document a separate `CONCURRENCY_CONTAINER=1` override.
- Exoplanet's first run will spend ~3 min in `numba`/`batman` JIT warmup. Mark this in the run script's preamble and bump `--max-trials` retry budget.

---

## 5. Per-task notes for the eval team

| task | GT skill(s) | likely failure mode after fix | hypothesis worth logging |
|------|-------------|-------------------------------|--------------------------|
| `court-form-filling` | unknown — need to read task.yaml GT | PDF form field discovery vs naive overlay | does the skill teach AcroForm field enumeration, or just generic PDF write? |
| `crystallographic-wyckoff-position-analysis` | `pymatgen-wyckoff` (TBD) | rational rounding edge cases; sympy `nsimplify` denominator bound | NS baseline likely 0% — model rarely knows `pymatgen.symmetry.analyzer.SpacegroupAnalyzer` cold |
| `data-to-d3` | `d3-bubble-chart` / `d3-force` (TBD) | Playwright assertion on tooltip/click linkage; bubble overlap | this is a UI-correctness test, not numerical — error taxonomy needs a new bucket |
| `dynamic-object-aware-egomotion` | `optical-flow-egomotion` (TBD) | mask sparsity format compliance more than the motion classification itself | NS baseline already proven on similar OpenCV tasks — expect Skills REDUNDANT |
| `edit-pdf` | `pdf-natural-language-edit` (TBD) | OCR vs text-layer ambiguity; "don't cover original text" constraint | high TIMEOUT risk — model may iterate fonts/positions |
| `exoplanet-detection-period` | `transit-least-squares` (TBD) | period precision to 5 decimal places; detrending of stellar variability | Skills ESSENTIAL candidate — TLS is domain-specific knowledge |

(GT slugs marked TBD should be filled in by reading each task's `task.toml` / `environment/skills/` before kickoff; v4 reports show the GT slug list was already in the JSONL — confirm it's populated for these 6.)

---

## 6. Validation plan

1. **Filter unit test.** Add `testsets/exec_eval_prefill/tests/test_is_lightweight.py` asserting:
   - All 6 currently-skipped task names return `True` (with override).
   - Tasks containing genuine `playwright`/`chromium`/`cuda` in `RUN` lines still return `False` unless allowlisted.
   - Comment-only `# Docker ...` no longer triggers a skip.

2. **Smoke (≤30 min).** Run host path on 5 tasks × 1 noise × pool=5 × seed=0 = 5 trials. Confirm pytest exit + agent stdout sane. Run container path on `data-to-d3` × seed=0 = 1 trial.

3. **Patch coverage.** Visually inspect workspace dirs of one successful and one failed trial per task. Confirm `_mirror_dockerfile_copies()` produced the expected layout (especially `edit-pdf` multi-level `input/`/`output/`).

4. **Full re-run.** Once smoke passes, kick `run_prefill_n5_hard_easy.sh` with the new 20-task list and N=5 seeds. Re-aggregate. v5 report supersedes v4.

5. **Phase B replay.** Re-run `run_phase_b.sh` (emptyframe + noiseonly N=5) on the now-complete 20. The 2×2 factorial finally has matched samples.

---

## 7. Risk register

| risk | likelihood | mitigation |
|------|-----------|------------|
| Host pollution breaks unrelated services | low | install into eval venv, isolate system pkgs to `tesseract`/`ffmpeg`/`poppler-utils` which are inert when idle |
| Container build cache invalidation on every run | medium | tag image by Dockerfile content hash; only rebuild when hash changes |
| Pinned pip versions conflict on host | medium | run provisioning in a fresh venv, not the system one; record `pip freeze` in `results/eval_env.lock` |
| `data-to-d3` Playwright headless flakiness | medium-high | retry budget 3 per seed; if all 3 fail with non-pytest exit, log as `INFRA_FAIL` and exclude from rates |
| `exoplanet-detection-period` numba JIT eats timeout budget | medium | bump `agent_timeout_sec` to 1500s for this task only, or pre-warm numba cache outside the trial timer |
| GT skills missing from `skillsbench_tasks.jsonl` for 6 tasks | unknown | audit JSONL before kickoff; if missing, copy from `skillsbench_repo/tasks/<id>/environment/skills/` |

---

## 8. Out-of-scope (deferred)

- TerminalBench container execution. The README still says TB skips with "needs container". S1-style work for TB is a separate phase.
- Switching SkillsBench host path → container path for the *already-working* 14 tasks. Lots of risk for no scientific gain.
- Re-running v1–v3 with the new task set. Only v5 needs the full 20.
- The four "missing conditions" in v4 §11 (Noise-only, GT+Random-text, Empty-prompt-frame, Prefill N=5). Those are tackled by `run_phase_b.sh`, which is orthogonal to this design.

---

## 9. Definition of done

- [ ] `is_lightweight()` regex tightened and allowlist added; unit tests green.
- [ ] `scripts/provision_eval_host.sh` runs idempotently on the eval host.
- [ ] Container branch wired for `data-to-d3` with content-hashed image tag.
- [ ] Smoke run produces non-skipped trials for all 6 task_ids in `sb_exec.jsonl`.
- [ ] v5 full run completes with 20-task coverage; `generate_20t_report_v5.py` emits sections 1–11 reflecting the new data.
- [ ] Phase B re-run uses the same 20-task selector.
