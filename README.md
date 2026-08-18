# ProcMem2Skills

## Repository Status

This repository is prepared for GitHub push with a code-first layout:
- Active code and entrypoints stay in `src/`, `scripts/`, `tests/`, `docs/`.
- Historical code and legacy experiment outputs are archived under `history/`.
- Runtime output root `experiments/` is kept as an empty writable directory (`.gitkeep`) for new runs.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Short server script entrypoints:
- Controlled study: `python scripts/server/study/controlled/run.py --help`
- Workflow induction: `python scripts/server/study/induce/workflows.py --help`
- Formal experiment: `python scripts/server/study/formal/run.py --help`
- SkillsBench live: `python scripts/server/study/live/skills.py --help`
- Terminal-Bench live: `python scripts/server/study/live/terminal.py --help`
- Transfer study: `python scripts/server/study/transfer/run.py --help`

## Project Website

The `website` branch contains a static Nerfies-inspired project page for
*Demystifying Agent Skills: Why They Work - Until They Don't*. The page is
plain HTML, CSS, and JavaScript so it can be served directly by GitHub Pages
from the repository subpath `/Procmem2Skills/`.

### Local preview

From the repository root:

```bash
python3 -m http.server 8000
```

Open <http://localhost:8000/>. Do not open `index.html` with `file://`: the
interactive data modules and relative assets are intended to be loaded over
HTTP.

### GitHub Pages

1. Push the `website` branch to GitHub.
2. In repository settings, open **Pages**.
3. Select **Deploy from a branch**, choose `website`, and select `/ (root)`.
4. The project page will be available at
   `https://zhiyuanjiang04.github.io/Procmem2Skills/` after deployment.

### Website data and assets

- `index.html` contains the semantic page structure and paper copy.
- `static/data/content.js` contains authors, narrative text, taxonomy labels,
  and citation metadata.
- `static/data/results.js` contains the confirmed chart values.
- `static/js/site.js` renders the representation and retrieval explorers,
  tooltips, tables, navigation, figure enlargement, and BibTeX copy action.
- `static/css/site.css` contains the responsive Nerfies-inspired visual system.
- `static/images/` contains the figures copied from the paper source directory.
