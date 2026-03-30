# ProcMem2Skills

English: see [README.en.md](./README.en.md)  
中文：见 [README.zh-CN.md](./README.zh-CN.md)

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

For complete documentation, use the language-specific README files above.

## Open Source

- [LICENSE](./LICENSE)
- [Contributing Guide](./CONTRIBUTING.md)
- [Code of Conduct](./CODE_OF_CONDUCT.md)
- [Security Policy](./SECURITY.md)
