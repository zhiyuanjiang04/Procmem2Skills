#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import runpy


def main() -> int:
    target = Path(__file__).resolve().parents[2] / "run_full_workflow_induction.py"
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
