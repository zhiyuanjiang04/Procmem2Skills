from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


class CliEntrypointTest(unittest.TestCase):
    def test_python_module_entrypoint_shows_help(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "src")
        completed = subprocess.run(
            [sys.executable, "-m", "procmem2skills.cli", "--help"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("Usage:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
