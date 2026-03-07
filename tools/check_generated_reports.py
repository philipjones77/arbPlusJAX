from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    _run([PYTHON, "tools/function_provenance_report.py"])
    _run([PYTHON, "-m", "pytest", "-q", "tests/test_function_provenance_reports.py"])


if __name__ == "__main__":
    main()
