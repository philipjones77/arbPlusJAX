from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def main() -> None:
    subprocess.run([PYTHON, "tools/check_generated_reports.py"], check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
