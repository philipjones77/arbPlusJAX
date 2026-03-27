from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


DRIFT_TESTS = [
    "tests/test_docs_indexes.py",
    "tests/test_report_status_refresh_inventory.py",
    "tests/test_public_metadata_contracts.py",
    "tests/test_repo_update_contracts.py",
]


def main() -> None:
    subprocess.run([PYTHON, "-m", "pytest", "-q", *DRIFT_TESTS], check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
