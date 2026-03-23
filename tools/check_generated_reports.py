from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    _run([PYTHON, "tools/generate_docs_indexes.py"])
    _run([PYTHON, "tools/generate_example_notebooks.py"])
    _run([PYTHON, "tools/function_provenance_report.py"])
    _run([PYTHON, "tools/hypgeom_status_report.py"])
    _run([PYTHON, "benchmarks/matrix_surface_workbook.py", "--n", "4", "--warmup", "0", "--runs", "1", "--steps", "4"])
    _run(
        [
            PYTHON,
            "-m",
            "pytest",
            "-q",
            "tests/test_function_provenance_reports.py",
            "tests/test_docs_indexes.py",
            "tests/test_example_notebook_inventory_contracts.py",
            "tests/test_example_notebook_content_contracts.py",
            "tests/test_matrix_surface_workbook_contracts.py",
        ]
    )


if __name__ == "__main__":
    main()
