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
    _run([PYTHON, "tools/cache_aware_surface_report.py"])
    _run([PYTHON, "tools/comparison_backend_defaults_report.py"])
    _run([PYTHON, "tools/api_cold_path_report.py"])
    _run([PYTHON, "tools/api_first_use_report.py"])
    _run([PYTHON, "tools/entry_script_startup_report.py"])
    _run([PYTHON, "tools/matrix_free_first_use_report.py"])
    _run([PYTHON, "tools/point_fast_jax_category_report.py"])
    _run([PYTHON, "tools/point_fast_jax_function_report.py"])
    _run([PYTHON, "tools/point_fast_jax_verification_report.py"])
    _run([PYTHON, "tools/parameterized_ad_verification_report.py"])
    _run([PYTHON, "tools/point_basic_surface_report.py"])
    _run([PYTHON, "tools/point_basic_function_verification_report.py"])
    _run([PYTHON, "tools/repo_standards_verification_report.py"])
    _run([PYTHON, "tools/special_function_status_report.py"])
    _run([PYTHON, "tools/generate_public_metadata_registry.py"])
    _run([PYTHON, "tools/report_status_refresh_inventory.py"])
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
            "tests/test_comparison_backend_defaults.py",
            "tests/test_api_cold_path_report.py",
            "tests/test_api_first_use_report.py",
            "tests/test_entry_script_startup_report.py",
            "tests/test_matrix_free_first_use_report.py",
            "tests/test_cache_aware_surface_inventory.py",
            "tests/test_special_function_status_report.py",
            "tests/test_point_fast_jax_docs_contracts.py",
            "tests/test_point_fast_jax_function_report.py",
            "tests/test_point_fast_jax_verification_report.py",
            "tests/test_parameterized_public_ad_audit.py",
            "tests/test_point_basic_surface_report.py",
            "tests/test_point_basic_function_verification_report.py",
            "tests/test_repo_standards_verification_report.py",
            "tests/test_public_metadata_contracts.py",
            "tests/test_report_status_refresh_inventory.py",
            "tests/test_docs_indexes.py",
            "tests/test_example_notebook_inventory_contracts.py",
            "tests/test_example_notebook_content_contracts.py",
            "tests/test_matrix_surface_workbook_contracts.py",
            "tests/test_repo_link_policy.py",
            "tests/test_platform_bootstrap_contracts.py",
        ]
    )


if __name__ == "__main__":
    main()
