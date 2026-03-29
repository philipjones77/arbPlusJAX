from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "reports" / "repo_standards_verification.md"


ROWS = (
    {
        "area": "runtime/api",
        "standards": (
            "docs/standards/jax_api_runtime_standard.md",
            "docs/standards/implicit_adjoint_operator_solve_standard.md",
        ),
        "reports": (
            "docs/reports/function_capability_registry.json",
            "docs/reports/parameterized_ad_verification.md",
            "docs/reports/point_basic_surface_status.md",
            "docs/reports/point_basic_function_verification.md",
        ),
        "tests": (
            "tests/test_function_provenance_reports.py",
            "tests/test_parameterized_family_ad_directions.py",
            "tests/test_parameterized_public_ad_audit.py",
            "tests/test_special_function_ad_directions.py",
            "tests/test_point_basic_function_verification_report.py",
        ),
    },
    {
        "area": "caching/recompilation",
        "standards": ("docs/standards/caching_recompilation_standard.md",),
        "reports": (
            "docs/reports/cache_aware_surface_inventory.md",
            "docs/reports/api_first_use_inventory.md",
            "docs/reports/matrix_free_first_use_inventory.md",
        ),
        "tests": (
            "tests/test_cache_aware_surface_inventory.py",
            "tests/test_api_first_use_report.py",
            "tests/test_matrix_free_first_use_report.py",
        ),
    },
    {
        "area": "startup/import/compile",
        "standards": (
            "docs/standards/startup_compile_standard.md",
            "docs/standards/startup_import_boundary_standard.md",
            "docs/standards/lazy_loading_standard.md",
        ),
        "reports": (
            "docs/reports/api_cold_path_inventory.md",
            "docs/reports/entry_script_startup_inventory.md",
            "docs/reports/point_fast_jax_verification.md",
        ),
        "tests": (
            "tests/test_api_cold_path_report.py",
            "tests/test_entry_script_startup_report.py",
            "tests/test_startup_compile_docs_contracts.py",
            "tests/test_point_fast_jax_categories.py",
            "tests/test_point_fast_jax_verification_report.py",
        ),
    },
    {
        "area": "point-only fast jax",
        "standards": (
            "docs/standards/operational_jax_standard.md",
            "docs/standards/fast_jax_standard.md",
        ),
        "reports": (
            "docs/reports/point_fast_jax_function_inventory.md",
            "docs/reports/point_fast_jax_category_matrix.md",
            "docs/reports/point_fast_jax_verification.md",
        ),
        "tests": (
            "tests/test_point_fast_jax_function_report.py",
            "tests/test_point_fast_jax_docs_contracts.py",
            "tests/test_point_fast_jax_categories.py",
            "tests/test_point_fast_jax_verification_report.py",
        ),
    },
    {
        "area": "release/process/bootstrap",
        "standards": (
            "docs/standards/release_governance_standard.md",
            "docs/standards/environment_portability_standard.md",
        ),
        "reports": ("docs/reports/report_status_refresh_inventory.md",),
        "tests": (
            "tests/test_release_execution_checklist_contracts.py",
            "tests/test_platform_bootstrap_contracts.py",
            "tests/test_report_status_refresh_inventory.py",
            "tests/test_docs_indexes.py",
        ),
    },
)


def _exists(paths: tuple[str, ...]) -> bool:
    return all((REPO_ROOT / path).exists() for path in paths)


def render() -> str:
    lines = [
        "Last updated: 2026-03-27T00:00:00Z",
        "",
        "# Repo Standards Verification",
        "",
        "This report records the repo-local verification surface for the runtime, caching, startup/compile, point-fast, and process standards that currently govern arbPlusJAX.",
        "",
        "It is a verification map, not a substitute for actually running the tests.",
        "",
        "| verification area | standards | generated inventories / reports | owning tests | artifact status |",
        "|---|---|---|---|---|",
    ]
    for row in ROWS:
        lines.append(
            "| "
            + f"`{row['area']}` | "
            + ", ".join(f"[{Path(path).name}](/{path})" for path in row["standards"])
            + " | "
            + ", ".join(f"[{Path(path).name}](/{path})" for path in row["reports"])
            + " | "
            + ", ".join(f"[{Path(path).name}](/{path})" for path in row["tests"])
            + " | "
            + f"`{'present' if _exists(row['standards']) and _exists(row['reports']) and _exists(row['tests']) else 'missing_artifact'}`"
            + " |"
        )
    lines.extend(
        [
            "",
            "## Current Scope",
            "",
            "- Runtime/API verification is carried by metadata/provenance tests and the AD-direction proof slices.",
            "- Cache/recompile verification is carried by the cache-aware inventory plus first-use budget reports.",
            "- Startup/import/compile verification is carried by cold-path inventories, entry-script startup inventory, and the startup-compile docs contracts.",
            "- Point-only fast-JAX verification is carried separately from the broader point/basic family ledger.",
            "- Release/process/bootstrap verification is carried by the release checklist, portability/bootstrap contracts, and report/index freshness tests.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
