from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "reports" / "report_status_refresh_inventory.md"


GENERATED_BY = {
    "docs/reports/README.md": "python tools/generate_docs_indexes.py",
    "docs/reports/cache_aware_surface_inventory.md": "python tools/cache_aware_surface_report.py",
    "docs/reports/current_repo_mapping.md": "python tools/generate_docs_indexes.py",
    "docs/reports/function_provenance_registry.md": "python tools/function_provenance_report.py",
    "docs/reports/function_capability_registry.json": "python tools/function_provenance_report.py",
    "docs/reports/function_implementation_index.md": "python tools/function_provenance_report.py",
    "docs/reports/function_engineering_status.md": "python tools/function_provenance_report.py",
    "docs/reports/arb_like_functions.md": "python tools/function_provenance_report.py",
    "docs/reports/alternative_functions.md": "python tools/function_provenance_report.py",
    "docs/reports/new_functions.md": "python tools/function_provenance_report.py",
    "docs/reports/hypgeom_status.md": "python tools/hypgeom_status_report.py",
    "docs/reports/matrix_surface_workbook.md": "python benchmarks/matrix_surface_workbook.py --n 4 --warmup 0 --runs 1 --steps 4",
    "docs/reports/comparison_backend_defaults.md": "python tools/comparison_backend_defaults_report.py",
    "docs/reports/point_fast_jax_category_matrix.md": "python tools/point_fast_jax_category_report.py",
    "docs/reports/point_fast_jax_function_inventory.md": "python tools/point_fast_jax_function_report.py",
    "docs/reports/report_status_refresh_inventory.md": "python tools/report_status_refresh_inventory.py",
    "docs/status/README.md": "python tools/generate_docs_indexes.py",
}

MANUAL_REFRESH = {
    "docs/reports/benchmark_group_inventory.md": "update the benchmark taxonomy/inventory doc when benchmark group ownership changes",
    "docs/reports/core_numeric_scalars_status.md": "update when scalar hardening or scalar status rollup changes",
    "docs/reports/cpu_validation_profiles.md": "rerun CPU validation profiles and update from retained run manifests/results",
    "docs/reports/environment_portability_inventory.md": "update when environment targets, bootstrap scripts, or portability support changes",
    "docs/reports/example_notebook_inventory.md": "regenerate notebooks and refresh inventory when canonical notebook set changes",
    "docs/reports/repo_organization_by_coverage_categories.md": "update when the repo grouping model or coverage-category map changes",
    "docs/status/audit.md": "rerun the audit workflow and refresh the checked-in audit snapshot",
    "docs/status/matrix_free_completion_plan.md": "update when matrix-free/operator completion planning changes",
    "docs/status/sparse_completion_plan.md": "update when sparse/block/vblock completion planning changes",
    "docs/status/test_coverage_matrix.md": "update when the category model or primary test ownership changes",
    "docs/status/test_gap_checklist.md": "update when direct-owner test coverage lands or gaps move",
    "docs/status/todo.md": "update when implementation status or active backlog changes",
}


def _status_rows() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for folder in ("docs/reports", "docs/status"):
        for path in sorted((REPO_ROOT / folder).glob("*.md")):
            repo_path = path.relative_to(REPO_ROOT).as_posix()
            if repo_path in GENERATED_BY:
                rows.append((repo_path, "generated", GENERATED_BY[repo_path]))
            else:
                rows.append((repo_path, "manual-authoritative", MANUAL_REFRESH.get(repo_path, "update the owning source document directly")))
    return rows


def render() -> str:
    lines = [
        "Last updated: 2026-03-25T00:00:00Z",
        "",
        "# Report And Status Refresh Inventory",
        "",
        "This report records how each checked-in `docs/reports/` and `docs/status/` markdown file should be refreshed.",
        "",
        "Policy:",
        "- generated documents should be refreshed by their owning tool before commit/push",
        "- manual-authoritative documents remain checked-in source documents, but their owning refresh path should still be explicit",
        "- `python tools/check_generated_reports.py` is the canonical umbrella refresh path for the generated subset",
        "",
        "| path | refresh mode | refresh path |",
        "|---|---|---|",
    ]
    for repo_path, mode, refresh in _status_rows():
        lines.append(f"| [{repo_path}](/{repo_path}) | `{mode}` | `{refresh}` |")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
