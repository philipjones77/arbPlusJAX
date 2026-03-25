Last updated: 2026-03-25T00:00:00Z

# Report And Status Refresh Inventory

This report records how each checked-in `docs/reports/` and `docs/status/` markdown file should be refreshed.

Policy:
- generated documents should be refreshed by their owning tool before commit/push
- manual-authoritative documents remain checked-in source documents, but their owning refresh path should still be explicit
- `python tools/check_generated_reports.py` is the canonical umbrella refresh path for the generated subset

| path | refresh mode | refresh path |
|---|---|---|
| [docs/reports/README.md](/docs/reports/README.md) | `generated` | `python tools/generate_docs_indexes.py` |
| [docs/reports/alternative_functions.md](/docs/reports/alternative_functions.md) | `generated` | `python tools/function_provenance_report.py` |
| [docs/reports/arb_like_functions.md](/docs/reports/arb_like_functions.md) | `generated` | `python tools/function_provenance_report.py` |
| [docs/reports/benchmark_group_inventory.md](/docs/reports/benchmark_group_inventory.md) | `manual-authoritative` | `update the benchmark taxonomy/inventory doc when benchmark group ownership changes` |
| [docs/reports/cache_aware_surface_inventory.md](/docs/reports/cache_aware_surface_inventory.md) | `generated` | `python tools/cache_aware_surface_report.py` |
| [docs/reports/comparison_backend_defaults.md](/docs/reports/comparison_backend_defaults.md) | `generated` | `python tools/comparison_backend_defaults_report.py` |
| [docs/reports/core_numeric_scalars_status.md](/docs/reports/core_numeric_scalars_status.md) | `manual-authoritative` | `update when scalar hardening or scalar status rollup changes` |
| [docs/reports/cpu_validation_profiles.md](/docs/reports/cpu_validation_profiles.md) | `manual-authoritative` | `rerun CPU validation profiles and update from retained run manifests/results` |
| [docs/reports/current_repo_mapping.md](/docs/reports/current_repo_mapping.md) | `generated` | `python tools/generate_docs_indexes.py` |
| [docs/reports/dense_matrix_engineering_status.md](/docs/reports/dense_matrix_engineering_status.md) | `manual-authoritative` | `update the owning source document directly` |
| [docs/reports/dense_matrix_surface_benchmark.md](/docs/reports/dense_matrix_surface_benchmark.md) | `manual-authoritative` | `update the owning source document directly` |
| [docs/reports/environment_portability_inventory.md](/docs/reports/environment_portability_inventory.md) | `manual-authoritative` | `update when environment targets, bootstrap scripts, or portability support changes` |
| [docs/reports/example_notebook_inventory.md](/docs/reports/example_notebook_inventory.md) | `manual-authoritative` | `regenerate notebooks and refresh inventory when canonical notebook set changes` |
| [docs/reports/function_engineering_status.md](/docs/reports/function_engineering_status.md) | `generated` | `python tools/function_provenance_report.py` |
| [docs/reports/function_implementation_index.md](/docs/reports/function_implementation_index.md) | `generated` | `python tools/function_provenance_report.py` |
| [docs/reports/function_provenance_registry.md](/docs/reports/function_provenance_registry.md) | `generated` | `python tools/function_provenance_report.py` |
| [docs/reports/hypgeom_status.md](/docs/reports/hypgeom_status.md) | `generated` | `python tools/hypgeom_status_report.py` |
| [docs/reports/matrix_free_krylov_benchmark.md](/docs/reports/matrix_free_krylov_benchmark.md) | `manual-authoritative` | `update the owning source document directly` |
| [docs/reports/matrix_surface_workbook.md](/docs/reports/matrix_surface_workbook.md) | `generated` | `python benchmarks/matrix_surface_workbook.py --n 4 --warmup 0 --runs 1 --steps 4` |
| [docs/reports/new_functions.md](/docs/reports/new_functions.md) | `generated` | `python tools/function_provenance_report.py` |
| [docs/reports/point_fast_jax_category_matrix.md](/docs/reports/point_fast_jax_category_matrix.md) | `generated` | `python tools/point_fast_jax_category_report.py` |
| [docs/reports/point_fast_jax_function_inventory.md](/docs/reports/point_fast_jax_function_inventory.md) | `generated` | `python tools/point_fast_jax_function_report.py` |
| [docs/reports/repo_organization_by_coverage_categories.md](/docs/reports/repo_organization_by_coverage_categories.md) | `manual-authoritative` | `update when the repo grouping model or coverage-category map changes` |
| [docs/reports/report_status_refresh_inventory.md](/docs/reports/report_status_refresh_inventory.md) | `generated` | `python tools/report_status_refresh_inventory.py` |
| [docs/status/README.md](/docs/status/README.md) | `generated` | `python tools/generate_docs_indexes.py` |
| [docs/status/audit.md](/docs/status/audit.md) | `manual-authoritative` | `rerun the audit workflow and refresh the checked-in audit snapshot` |
| [docs/status/matrix_free_completion_plan.md](/docs/status/matrix_free_completion_plan.md) | `manual-authoritative` | `update when matrix-free/operator completion planning changes` |
| [docs/status/point_fast_jax_plan.md](/docs/status/point_fast_jax_plan.md) | `manual-authoritative` | `update the owning source document directly` |
| [docs/status/sparse_completion_plan.md](/docs/status/sparse_completion_plan.md) | `manual-authoritative` | `update when sparse/block/vblock completion planning changes` |
| [docs/status/test_coverage_matrix.md](/docs/status/test_coverage_matrix.md) | `manual-authoritative` | `update when the category model or primary test ownership changes` |
| [docs/status/test_gap_checklist.md](/docs/status/test_gap_checklist.md) | `manual-authoritative` | `update when direct-owner test coverage lands or gaps move` |
| [docs/status/todo.md](/docs/status/todo.md) | `manual-authoritative` | `update when implementation status or active backlog changes` |
