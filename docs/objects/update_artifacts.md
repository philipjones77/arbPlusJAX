Last updated: 2026-03-27T00:00:00Z

# Update Artifacts

This object catalog records the repo artifacts that must be refreshed to keep
the checked-in repository state current.

## Canonical Commands

- refresh generated subset:
  - `python tools/update_repo_artifacts.py`
- check for drift without rewriting:
  - `python tools/check_repo_update_drift.py`

## Artifact Inventory

| artifact | class | source of truth | refresh path | drift check |
|---|---|---|---|---|
| [README.md](/README.md) | `generated-authoritative` | filesystem layout and docs index generator | `python tools/generate_docs_indexes.py` | `python -m pytest -q tests/test_docs_indexes.py` |
| [docs/standards/README.md](/docs/standards/README.md) | `generated-authoritative` | filesystem layout and docs index generator | `python tools/generate_docs_indexes.py` | `python -m pytest -q tests/test_docs_indexes.py` |
| [docs/implementation/README.md](/docs/implementation/README.md) | `generated-authoritative` | filesystem layout and docs index generator | `python tools/generate_docs_indexes.py` | `python -m pytest -q tests/test_docs_indexes.py` |
| [docs/objects/README.md](/docs/objects/README.md) | `generated-authoritative` | filesystem layout and docs index generator | `python tools/generate_docs_indexes.py` | `python -m pytest -q tests/test_docs_indexes.py` |
| [docs/reports/report_status_refresh_inventory.md](/docs/reports/report_status_refresh_inventory.md) | `generated-authoritative` | `tools/report_status_refresh_inventory.py` inventory map | `python tools/report_status_refresh_inventory.py` | `python -m pytest -q tests/test_report_status_refresh_inventory.py` |
| [src/arbplusjax/public_metadata_registry.json](/src/arbplusjax/public_metadata_registry.json) | `runtime-critical-generated` | public surface plus metadata builder logic | `python tools/generate_public_metadata_registry.py` | `python -m pytest -q tests/test_public_metadata_contracts.py` |
| [docs/objects/function_catalog.md](/docs/objects/function_catalog.md) | `manual-authoritative-with-refresh-path` | repo-maintained function/object catalog | update when public object catalog structure or ownership changes | reviewed through normal docs/test coverage |
| [docs/reports/point_fast_jax_function_inventory.md](/docs/reports/point_fast_jax_function_inventory.md) | `generated-authoritative` | `tools/point_fast_jax_function_report.py` | `python tools/point_fast_jax_function_report.py` | `python -m pytest -q tests/test_point_fast_jax_function_report.py` |
| [docs/reports/point_fast_jax_category_matrix.md](/docs/reports/point_fast_jax_category_matrix.md) | `generated-authoritative` | `tools/point_fast_jax_category_report.py` | `python tools/point_fast_jax_category_report.py` | `python -m pytest -q tests/test_point_fast_jax_docs_contracts.py` |
| [docs/reports/point_fast_jax_verification.md](/docs/reports/point_fast_jax_verification.md) | `generated-authoritative` | `tools/point_fast_jax_verification_report.py` | `python tools/point_fast_jax_verification_report.py` | `python -m pytest -q tests/test_point_fast_jax_verification_report.py` |
| [docs/reports/parameterized_ad_verification.md](/docs/reports/parameterized_ad_verification.md) | `generated-authoritative` | `tools/parameterized_ad_verification_report.py` | `python tools/parameterized_ad_verification_report.py` | `python -m pytest -q tests/test_parameterized_public_ad_audit.py` |
| [docs/reports/point_basic_surface_status.md](/docs/reports/point_basic_surface_status.md) | `generated-authoritative` | `tools/point_basic_surface_report.py` | `python tools/point_basic_surface_report.py` | `python -m pytest -q tests/test_point_basic_surface_report.py` |
| [docs/reports/point_basic_function_verification.md](/docs/reports/point_basic_function_verification.md) | `generated-authoritative` | `tools/point_basic_function_verification_report.py` | `python tools/point_basic_function_verification_report.py` | `python -m pytest -q tests/test_point_basic_function_verification_report.py` |
| [docs/reports/repo_standards_verification.md](/docs/reports/repo_standards_verification.md) | `generated-authoritative` | `tools/repo_standards_verification_report.py` | `python tools/repo_standards_verification_report.py` | `python -m pytest -q tests/test_repo_standards_verification_report.py` |
| [docs/reports/cache_aware_surface_inventory.md](/docs/reports/cache_aware_surface_inventory.md) | `generated-authoritative` | `tools/cache_aware_surface_report.py` | `python tools/cache_aware_surface_report.py` | `python -m pytest -q tests/test_cache_aware_surface_inventory.py` |
| [docs/reports/function_provenance_registry.md](/docs/reports/function_provenance_registry.md) | `generated-authoritative` | `tools/function_provenance_report.py` | `python tools/function_provenance_report.py` | `python -m pytest -q tests/test_function_provenance_reports.py` |
| [docs/reports/function_capability_registry.json](/docs/reports/function_capability_registry.json) | `generated-authoritative` | `tools/function_provenance_report.py` | `python tools/function_provenance_report.py` | `python -m pytest -q tests/test_function_provenance_reports.py` |
| [docs/reports/hypgeom_status.md](/docs/reports/hypgeom_status.md) | `generated-authoritative` | `tools/hypgeom_status_report.py` | `python tools/hypgeom_status_report.py` | `python -m pytest -q tests/test_hypgeom_status_report.py` |
| startup probe outputs under [benchmarks/results](/benchmarks/results) | `manual-authoritative-with-refresh-path` | owning startup probe script and retained benchmark result policy | rerun the owning `*_startup_probe.py` when startup boundaries or hot-family compile behavior changes | probe-specific contract/tests when present |

## Notes

- The generated refresh inventory in
  [report_status_refresh_inventory.md](/docs/reports/report_status_refresh_inventory.md)
  remains the canonical list for `docs/reports/` and `docs/status/`.
- This document exists at the object layer because it describes the maintained
  artifact set itself, not just the policy around it.
