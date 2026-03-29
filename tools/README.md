Last updated: 2026-03-29T00:00:00Z

# Tools

`tools/` is the repo utility layer.

Use this directory for:

- repo maintenance and regeneration scripts
- status and report generators
- runtime/environment/bootstrap helpers
- packaging helpers
- correctness-oriented harness entrypoints

Do not use `tools/` for benchmark implementations or benchmark-specific launch scripts. Those belong under `benchmarks/`.

## Canonical Entry Points
- runtime check:
  - `python tools/check_jax_runtime.py --quick-bench`
- test harness:
  - `python tools/run_test_harness.py --profile chassis --jax-mode cpu`
- validation wrapper:
  - `python tools/run_validation.py --jax-mode cpu`
- refresh generated reports/docs:
  - `python tools/check_generated_reports.py`
- refresh repo-maintained artifacts:
  - `python tools/update_repo_artifacts.py`
- check repo-maintained artifact drift:
  - `python tools/check_repo_update_drift.py`
- regenerate docs indexes:
  - `python tools/generate_docs_indexes.py`
- regenerate tools README:
  - `python tools/generate_tools_readme.py`
- regenerate example notebooks:
  - `python tools/generate_example_notebooks.py`
- execute canonical example notebooks:
  - `python tools/run_example_notebooks.py --jax-mode cpu --jax-dtype float64`
- package source bundle:
  - `python tools/make_zip.py`

## Naming And Consolidation Rules

Canonical rules:
- prefer lower-case, descriptive `verb_object.py` or `subject_report.py` names for new Python tools
- prefer `generate_*` for generators, `check_*` for non-mutating freshness/validation checks, and `run_*` for harness-style execution tools
- keep shell or PowerShell wrappers only when they add platform-specific value over the canonical Python entrypoint
- avoid adding new compatibility wrappers when the canonical tool name can be referenced directly

Current compatibility / consolidation notes:
- `update_docs_indexes.py` has been removed; use `generate_docs_indexes.py` directly.
- `MAKE_ZIP.py` has been removed; use `make_zip.py` directly.

## Repo Maintenance And Generation

- `check_generated_reports.py`
  - Canonical umbrella refresh for generated docs/report surfaces and their freshness tests.
- `update_repo_artifacts.py`
  - Canonical policy-level refresh entrypoint for repo-maintained generated artifacts.
- `check_repo_update_drift.py`
  - Non-mutating freshness checks for repo-maintained generated artifacts.
- `generate_docs_indexes.py`
  - Canonical generator for repo-root and docs subtree indexes.
- `generate_tools_readme.py`
  - Canonical generator for `tools/README.md`.
- `function_provenance_report.py`
  - Generates provenance, implementation, and capability reports.
- `api_surface_structure_report.py`
  - Generates the consolidated public API surface structure report.
- `production_readiness_report.py`
  - Generates the production-readiness governance report.
- `report_status_refresh_inventory.py`
  - Generates the refresh map for `docs/reports/` and `docs/status/`.
- `generate_public_metadata_registry.py`
  - Regenerates the checked-in static public metadata registry.
- `generate_example_notebooks.py`
  - Regenerates managed example notebooks in `examples/`.

## Report Generators

- `api_cold_path_report.py`
  - Generates the package-import and API cold-path inventory.
- `api_first_use_report.py`
  - Generates the representative first-use inventory for point/matrix/tail surfaces.
- `cache_aware_surface_report.py`
  - Generates the cache-aware surface inventory.
- `comparison_backend_defaults_report.py`
  - Generates the comparison-backend defaults report.
- `entry_script_startup_report.py`
  - Generates the entry-script startup inventory.
- `hypgeom_status_report.py`
  - Generates the hypergeometric status report.
- `matrix_free_first_use_report.py`
  - Generates the matrix-free first-use inventory.
- `parameterized_ad_verification_report.py`
  - Generates the parameterized public AD verification ledger.
- `point_basic_surface_report.py`
  - Generates the joined public point/basic family status report.
- `point_basic_function_verification_report.py`
  - Generates the per-function point/basic verification ledger.
- `point_fast_jax_category_report.py`
  - Generates the point-fast JAX category matrix.
- `point_fast_jax_function_report.py`
  - Generates the point-fast JAX function inventory.
- `point_fast_jax_verification_report.py`
  - Generates the point-fast JAX verification report.
- `repo_standards_verification_report.py`
  - Generates the repo standards verification map.
- `special_function_status_report.py`
  - Generates the cross-family special-function status report.
- `slq_logdet_contract_report.py`
  - Generates a JSON contract report for matrix-free SLQ logdet behavior.

## Runtime, Environment, And Portability Helpers

- `check_jax_runtime.py`
  - Prints runtime/device information and can run a small JIT benchmark.
- `python_resolver.py`
  - Resolves the preferred Python interpreter and JAX platform environment.
- `runtime_manifest.py`
  - Shared runtime manifest collection/writing utilities for harnesses and reports.
- `source_tree_bootstrap.py`
  - Adds repo root and `src/` to `sys.path` from a script anchor path.
- `colab_bootstrap.sh`
  - Bootstraps a Colab environment from the repo's Colab requirements.
- `setup_linux_gpu.sh`
  - Local Linux/WSL bootstrap helper for editable installs and optional GPU JAX setup.

## Validation, Execution, And Notebook Harnesses

- `run_test_harness.py`
  - Canonical correctness-oriented pytest harness with named profiles.
- `run_validation.py`
  - Higher-level validation wrapper around tests and benchmark smoke/profile work.
- `run_validation.sh`
  - Shell wrapper for `run_validation.py`.
- `run_validation.ps1`
  - PowerShell wrapper for `run_validation.py`.
- `run_example_notebooks.py`
  - Executes canonical example notebooks and retains runtime/summary artifacts.
- `run_notebook_sweeps.py`
  - Runs larger example/notebook-oriented sweep jobs from the command line.
- `audit_coverage.py`
  - Audits implementation/test coverage against reference surfaces.

## Legacy Or Specialized Helpers

- `core_status_report.py`
  - Legacy core-function status generator; retained while downstream tests/report flows still depend on it.
- `point_status_report.py`
  - Legacy point-wrapper availability status generator; retained while dependent status tools still use it.
- `custom_core_report.py`
  - Curated custom-core status report layered on top of legacy core/point status helpers.

## Ownership Boundary

- `tools/` owns repo utilities and maintenance helpers.
- `benchmarks/` owns benchmark implementations, benchmark comparisons, and benchmark-facing launchers.
- `tests/` owns pytest conformance/regression coverage.
- `examples/` owns runnable demonstration notebooks and scripts.

Related standards and governance:
- [docs/standards/repo_standards.md](/docs/standards/repo_standards.md)
- [docs/standards/implementation_docs_standard.md](/docs/standards/implementation_docs_standard.md)
- [docs/standards/production_readiness_standard.md](/docs/standards/production_readiness_standard.md)
- [docs/objects/update_artifacts.md](/docs/objects/update_artifacts.md)
