Last updated: 2026-03-22T00:00:00Z

# Tools

`tools/` is the repo utility layer.

Use this directory for:

- repo maintenance and regeneration scripts
- status and report generators
- runtime/environment/bootstrap helpers
- packaging helpers
- correctness-oriented harness entrypoints

Do not use `tools/` for benchmark implementations or benchmark-specific launch
scripts. Those belong under `benchmarks/`.

## Main Entry Points

- runtime check:
  - `python tools/check_jax_runtime.py --quick-bench`
- test harness:
  - `python tools/run_test_harness.py --profile chassis --jax-mode cpu`
- validation wrapper:
  - `python tools/run_validation.py --jax-mode cpu`
- regenerate reports:
  - `python tools/check_generated_reports.py`
- refresh repo-maintained artifacts:
  - `python tools/update_repo_artifacts.py`
- check repo-maintained artifact drift:
  - `python tools/check_repo_update_drift.py`
- regenerate docs indexes:
  - `python tools/generate_docs_indexes.py`
- regenerate API cold-path inventory:
  - `python tools/api_cold_path_report.py`
- regenerate API first-use inventory:
  - `python tools/api_first_use_report.py`
- regenerate entry-script startup inventory:
  - `python tools/entry_script_startup_report.py`
- regenerate matrix-free first-use inventory:
  - `python tools/matrix_free_first_use_report.py`
- regenerate point/basic family status inventory:
  - `python tools/point_basic_surface_report.py`
- regenerate point/basic per-function verification inventory:
  - `python tools/point_basic_function_verification_report.py`
- regenerate point-only fast-JAX verification inventory:
  - `python tools/point_fast_jax_verification_report.py`
- regenerate repo standards verification map:
  - `python tools/repo_standards_verification_report.py`
- regenerate example notebooks:
  - `python tools/generate_example_notebooks.py`
- regenerate static public metadata registry:
  - `python tools/generate_public_metadata_registry.py`
- package source bundle:
  - `python tools/MAKE_ZIP.py`

## Script Map

### Repo maintenance and generation

- `check_generated_reports.py`
  - Regenerates report-style docs and runs the provenance/report freshness test.
  - Use after changing provenance, capability, or report-generation logic.

- `update_repo_artifacts.py`
  - Canonical policy-level refresh entrypoint for repo-maintained generated artifacts.
  - Use when you want the standard update flow instead of calling the lower-level generators directly.

- `check_repo_update_drift.py`
  - Runs non-mutating freshness checks for repo-maintained generated artifacts.
  - Use in CI or locally when you need to confirm the checked-in update surface is current without rewriting files.

- `function_provenance_report.py`
  - Writes the function provenance and implementation registry reports under `docs/reports/`.
  - Use when public function inventory or provenance metadata changes.

- `api_cold_path_report.py`
  - Writes the current package-import and `api`-import cold-path inventory under `docs/reports/`.
  - Use after changing lazy-loading boundaries, `api` imports, or import-tier budgets.

- `api_first_use_report.py`
  - Writes the representative first-use module inventory for core point, matrix point, and tail surfaces under `docs/reports/`.
  - Use after changing point/tail lazy boundaries or first-use import budgets.

- `entry_script_startup_report.py`
  - Writes the benchmark/example entry-script startup inventory under `docs/reports/`.
  - Use after changing benchmark/example import structure or when separating repo import debt from JAX/runtime startup cost.

- `matrix_free_first_use_report.py`
  - Writes the representative first-use module inventory for matrix-free operator construction, primitive apply, Krylov solve, and implicit-adjoint solve under `docs/reports/`.
  - Use after changing matrix-free runtime boundaries or matrix-free import budgets.

- `point_basic_surface_report.py`
  - Writes the joined public point/basic family status report under `docs/reports/`.
  - Use after changing public metadata, point/basic ownership, canonical tests, benchmarks, or notebooks for the seven public function families and the curvature helper layer.

- `point_basic_function_verification_report.py`
  - Writes the per-function point/basic verification ledger under `docs/reports/`.
  - Use after changing public metadata, direct target coverage, or the mapped benchmark/notebook evidence for point/basic families.

- `point_fast_jax_verification_report.py`
  - Writes the point-only fast-JAX verification report under `docs/reports/`.
  - Use after changing point batch fastpaths, category-owned point-fast tests, or the canonical point-fast benchmark/notebook evidence.

- `repo_standards_verification_report.py`
  - Writes the repo-level standards verification map under `docs/reports/`.
  - Use after changing the owning runtime/cache/startup/release standards, their generated inventories, or the tests that enforce them.

- `hypgeom_status_report.py`
  - Writes the hypergeometric status report under `docs/reports/`.
  - Use when hypgeom mode/kernel status changes.

- `special_function_status_report.py`
  - Writes the cross-family special-function status report under `docs/reports/`.
  - Use when special-function benchmarks, startup probes, canonical notebooks, or owner tests change.

- `core_status_report.py`
  - Generates the canonical core-function implementation status report.

- `point_status_report.py`
  - Generates point-wrapper availability status for core functions.

- `custom_core_report.py`
  - Generates the curated custom-core status report layered on top of the core/point status data.

- `generate_docs_indexes.py`
  - Regenerates the repo-root `README.md`, docs landing pages, and generated section indexes including the implementation subtree indexes.
  - Use after adding/removing docs, reports, status files, or indexed implementation notes.

- `update_docs_indexes.py`
  - Legacy compatibility wrapper around `generate_docs_indexes.py`.
  - Prefer the canonical generator directly in new docs and scripts.

- `generate_example_notebooks.py`
  - Regenerates the managed example notebooks in `examples/`.
  - Use after changing notebook standards, benchmark-profile wiring, or default example content.

- `generate_public_metadata_registry.py`
  - Regenerates the checked-in static public metadata registry used by runtime metadata lookup.
  - Use after changing public metadata shape, public API inventory, or metadata inference logic.

- `run_example_notebooks.py`
  - Executes the canonical example notebooks and retains executed notebooks plus runtime/summary artifacts under the owning `examples/outputs/` folders.
  - Use for CPU/GPU notebook validation runs without opening Jupyter manually.

- `audit_coverage.py`
  - Audits implementation/test coverage against external/reference surfaces.
  - Use for coverage gap analysis rather than normal runtime execution.

### Runtime, environment, and portability helpers

- `check_jax_runtime.py`
  - Prints runtime/device information and can run a small JIT benchmark.
  - Use to confirm CPU/GPU backend selection, `x64`, and environment settings.

- `runtime_manifest.py`
  - Shared runtime manifest collection/writing utilities used by harnesses and benchmark/report flows.
  - This is a support module, not usually run directly.

- `python_resolver.py`
  - Resolves the preferred Python interpreter and JAX platform environment.
  - On Linux, the default interpreter policy is the shared `jax` environment unless a caller explicitly overrides it.
  - This is a support module for harness scripts.

- `source_tree_bootstrap.py`
  - Adds repo root and `src/` to `sys.path` from a script anchor path.
  - Use from repo utilities that need source-tree execution.

- `colab_bootstrap.sh`
  - Bootstraps a Colab environment for this repo from [requirements-colab.txt](/requirements-colab.txt), with optional GPU and comparison-software toggles.
  - Intended to be run inside Colab after cloning the repo.

- `setup_linux_gpu.sh`
  - Local Linux setup/bootstrap helper for editable installs and optional GPU JAX setup.
  - Use on Linux workstations or WSL-style environments where that bootstrap model fits.

### Test and validation harnesses

- `run_test_harness.py`
  - Canonical correctness-oriented pytest harness with named profiles and runtime manifest output.
  - Use for portable test runs across local Linux, WSL, and Colab.

- `run_validation.py`
  - Live-status wrapper that runs tests and optional benchmark smoke/profile work with progress output.
  - Use when you want one higher-level command rather than calling pytest and benchmark runners separately.

- `run_validation.sh`
  - Shell wrapper for `run_validation.py`.

- `run_validation.ps1`
  - PowerShell wrapper for `run_validation.py`.

- `run_notebook_sweeps.py`
  - Runs larger example/notebook-oriented sweep jobs from the command line.
  - Use when notebook-style sweeps need orchestration without opening the notebooks themselves.

### Packaging and contract helpers

- `MAKE_ZIP.py`
  - Creates a constrained source bundle for the repo with naming validation.
  - Use when producing a clean source archive for transfer or release packaging.

- `slq_logdet_contract_report.py`
  - Writes a JSON report for matrix-free `SLQ logdet` contract-style checks.
  - Use when validating or documenting the current `SLQ logdet` contract behavior.

## Ownership Boundary

- `tools/` owns repo utilities and maintenance helpers.
- `benchmarks/` owns benchmark implementations, benchmark comparisons, and benchmark-facing launchers.
- `tests/` owns pytest conformance/regression coverage.
- `examples/` owns runnable demonstration notebooks and scripts.

If a script primarily exists to benchmark, compare software, or emit benchmark
artifacts, it should live in `benchmarks/`, not here.
