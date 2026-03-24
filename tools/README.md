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
- regenerate docs indexes:
  - `python tools/generate_docs_indexes.py`
- regenerate example notebooks:
  - `python tools/generate_example_notebooks.py`
- package source bundle:
  - `python tools/package_repo.py`

## Script Map

### Repo maintenance and generation

- `check_generated_reports.py`
  - Regenerates report-style docs and runs the provenance/report freshness test.
  - Use after changing provenance, capability, or report-generation logic.

- `function_provenance_report.py`
  - Writes the function provenance and implementation registry reports under `docs/reports/`.
  - Use when public function inventory or provenance metadata changes.

- `hypgeom_status_report.py`
  - Writes the hypergeometric status report under `docs/reports/`.
  - Use when hypgeom mode/kernel status changes.

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

- `package_repo.py`
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
