Last updated: 2026-03-27T00:00:00Z

# Release Execution Checklist

This is the concrete runbook for satisfying
[release_governance_standard.md](/docs/standards/release_governance_standard.md).

## Required Deliverables

Before a release-quality change is considered complete, record:

- scope touched
- tests run
- benchmark/probe commands run
- example notebook commands run when applicable
- generated artifact refresh commands run
- known residual risks or intentionally skipped slices

## Minimum Command Matrix

Use the minimum subset that matches the touched area.

### 1. Correctness

Owner tests:

```bash
python -m pytest -q <targeted tests>
```

Broader chassis/profile slice when runtime behavior changed:

```bash
python tools/run_test_harness.py --profile chassis --jax-mode cpu
```

### 2. Startup / Import / Compile

Cold-path and first-use reports:

```bash
python tools/api_cold_path_report.py
python tools/api_first_use_report.py
python tools/matrix_free_first_use_report.py
```

Family probes where touched:

```bash
python benchmarks/matrix_point_startup_probe.py
python benchmarks/hypgeom_point_startup_probe.py
```

### 3. Benchmarks

Use the owning benchmark or probe for the touched family. Examples:

```bash
python benchmarks/special_function_hardening_benchmark.py
python benchmarks/run_benchmarks.py --profile quick
```

### 4. Examples

When public notebook surfaces changed:

```bash
python tools/run_example_notebooks.py --jax-mode cpu --jax-dtype float64
```

### 5. Generated Artifacts

Refresh:

```bash
python tools/update_repo_artifacts.py
```

Non-mutating drift check:

```bash
python tools/check_repo_update_drift.py
```

## Matrix / Matrix-Free Addendum

When dense, sparse, or matrix-free surfaces changed, also consider:

```bash
python -m pytest -q tests/test_sparse_format_modes.py
python -m pytest -q tests/test_matrix_free_import_boundaries.py
python -m pytest -q tests/test_matrix_free_first_use_report.py
python -m pytest -q tests/test_matrix_free_implicit_adjoint_gradients.py
```

## Special-Function Addendum

When special-function surfaces changed, also consider:

```bash
python -m pytest -q tests/test_special_function_hardening.py
python -m pytest -q tests/test_special_function_ad_directions.py
python benchmarks/special_function_hardening_benchmark.py
```

## Checklist Close-Out

A release-quality close-out message should state:

- what changed
- what commands were run
- which generated artifacts were refreshed
- whether startup/import, compile, AD, benchmark, and example surfaces were
  covered
- what remains intentionally out of scope
