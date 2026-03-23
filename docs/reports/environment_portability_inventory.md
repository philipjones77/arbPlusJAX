Last updated: 2026-03-22T00:00:00Z

# Environment Portability Inventory

This report records the current portability surfaces for WSL and cloud notebook
environments.

Policy lives in:

- [environment_portability_standard.md](/docs/standards/environment_portability_standard.md)

Checked-in configuration now also lives in:

- [cpu_validation_profiles.json](/configs/cpu_validation_profiles.json)

## Current Portable Entry Points

### Test harness

- [run_test_harness.py](/tools/run_test_harness.py)

Current portability-relevant behavior:

- explicit `--jax-mode`
- repo-relative invocation
- shared `runtime_manifest.json` output via `--outdir`
- category-aligned bounded profiles now exist for:
  - `core`
  - `modes`
  - `dense`
  - `sparse`
  - `matrix-free`
  - `special`

### Benchmark harness

- [run_benchmarks.py](/benchmarks/run_benchmarks.py)
- [bench_harness.py](/benchmarks/bench_harness.py)

Current portability-relevant behavior:

- shared runtime environment handling
- shared runtime manifest output
- optional external backends degrade by environment

### Colab bootstrap

- [colab_bootstrap.sh](/tools/colab_bootstrap.sh)

Status:

- present

### Example notebooks

- [README.md](/examples/README.md)

Current portability-relevant behavior:

- explicit `JAX_MODE`
- explicit `JAX_DTYPE`
- environment printout in notebooks

### Runtime manifest

- [runtime_manifest.py](/tools/runtime_manifest.py)
- [test_runtime_manifest.py](/tests/test_runtime_manifest.py)

Status:

- shared manifest schema already in use across test and benchmark harnesses

## Current WSL / Colab Docs

- [run_platform_implementation.md](/docs/implementation/run_platform_implementation.md)
- [linux_gpu_colab_implementation.md](/docs/implementation/linux_gpu_colab_implementation.md)
- [running.md](/docs/practical/running.md)

## Current Strengths

- WSL is already used in practice for local JAX runs
- Google Colab is already mentioned in test/example run docs
- runtime manifests already capture environment details
- examples/tests/benchmarks already have explicit environment-aware harness entrypoints
- the bounded CPU validation slice is now expressed as checked-in config under `configs/`

## Current Gaps

- portability guidance is spread across multiple docs
- not every notebook or experiment currently summarizes portability assumptions explicitly
- the third-party comparison software install matrix is still documented in prose rather than a single machine-readable inventory

## Current Portability Conclusion

The repo is already partly portable across WSL and Colab, but this portability
was previously documented in a scattered way.

The new portability standard turns that into an explicit repo rule.
