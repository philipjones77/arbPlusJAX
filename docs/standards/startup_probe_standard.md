Last updated: 2026-03-26T00:00:00Z

# Startup Probe Standard

## Purpose

Every hot family needs an import/compile probe and a CI budget.

This standard defines the minimum measurement contract for startup-sensitive
runtime families.

Companion documents:

- [startup_compile_standard.md](/docs/standards/startup_compile_standard.md)
- [startup_import_boundary_standard.md](/docs/standards/startup_import_boundary_standard.md)

## Required Measurements

Each representative startup probe must measure these phases separately:

1. import time
2. JAX backend initialization time
3. first compile plus first real call
4. steady repeated call

For prepare/apply families, probes should also measure prepare cost separately.

## Required Coverage

The repo must maintain probes for representative hot families across the main
runtime categories.

At minimum:

- one point special-function family
- one matrix or sparse cached-apply family
- one alternative/provider family when such a family is public and hot

## CI Budget Policy

1. Probes must produce checked-in machine-readable artifacts.
   Preferred outputs:
   - JSON
   - short Markdown summary

2. CI or release checks must compare probe results against budgets.
   Budgets may be:
   - hard thresholds
   - regression deltas against a checked-in baseline

3. Compile failures count as startup regressions.
   A probe that returns a compile error instead of timings must fail the budget
   check unless the family is explicitly marked experimental.

## Probe Design Rules

- use cold subprocesses for import measurement
- keep backend-init measurement separate from module import
- use stable-shape representative inputs
- make family/shape/static kwargs explicit in the output artifact

## Representative Probes In This Repo

- [hypgeom_point_startup_probe.py](/benchmarks/hypgeom_point_startup_probe.py)
- [matrix_point_startup_probe.py](/benchmarks/matrix_point_startup_probe.py)
- [dirichlet_point_startup_probe.py](/benchmarks/dirichlet_point_startup_probe.py)
- [sparse_cached_apply_startup_probe.py](/benchmarks/sparse_cached_apply_startup_probe.py)
- [double_gamma_point_startup_probe.py](/benchmarks/double_gamma_point_startup_probe.py)
