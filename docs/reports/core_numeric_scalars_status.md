Last updated: 2026-03-28T00:00:00Z

# Core Numeric Scalars Status

## Purpose

This report is the canonical rollup for
`1. Core Numeric Scalars` from
[test_coverage_matrix.md](/docs/status/test_coverage_matrix.md).

It records which families are:

- full scalar interval/core surfaces
- point-only helper scalar surfaces
- covered by direct tests
- covered by parity/reference checks
- covered by benchmarks
- covered by canonical examples
- verified on CPU and GPU for the owned scalar tranche

## Category Scope

This category covers:

- `arb_core`
- `acb_core`
- `arf`
- `acf`
- `fmpr`
- `fmpzi`
- `arb_fpwrap`

Supporting scalar-adjacent layers also exist, but this report is focused on the
primary family set above.

## Family Rollup

| family | surface kind | interval / box modes | rigorous status | parity status | benchmark status | example status |
|---|---|---|---|---|---|---|
| `arb_core` | canonical scalar real interval core | yes | specialized interval kernel surface | direct C parity owner | compare + chassis + parity covered | covered |
| `acb_core` | canonical scalar complex box core | yes | specialized interval kernel surface | direct C parity owner | compare + chassis + parity covered | covered |
| `arf` | point helper scalar real arithmetic | no | not applicable; point-only helper | direct parity owner | covered | covered |
| `acf` | point helper scalar complex arithmetic | no | not applicable; point-only helper | direct parity owner | covered | covered |
| `fmpr` | point helper scalar floating arithmetic | no | not applicable; point-only helper | direct parity owner | covered | covered |
| `fmpzi` | point helper integer-interval arithmetic | no four-mode dispatch; integer interval helper semantics only | helper-level interval object, not four-mode scalar rigorous dispatch | direct parity owner | covered | covered |
| `arb_fpwrap` | point helper float/complex wrapper surface | no | not applicable; point-only helper | direct parity owner | covered | covered |

## Interpretation

The important distinction is:

- `arb_core` and `acb_core` are the true scalar interval/box kernel surfaces in
  this category
- `arf`, `acf`, `fmpr`, `fmpzi`, and `arb_fpwrap` are helper scalar families
  and are intentionally point-oriented rather than four-mode interval kernels

So for this category, "specialized rigorous" applies to the canonical
`arb_core` / `acb_core` interval families, not to every helper family.

That is the correct production interpretation for the tranche.

## Completion Criteria

For this category to be treated as complete in the current CPU/GPU tranche, the
repo should have all of the following:

- direct owner tests for each family
- parity/reference coverage for each family where reference code exists
- scalar benchmark coverage for helper families
- scalar comparison coverage for `arb_core` and `acb_core`
- canonical notebook coverage for scalar/API usage
- retained runtime/summary artifacts for the CPU and GPU notebook runs
- CPU and GPU execution of the owned scalar JAX-facing test slice
- explicit backend-realized performance guidance for when CPU remains the
  preferred backend versus when GPU is worth using

## Current Status

Status: `done` for the current CPU/GPU tranche.

Why:

- direct owner tests exist for all listed families
- parity owners exist for all listed families
- scalar benchmark scripts exist and emit shared-schema JSON
- `compare_arb_core.py` and `compare_acb_core.py` provide C-reference accuracy
  artifacts
- `example_core_scalar_surface.ipynb` is the canonical scalar notebook
- `example_api_surface.ipynb` provides the routed API companion notebook
- CPU and GPU notebook outputs now retain executed notebooks, manifests,
  summaries, comparison status, and plots under `examples/outputs/`
- the owned scalar CPU parity/owner slice passes:
  `38 passed, 7 skipped`
- the owned scalar GPU owner slice passes on CUDA:
  `38 passed`
- the current backend-realized result is explicit rather than assumed:
  CPU remains the default winner for many tiny repeated scalar service calls,
  while GPU is validated and available for larger repeated batch-heavy scalar
  workloads

## Current Backend Interpretation

The scalar category is now structurally and operationally closed, but the
backend conclusion is workload-sensitive.

What is true now:

- CPU and GPU execution are both verified for the owned scalar JAX-facing
  tranche
- backend-aware binders, prewarm hooks, diagnostics-bearing binders, and stable
  shape controls exist on the public scalar service surface
- canonical CPU and GPU notebooks retain benchmark artifacts, AD plots, and
  profile summaries

What the current scalar benchmark evidence says:

- CPU is still the preferred default for many tiny scalar service workloads
- GPU is available and validated, but it is not a universal latency win for
  these scalar helper kernels
- padded or batch-heavy raw/API paths can help on GPU, but service-binder
  overhead and launch/compile cost still matter heavily for small scalar jobs

So this category is complete in the sense of:

- public/API/mode surface present
- tests/parity/benchmarks/examples retained
- CPU/GPU verified
- backend guidance explicit

It is not complete in the sense of:

- GPU must beat CPU for every scalar workload

That stronger claim would be false and is not the repo standard.

## Canonical Evidence

Primary tests:

- [test_arb_core_chassis.py](/tests/test_arb_core_chassis.py)
- [test_acb_core_chassis.py](/tests/test_acb_core_chassis.py)
- [test_arf_chassis.py](/tests/test_arf_chassis.py)
- [test_acf_chassis.py](/tests/test_acf_chassis.py)
- [test_fmpr_chassis.py](/tests/test_fmpr_chassis.py)
- [test_fmpzi_chassis.py](/tests/test_fmpzi_chassis.py)
- [test_arb_fpwrap_chassis.py](/tests/test_arb_fpwrap_chassis.py)
- [test_core_scalar_api_contracts.py](/tests/test_core_scalar_api_contracts.py)
- [test_core_scalar_service_contracts.py](/tests/test_core_scalar_service_contracts.py)

Parity tests:

- [test_arb_core_parity.py](/tests/test_arb_core_parity.py)
- [test_acb_core_parity.py](/tests/test_acb_core_parity.py)
- [test_arf_parity.py](/tests/test_arf_parity.py)
- [test_acf_parity.py](/tests/test_acf_parity.py)
- [test_fmpr_parity.py](/tests/test_fmpr_parity.py)
- [test_fmpzi_parity.py](/tests/test_fmpzi_parity.py)
- [test_arb_fpwrap_parity.py](/tests/test_arb_fpwrap_parity.py)

Benchmarks and comparisons:

- [benchmark_arf.py](/benchmarks/benchmark_arf.py)
- [benchmark_acf.py](/benchmarks/benchmark_acf.py)
- [benchmark_fmpr.py](/benchmarks/benchmark_fmpr.py)
- [benchmark_fmpzi.py](/benchmarks/benchmark_fmpzi.py)
- [benchmark_arb_fpwrap.py](/benchmarks/benchmark_arb_fpwrap.py)
- [compare_arb_core.py](/benchmarks/compare_arb_core.py)
- [compare_acb_core.py](/benchmarks/compare_acb_core.py)

Examples:

- [example_core_scalar_surface.ipynb](/examples/example_core_scalar_surface.ipynb)
- [example_api_surface.ipynb](/examples/example_api_surface.ipynb)

Retained executed artifacts:

- [example_core_scalar_surface_cpu_executed.ipynb](/examples/outputs/example_core_scalar_surface/example_core_scalar_surface_cpu_executed.ipynb)
- [example_core_scalar_surface_gpu_executed.ipynb](/examples/outputs/example_core_scalar_surface/example_core_scalar_surface_gpu_executed.ipynb)
- [execution_summary_cpu.json](/examples/outputs/example_core_scalar_surface/execution_summary_cpu.json)
- [execution_summary_gpu.json](/examples/outputs/example_core_scalar_surface/execution_summary_gpu.json)
- [benchmark_core_scalar_service_api_cpu_policy_refresh.json](/benchmarks/results/benchmark_core_scalar_service_api/benchmark_core_scalar_service_api_cpu_policy_refresh.json)
- [benchmark_core_scalar_service_api_gpu_policy_refresh.json](/benchmarks/results/benchmark_core_scalar_service_api/benchmark_core_scalar_service_api_gpu_policy_refresh.json)
