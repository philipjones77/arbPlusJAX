Last updated: 2026-03-22T00:00:00Z

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

For this category to be treated as complete in the current CPU tranche, the
repo should have all of the following:

- direct owner tests for each family
- parity/reference coverage for each family where reference code exists
- scalar benchmark coverage for helper families
- scalar comparison coverage for `arb_core` and `acb_core`
- canonical notebook coverage for scalar/API usage
- retained runtime/summary artifacts for the CPU notebook run

## Current Status

Status: `done` for the current CPU tranche.

Why:

- direct owner tests exist for all listed families
- parity owners exist for all listed families
- scalar benchmark scripts exist and emit shared-schema JSON
- `compare_arb_core.py` and `compare_acb_core.py` provide C-reference accuracy
  artifacts
- `example_core_scalar_surface.ipynb` is the canonical scalar notebook
- `example_api_surface.ipynb` provides the routed API companion notebook
- CPU notebook outputs now retain executed notebooks, manifests, summaries,
  comparison status, and plots under `examples/outputs/`

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
