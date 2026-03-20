Last updated: 2026-03-18T00:00:00Z

# Dense Matrix Tranche

## Purpose

This document collects the current dense-matrix tranche into one implementation-facing description.

It covers:

- dense real interval matrices in `arb_mat`
- dense complex box matrices in `acb_mat`
- symmetric / Hermitian structured dense paths
- dense `matvec`
- cached dense `matvec`
- dense solve / factorization reuse
- padded batch helpers used to reduce recompilation pressure

This is the dense matrix layer only. Sparse/operator work is documented separately.

## Public Dense Surface

The dense tranche is centered on:

- direct dense kernels:
  - `matmul`
  - `matvec`
  - `solve`
  - `inv`
  - `lu`
  - `qr`
  - `det`
  - `trace`
  - `norm_*`
- cached dense reuse:
  - `matvec_cached_prepare`
  - `matvec_cached_apply`
  - dense `DenseMatvecPlan` prepare/apply
  - dense LU solve-plan prepare/apply
  - dense SPD / HPD solve-plan prepare/apply
- structured dense paths:
  - symmetric / Hermitian part
  - midpoint structure predicates
  - Cholesky
  - LDL
  - SPD / HPD solve and inverse
  - `eigvalsh`
  - `eigh`
- PETSc-style dense factorization-solve aliases:
  - `solve_lu`
  - `solve_transpose`
  - `solve_add`
  - `solve_transpose_add`
  - `mat_solve`
  - `mat_solve_transpose`
- matrix functions and constructors:
  - `charpoly`
  - `pow_ui`
  - `exp`
  - companion / Hilbert / Pascal / Stirling
- block helpers:
  - block assemble
  - block diagonal
  - block extract / row / col
  - block matmul

All of these are intended to work across the four precision modes:

- `point`
- `basic`
- `rigorous`
- `adaptive`

with direct dense, `matvec`, cached `matvec`, and fixed/padded batch surfaces preserved as first-class entry points.

## Layout Contracts

Real dense:

- matrix: `(..., m, n, 2)`
- vector / rhs: `(..., n, 2)`

Complex dense:

- matrix: `(..., m, n, 4)`
- vector / rhs: `(..., n, 4)`

Important detail:

- the original dense square matrix contract still applies where mathematically required
- the dense execution layer now also accepts rectangular matrices for:
  - dense `matmul`
  - dense `matvec`
  - cached dense `matvec`
  - block assembly/block multiplication internals

That rectangular widening was needed so block dense algebra is valid without forcing every subblock through a square-only validator.

## Common Layer

Dense shared helpers live primarily in:

- [mat_common.py](/home/phili/projects/arbplusJAX/src/arbplusjax/mat_common.py)

This common layer owns:

- dense plan dataclasses:
  - `DenseMatvecPlan`
  - `DenseLUSolvePlan`
  - `DenseCholeskySolvePlan`
- interval/box shape validation
- rectangular dense validators for reusable dense kernels
- midpoint structure helpers
- Cholesky solve helpers
- batch padding helpers
- common finite/full enclosure fallbacks
- matrix-function helpers used by both real and complex dense modules

The main reason for pushing these pieces into the common layer is to keep:

- JAX tracing behavior aligned across real and complex dense code
- plan handling consistent
- padded batch behavior uniform
- helper duplication low enough that dense fixes do not fork into separate real/complex bugs

## Mode Behavior

`point`

- dedicated midpoint kernels in `point_wrappers`
- fastest dense path
- intended to compile independently from enclosure-heavy paths

`basic`

- midpoint evaluation plus outward rounding
- primary practical box/interval dense mode

`rigorous`

- strongest specialized tightening currently exists for:
  - `det`
  - `trace`
  - norm helpers
- many solve/factorization families still route through midpoint-plus-enclosure rather than a distinct tightened rigorous algorithm

`adaptive`

- fixed-grid midpoint sampling plus inflation
- used where a wider but more robust enclosure is preferred over the lighter `basic` path

## Structured Dense Policy

The dense layer is not purely generic. When midpoint structure checks pass, it intentionally routes to a structured dense path.

Real:

- symmetry checks
- SPD checks
- Cholesky / LDL reuse
- symmetric eigensolver route for `charpoly` and `exp` where appropriate

Complex:

- Hermitian checks
- HPD checks
- Hermitian Cholesky / LDL reuse
- Hermitian eigensolver route for `charpoly` and `exp` where appropriate

This is the current dense analogue of Arb/FLINT’s structured dense behavior and is also the place where the library picks up the most obvious Boost-style structured-matrix advantage.

## JAX Engineering Notes

The dense tranche is explicitly optimized around JAX constraints:

- fixed-shape plan objects are reused rather than reconstructed on every call
- padded batch helpers exist to reduce shape-driven recompilation
- dense plans are treated as static-like pytrees instead of generic padded tensors
- helper logic is centralized so real/complex dense tracing is structurally similar
- midpoint-heavy kernels are kept separate from wrapper-driven enclosure logic

Current known bottlenecks:

- padded cached `matvec`
- complex padded cached `matvec` in particular
- block-diagonal assembly due to repeated concatenate-heavy construction

These are runtime bottlenecks, not correctness blockers.

## Validation

Current dense tranche validation includes:

- [test_dense_plan_modes.py](/home/phili/projects/arbplusJAX/tests/test_dense_plan_modes.py)
- [test_dense_broad_surface.py](/home/phili/projects/arbplusJAX/tests/test_dense_broad_surface.py)
- [test_dense_eigh_and_aliases.py](/home/phili/projects/arbplusJAX/tests/test_dense_eigh_and_aliases.py)
- [test_dense_structured_modes.py](/home/phili/projects/arbplusJAX/tests/test_dense_structured_modes.py)
- [test_mat_modes.py](/home/phili/projects/arbplusJAX/tests/test_mat_modes.py)
- [test_arb_mat_chassis.py](/home/phili/projects/arbplusJAX/tests/test_arb_mat_chassis.py)
- [test_acb_mat_chassis.py](/home/phili/projects/arbplusJAX/tests/test_acb_mat_chassis.py)

Most recent dense tranche run:

- `55 passed in 82.23s`

The latest pass also fixed:

- block helper support for rectangular dense subblocks
- ambiguous boolean conversion in dense containment-mode tests

## Benchmark Entry Points

Primary dense benchmark:

- [benchmark_dense_matrix_surface.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_dense_matrix_surface.py)

Current benchmark report:

- [dense_matrix_surface_benchmark.md](/home/phili/projects/arbplusJAX/docs/status/reports/dense_matrix_surface_benchmark.md)

## Example Entry Points

- [example_dense_matrix_surface.ipynb](/home/phili/projects/arbplusJAX/examples/example_dense_matrix_surface.ipynb)
- [example_dense_structured_spectral.ipynb](/home/phili/projects/arbplusJAX/examples/example_dense_structured_spectral.ipynb)

## Remaining Boundaries

This tranche is broad, but not literal full Arb/FLINT dense parity.

Main remaining boundaries:

- many `basic` / `rigorous` solve-family paths are still midpoint-first rather than true enclosure-native algorithms
- high-end general spectral work remains limited compared with a full external linear algebra stack
- padded batch helpers still trade latency for compile stability; they are not the fastest path for small one-off calls

For the current repository direction, the dense tranche should be treated as:

- feature-complete enough for dense runtime use
- strongly validated across the four mode surfaces
- still open to performance tightening rather than major surface redesign
