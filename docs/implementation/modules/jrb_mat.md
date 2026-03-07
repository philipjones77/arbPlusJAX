Last updated: 2026-03-07T00:00:00Z

# jrb_mat

## Role

`jrb_mat` is the Jones-labeled subsystem for new real matrix-function work.

It is separate from [arb_mat](/home/phili/projects/arbplusJAX/src/arbplusjax/arb_mat.py):
- `arb_mat`: canonical Arb/FLINT-style JAX matrix extension surface
- `jrb_mat`: new matrix-function subsystem for contour-integral, Schur, and rational-Krylov style algorithms

## Layout Contracts

Canonical layouts:
- matrix: `(..., n, n, 2)`
- vector: `(..., n, 2)`

Interpretation:
- trailing `2` stores real intervals as `[lo, hi]`
- matrices must be square
- vectors are column-like logical vectors stored as rank-1 interval arrays

Public contract helpers:
- `jrb_mat_as_interval_matrix(a)`
- `jrb_mat_as_interval_vector(x)`
- `jrb_mat_shape(a)`

## Current Implemented Substrate

Point mode:
- `jrb_mat_matmul_point(a, b)`
- `jrb_mat_matvec_point(a, x)`
- `jrb_mat_solve_point(a, b)`

Basic mode:
- `jrb_mat_matmul_basic(a, b)`
- `jrb_mat_matvec_basic(a, x)`
- `jrb_mat_solve_basic(a, b)`

Precision/JIT entry points:
- `jrb_mat_matmul_basic_prec`
- `jrb_mat_matvec_basic_prec`
- `jrb_mat_solve_basic_prec`
- `jrb_mat_matmul_basic_jit`
- `jrb_mat_matvec_basic_jit`
- `jrb_mat_solve_basic_jit`

## Current Methodology

Point:
- midpoint real linear algebra
- result boxed back to an outward interval

Basic:
- `matmul` / `matvec` use interval arithmetic on the canonical `(..., 2)` layout
- `solve_basic` currently uses midpoint solve plus outward boxing

This means:
- `matmul_basic` and `matvec_basic` are genuine interval substrate operations
- `solve_basic` is currently a first substrate implementation, not a final rigorous interval linear solve

## Not Yet Implemented

Planned matrix-function families:
- `jrb_mat_logm`
- `jrb_mat_sqrtm`
- `jrb_mat_rootm`
- `jrb_mat_signm`

Planned lower-level substrate:
- `triangular_solve`
- `lu`
- `qr`
- Hessenberg / Schur-compatible reductions

## Design Intent

- obey repo dtype, batching, and AD rules
- keep matrix substrate separate from the canonical Arb-like `arb_mat` namespace
- make the substrate reusable for contour-integral and rational-Krylov matrix-function algorithms
