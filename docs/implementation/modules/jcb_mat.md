Last updated: 2026-03-07T00:00:00Z

# jcb_mat

## Role

`jcb_mat` is the Jones-labeled subsystem for new complex matrix-function work.

It is separate from [acb_mat](/home/phili/projects/arbplusJAX/src/arbplusjax/acb_mat.py):
- `acb_mat`: canonical Arb/FLINT-style JAX extension surface for complex box matrices
- `jcb_mat`: new matrix-function subsystem for contour-integral, Schur, and rational-Krylov style algorithms

## Layout Contracts

Canonical layouts:
- matrix: `(..., n, n, 4)`
- vector: `(..., n, 4)`

Interpretation:
- trailing `4` stores complex boxes as `[re_lo, re_hi, im_lo, im_hi]`
- matrices must be square
- vectors are logical column vectors stored in box layout

Public contract helpers:
- `jcb_mat_as_box_matrix(a)`
- `jcb_mat_as_box_vector(x)`
- `jcb_mat_shape(a)`

## Current Implemented Substrate

Point mode:
- `jcb_mat_matmul_point(a, b)`
- `jcb_mat_matvec_point(a, x)`
- `jcb_mat_solve_point(a, b)`

Basic mode:
- `jcb_mat_matmul_basic(a, b)`
- `jcb_mat_matvec_basic(a, x)`
- `jcb_mat_solve_basic(a, b)`

Precision/JIT entry points:
- `jcb_mat_matmul_basic_prec`
- `jcb_mat_matvec_basic_prec`
- `jcb_mat_solve_basic_prec`
- `jcb_mat_matmul_basic_jit`
- `jcb_mat_matvec_basic_jit`
- `jcb_mat_solve_basic_jit`

## Current Methodology

Point:
- midpoint complex linear algebra
- result boxed back to an outward complex interval box

Basic:
- `matmul` / `matvec` use box arithmetic in canonical `(..., 4)` layout
- `solve_basic` currently uses midpoint solve plus outward boxing

This means:
- `matmul_basic` and `matvec_basic` are genuine complex-box substrate operations
- `solve_basic` is currently a first substrate implementation, not yet a final rigorous box-linear-solve path

## Not Yet Implemented

Planned matrix-function families:
- `jcb_mat_logm`
- `jcb_mat_sqrtm`
- `jcb_mat_rootm`
- `jcb_mat_signm`

Planned lower-level substrate:
- `triangular_solve`
- `lu`
- `qr`
- Hessenberg / Schur-compatible reductions

## Design Intent

- obey repo dtype, batching, and AD rules
- keep matrix substrate separate from the canonical Arb-like `acb_mat` namespace
- make the substrate reusable for contour-integral and rational-Krylov matrix-function algorithms
