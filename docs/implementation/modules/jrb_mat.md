Last updated: 2026-03-07T00:00:00Z

# jrb_mat

## Role

`jrb_mat` is the Jones-labeled subsystem for new real matrix-function work.

It is separate from [arb_mat](/home/phili/projects/arbplusJAX/src/arbplusjax/arb_mat.py):
- `arb_mat`: canonical Arb/FLINT-style JAX matrix extension surface
- `jrb_mat`: new matrix-function subsystem for contour-integral, Schur, and rational-Krylov style algorithms

## Current State

- scaffold only
- canonical layout helper:
  - `jrb_mat_as_interval_matrix(a)` for `(..., n, n, 2)`
- planned public families:
  - `jrb_mat_logm`
  - `jrb_mat_sqrtm`
  - `jrb_mat_rootm`
  - `jrb_mat_signm`

## Design Intent

- obey repo dtype, batching, and AD rules
- support four-mode semantics where mathematically appropriate
- hold Jones-lineage matrix-function algorithms without overloading the canonical Arb-like matrix namespace

## Notes

- this module currently defines structure and naming only
- algorithm work is tracked in [todo.md](/home/phili/projects/arbplusJAX/docs/todo.md)
