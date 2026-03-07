Last updated: 2026-03-07T00:00:00Z

# jcb_mat

## Role

`jcb_mat` is the Jones-labeled subsystem for new complex matrix-function work.

It is separate from [acb_mat](/home/phili/projects/arbplusJAX/src/arbplusjax/acb_mat.py):
- `acb_mat`: canonical Arb/FLINT-style JAX matrix extension surface
- `jcb_mat`: new matrix-function subsystem for contour-integral, Schur, and rational-Krylov style algorithms on complex box matrices

## Current State

- scaffold only
- canonical layout helper:
  - `jcb_mat_as_box_matrix(a)` for `(..., n, n, 4)`
- planned public families:
  - `jcb_mat_logm`
  - `jcb_mat_sqrtm`
  - `jcb_mat_rootm`
  - `jcb_mat_signm`

## Design Intent

- obey repo dtype, batching, and AD rules
- support four-mode semantics where mathematically appropriate
- hold Jones-lineage matrix-function algorithms without overloading the canonical Arb-like matrix namespace

## Notes

- this module currently defines structure and naming only
- algorithm work is tracked in [todo.md](/home/phili/projects/arbplusJAX/docs/todo.md)
