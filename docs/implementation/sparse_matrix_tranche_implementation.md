Last updated: 2026-03-18T00:00:00Z

# Sparse Matrix Tranche

## Scope

The sparse matrix layer now has a dense-style public wrapper surface for the core point-mode kernels in:

- `srb_mat` for real sparse matrices
- `scb_mat` for complex sparse matrices

This tranche does not claim interval enclosure parity with dense. The sparse kernels are still point implementations underneath, but the public mode-dispatch surface is now aligned enough that callers can use the same `impl="point|basic|rigorous|adaptive"` shape without a separate sparse-only calling convention.

## Shared Common Layer

The sparse common layer in `src/arbplusjax/sparse_common.py` now owns the reusable helpers that were drifting across sparse modules:

- sparse storage dataclasses and plan dataclasses
- sparse matvec-plan prepare/apply
- padded batch repeat-last logic
- vmapped fixed-batch and padded-batch helper execution

This removes duplicated batch padding and cached-matvec-plan code from `srb_mat.py` and `scb_mat.py`.

## Public Sparse Wrapper Surface

`src/arbplusjax/mat_wrappers.py` now exports mode wrappers for sparse core functions, including:

- constructors and structural helpers
- `transpose` / `conjugate_transpose`
- `symmetric_part` / `hermitian_part`
- `is_symmetric` / `is_hermitian`
- `is_spd` / `is_hpd`
- `matvec`, cached `matvec`, sparse-dense multiply, sparse-sparse multiply
- triangular solve, direct solve, LU solve, transpose/add solve aliases
- LU and Cholesky/HPD plan prepare/apply
- `det`, `inv`, `sqr`, norms, trace

Batch mode wrappers are exposed for the sparse operations that already have fixed/padded batch kernels:

- `matvec`
- cached `matvec`
- `solve`
- triangular solve
- LU plan apply
- SPD/HPD plan apply

## API Surface

`src/arbplusjax/api.py` now binds sparse names to the sparse mode wrappers for interval-mode dispatch and registers the sparse batch mode fastpaths. This means:

- `api.eval_interval("srb_mat_det", sparse, mode="adaptive")` now resolves to the sparse mode wrapper instead of falling back to the raw point function
- padded sparse batch solves can use `api.eval_interval_batch(...)` directly

## Remaining Work

Sparse still differs from dense in important ways:

- sparse `basic` / `rigorous` / `adaptive` currently reuse point kernels rather than producing interval/box enclosures
- sparse matrix-function coverage is narrower than dense
- sparse eigenspectrum and broader structured spectral surfaces are still incomplete
- there is still duplication between dense and sparse algorithm families at the numerical-kernel level; this tranche centralizes shared wrapper/helper plumbing, not the dense and sparse linear algebra kernels themselves
