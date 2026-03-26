Last updated: 2026-03-18T00:00:00Z

# Sparse Matrix Tranche

## Scope

The sparse matrix layer now has a dense-style public wrapper surface for the core point-mode kernels in:

- `srb_mat` for real sparse matrices
- `scb_mat` for complex sparse matrices

This tranche now has broader native enclosure parity for the main sparse core
surface as well as block/vblock `basic` paths. Direct sparse `basic`
determinant, inverse, square, Cholesky/LDL, triangular-solve, and solve
entrypoints now route through sparse-native point/factor paths and then lift the
result into interval/box form, while the main sparse `basic` LU and SPD/HPD
solve-plan prepare surfaces continue to lift into dense interval/box solve
plans instead of reusing sparse point plans. The broader sparse stack is still
not at dense-style interval coverage, but callers no longer rely only on
wrapper-level midpoint alignment for those block/vblock surfaces or on dense
bridges for the direct sparse `basic` core entrypoints listed above.

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

- sparse `basic` / `rigorous` / `adaptive` still reuse point kernels across much
  of the core sparse stack, even though main-stack sparse direct `basic`
  determinant, inverse, square, Cholesky/LDL, triangular-solve, and solve
  entrypoints now compute in-family sparse-native midpoint results before
  interval/box lifting, and the main sparse `basic` LU / SPD / HPD plan-prepare
  surfaces now return dense interval/box solve plans
- sparse matrix-function coverage is narrower than dense
- sparse eigenspectrum and broader structured spectral surfaces are still incomplete
- there is still duplication between dense and sparse algorithm families at the numerical-kernel level; this tranche centralizes shared wrapper/helper plumbing, not the dense and sparse linear algebra kernels themselves
- benchmark reports now separate storage-format preparation, cached-plan
  preparation, and factor-plan preparation from solver-quality kernels for
  block/vblock and the main sparse surface; the remaining work is to extend that
  separation more broadly across additional sparse benchmark families
