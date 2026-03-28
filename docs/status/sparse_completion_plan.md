Last updated: 2026-03-20T00:00:00Z

# Sparse Completion Plan

## Goal

Bring the sparse real/complex matrix surface closer to the dense tranche in:

- structured solve coverage
- plan reuse
- transpose/add/multi-RHS solve aliases
- batch fixed/padded helpers
- shared common-layer implementation

This does not mean sparse interval/box enclosure parity is already complete. The sparse layer is still primarily point-mode today.

New in the current pass:

- dense-style sparse mode wrappers in `mat_wrappers`
- sparse interval-mode API registration through those wrappers
- sparse batch mode fastpaths for LU and SPD/HPD plan-apply surfaces
- shared sparse batch helper utilities and shared sparse cached-matvec-plan helpers
- a broad sparse mode surface test and a sparse matrix surface benchmark

## Current Gap Versus Dense

Dense currently has:

- four-mode wrapper surface
- structured symmetric / Hermitian solve routing
- LU / Cholesky plan reuse
- transpose solve, add-solve, multi-RHS solve aliases
- broader batch and padded-batch coverage

Sparse currently had, before this pass:

- point-mode storage and conversion
- sparse `matvec`, cached `matvec`, and iterative solve
- LU / QR factorization surfaces
- selected sparse Krylov and logdet work in `jrb_mat`

but was missing:

- sparse structured symmetric / Hermitian predicates
- sparse SPD / HPD solve-plan reuse
- sparse LU solve-plan reuse as a first-class plan object
- transpose/add/multi-RHS solve aliases
- a common sparse plan layer matching the dense engineering style

## First Sparse Structured Pass

The first completion pass lands:

- shared sparse LU/Cholesky solve plans in `sparse_common`
- real sparse:
  - `symmetric_part`
  - `is_symmetric`
  - `is_spd`
  - `cho`
  - `ldl`
  - LU solve-plan prepare/apply
  - SPD solve-plan prepare/apply
  - `solve_lu`
  - `solve_transpose`
  - `solve_add`
  - `solve_transpose_add`
  - `mat_solve`
  - `mat_solve_transpose`
  - batch fixed/padded plan-apply helpers
- complex sparse:
  - `hermitian_part`
  - `is_hermitian`
  - `is_hpd`
  - `cho`
  - `ldl`
  - LU solve-plan prepare/apply
  - HPD solve-plan prepare/apply
  - `solve_lu`
  - `solve_transpose`
  - `solve_add`
  - `solve_transpose_add`
  - `mat_solve`
  - `mat_solve_transpose`
  - batch fixed/padded plan-apply helpers

## Remaining Sparse Completion Work

### Landed Sparse-Native Tranche

The first sparse-native tranche is now landed in code:

1. symmetric / Hermitian `eigvalsh` and `eigh` route through sparse spectral surfaces
2. `pow_ui` uses sparse repeated squaring and sparse sparse-matmul
3. `exp` has an explicit sparse action-first / dense-output policy surface
4. `charpoly` uses a sparse trace-based policy instead of silent densification
5. higher sparse functions now expose policy diagnostics showing sparse-native routing and whether sparse output is preserved

The remaining open item from that original list is:

6. `basic` built on a sparse interval/box substrate rather than dense lifting

### Policy Notes

- Sparse code should be optimized for sparse execution, not hidden dense fallback.
- Tiny dense fallback is acceptable only when it is explicit policy and documented as such.
- Structured real symmetric / SPD and complex Hermitian / HPD paths should be preferred whenever those conditions hold.
- Action-first matrix-function APIs are the correct sparse target for functions like `exp` where sparse output is not generally preserved.

### Follow-On Tranche

1. sparse structured diagnostics and benchmarks beyond the current policy diagnostics
2. sparse symmetric / Hermitian cached operator plans beyond plain `matvec`
3. deeper sparse eigenspectrum / structured spectral surfaces
4. deeper sparse selected-inverse and trace-inverse surfaces
5. algorithm-level deduplication between dense and sparse where the numerical kernels genuinely overlap
6. structured sparse plans beyond solve/matvec
7. sparse `basic` interval/box semantics without dense lifting

### Krylov Reuse And Shifted-Solve TODOs

- add shared preconditioner-plan support at the sparse/operator layer
- add multi-shift sparse solve support for rational matrix functions and trace/logdet estimators
- add recycled Krylov support for sequences of closely related sparse solves
- add block multi-RHS Krylov support with shared-basis reuse
- prefer structured reuse where available:
  - symmetric / SPD sparse real should prefer block CG, multi-shift CG, and recycled Lanczos
  - Hermitian / HPD sparse complex should prefer Hermitian-specialized shared-basis solvers
- keep rational node/pole selection separate from sparse operator reuse infrastructure

## Validation Entry Points

- [test_srb_mat_chassis.py](/tests/test_srb_mat_chassis.py)
- [test_scb_mat_chassis.py](/tests/test_scb_mat_chassis.py)
- [test_sparse_structured_surface.py](/tests/test_sparse_structured_surface.py)
- [test_sparse_mode_surface.py](/tests/test_sparse_mode_surface.py)
- [benchmark_sparse_matrix_surface.py](/benchmarks/benchmark_sparse_matrix_surface.py)
