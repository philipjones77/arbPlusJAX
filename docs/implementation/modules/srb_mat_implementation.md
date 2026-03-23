Last updated: 2026-03-15T00:00:00Z

# srb_mat

## Role

`srb_mat` is the sparse real JAX matrix layer.

It is separate from:

- [arb_mat](/src/arbplusjax/arb_mat.py): dense interval matrix layer
- [jrb_mat](/src/arbplusjax/jrb_mat.py): matrix-free operator layer

## Current Scope

Implemented storage families:

- `COO`
- `CSR`
- `BCOO`

Implemented kernel families:

- `shape` / `nnz`
- sparse `zero` / sparse `identity` / sparse permutation matrices
- dense-to-sparse conversion
- sparse-to-dense conversion
- `COO -> CSR`
- `CSR -> COO`
- `COO -> BCOO`
- `CSR -> BCOO`
- `BCOO -> COO`
- transpose
- diagonal extraction
- sparse diagonal-matrix construction
- submatrix extraction
- sparse `matvec`
- sparse times dense-RHS `matmul`
- sparse-sparse `matmul`
- sparse `add` / `sub` / `scale`
- sparse triangular solve
- iterative sparse solve (`cg`, `gmres`, `bicgstab` surface)
- iterative sparse solve (`cg`, `gmres`, `bicgstab` surface), with callable left preconditioners now supported on the `cg` path
- pivoted sparse-fronted `lu` with explicit permutation output
- sparse `lu_solve`
- sparse Householder `qr` factor
- `qr` apply-`Q`, explicit-`Q`, `R` extraction, and `qr_solve`
- cached sparse `matvec` prepare/apply
- fixed/padded batched `matvec` helpers

## Current Constraint

This is a point-value sparse layer, not an interval sparse layer yet.

That means:

- values are plain real JAX arrays
- there is no `basic` / `adaptive` / `rigorous` enclosure semantics yet
- sparse direct factorisation is point-mode only and still lighter than a full symbolic/numeric sparse package
- sparse `qr` now uses a structured Householder factor surface instead of returning dense `Q` as the native object
- dense-only interval families from `arb_mat` are intentionally not mirrored here yet
- `cg` preconditioning is currently function-based: pass `M(v)` as an approximate application of \(A^{-1}\) to the residual, for example a Jacobi/diagonal map

## Coverage

- tests: [test_srb_mat_chassis.py](/tests/test_srb_mat_chassis.py)
- API tests: [test_sparse_point_api.py](/tests/test_sparse_point_api.py)
- benchmark: [benchmark_sparse_matrix_surface.py](/benchmarks/benchmark_sparse_matrix_surface.py)
