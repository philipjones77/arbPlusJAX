Last updated: 2026-03-15T00:00:00Z

# scb_mat

## Role

`scb_mat` is the sparse complex JAX matrix layer.

It is separate from:

- [acb_mat](/home/phili/projects/arbplusJAX/src/arbplusjax/acb_mat.py): dense complex box matrix layer
- [jcb_mat](/home/phili/projects/arbplusJAX/src/arbplusjax/jcb_mat.py): matrix-free complex operator layer

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
- conjugate-transpose
- diagonal extraction
- sparse diagonal-matrix construction
- submatrix extraction
- sparse `matvec`
- sparse times dense-RHS `matmul`
- sparse-sparse `matmul`
- sparse `add` / `sub` / `scale`
- sparse triangular solve
- iterative sparse solve (`cg`, `gmres`, `bicgstab` surface)
- pivoted sparse-fronted `lu` with explicit permutation output
- sparse `lu_solve`
- sparse Householder `qr` factor
- `qr` apply-`Q`, explicit-`Q`, `R` extraction, and `qr_solve`
- cached sparse `matvec` prepare/apply
- fixed/padded batched `matvec` helpers

## Current Constraint

This is a point-value sparse complex layer, not a box/interval sparse layer yet.

That means:

- values are plain complex JAX arrays
- there is no `basic` / `adaptive` / `rigorous` enclosure semantics yet
- sparse direct factorisation is point-mode only and still lighter than a full symbolic/numeric sparse package
- sparse `qr` now uses a structured Householder factor surface instead of returning dense `Q` as the native object
- dense-only box/interval families from `acb_mat` are intentionally not mirrored here yet

## Coverage

- tests: [test_scb_mat_chassis.py](/home/phili/projects/arbplusJAX/tests/test_scb_mat_chassis.py)
- API tests: [test_sparse_point_api.py](/home/phili/projects/arbplusJAX/tests/test_sparse_point_api.py)
- benchmark: [benchmark_sparse_matrix_surface.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_sparse_matrix_surface.py)
