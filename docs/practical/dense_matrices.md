Last updated: 2026-03-18T00:00:00Z

# Dense Matrices

## When To Use Which Dense Path

Use direct dense kernels when:

- you have a one-off call
- the matrix shape changes often
- you do not benefit from plan reuse

Use cached dense `matvec` when:

- the matrix is fixed
- you are applying it repeatedly to many vectors
- you want lower steady-state overhead without switching to an operator abstraction

Use LU / SPD / HPD plans when:

- the matrix is fixed
- you solve repeatedly with different right-hand sides
- transpose solve and add-solve are part of the workflow

Use padded batch helpers when:

- many calls arrive with varying batch counts
- you want stable JAX compilation behavior
- raw single-call latency is less important than keeping recompilation under control

Do not use padded batch helpers for tiny one-off workloads if latency matters. They are primarily a compile-stability tool.

## Recommended Dense Workflow

General dense:

1. prepare the dense matrix in `arb_mat` or `acb_mat` layout
2. use direct `matvec` / `solve` first
3. switch to cached `matvec` or LU plan reuse when repeated calls dominate runtime

Structured dense:

1. let the dense layer auto-detect SPD / HPD structure where applicable
2. use explicit SPD / HPD solve-plan reuse for repeated solves
3. use `eigh` / `eigvalsh` for symmetric / Hermitian spectral work

## JAX Practical Notes

- `point` is the fastest mode for midpoint-only dense work
- `basic` is the main practical enclosure mode
- `rigorous` is strongest today for determinant/trace/norm-style paths
- `adaptive` is useful when you want a wider fallback enclosure

If you are tuning runtime:

- benchmark cached `matvec` separately from padded cached `matvec`
- benchmark real and complex dense paths separately
- expect complex box padded paths to cost materially more than real interval paths

## Where To Look

Implementation details:

- [dense_matrix_tranche.md](/docs/implementation/dense_matrix_tranche.md)
- [arb_mat.md](/docs/implementation/modules/arb_mat.md)
- [acb_mat.md](/docs/implementation/modules/acb_mat.md)

Validation and performance:

- [dense_matrix_surface_benchmark.md](/docs/status/reports/dense_matrix_surface_benchmark.md)
- [example_dense_matrix_surface.ipynb](/examples/example_dense_matrix_surface.ipynb)
- [example_dense_structured_spectral.ipynb](/examples/example_dense_structured_spectral.ipynb)
