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
2. for repeated point-mode work, prefer cached `matvec` prepare/apply immediately
3. bind the repeated point batch with `api.bind_point_batch_jit(..., pad_to=...)` when batch counts vary
4. use direct `matvec` / `solve` only when the workload is genuinely one-off or shape churn is unavoidable

Structured dense:

1. let the dense layer auto-detect SPD / HPD structure where applicable
2. use explicit SPD / HPD solve-plan reuse for repeated solves
3. use `eigh` / `eigvalsh` for symmetric / Hermitian spectral work

## Startup-Oriented Dense Workflow

If startup compile behavior matters, the default teaching path is:

1. point-mode dense matrix path
2. cached prepare/apply for repeated matrix application or solve-plan reuse for repeated solves
3. padded point batch binding such as `api.bind_point_batch_jit("arb_mat_matvec_cached_apply", pad_to=...)`

This keeps point-only matrix traffic off interval/mode wrapper imports during startup and makes shape-stable repeated calls the default runtime contract.

## JAX Practical Notes

- `point` is the fastest mode for midpoint-only dense work
- `basic` is the main practical enclosure mode
- `rigorous` is strongest today for determinant/trace/norm-style paths
- `adaptive` is useful when you want a wider fallback enclosure

If you are tuning runtime:

- benchmark point-only API import separately from first compiled cached `matvec`
- benchmark cached `matvec` separately from padded cached `matvec`
- benchmark real and complex dense paths separately
- expect complex box padded paths to cost materially more than real interval paths

## Where To Look

Implementation details:

- [dense_matrix_tranche_implementation.md](/docs/implementation/dense_matrix_tranche_implementation.md)
- [arb_mat_implementation.md](/docs/implementation/modules/arb_mat_implementation.md)
- [acb_mat_implementation.md](/docs/implementation/modules/acb_mat_implementation.md)

Validation and performance:

- [dense_matrix_surface_benchmark.md](/docs/reports/dense_matrix_surface_benchmark.md)
- [dense_matrix_engineering_status.md](/docs/reports/dense_matrix_engineering_status.md)
- [example_dense_matrix_surface.ipynb](/examples/example_dense_matrix_surface.ipynb)
- [example_dense_structured_spectral.ipynb](/examples/example_dense_structured_spectral.ipynb)
