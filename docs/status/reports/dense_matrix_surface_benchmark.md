# Dense Matrix Surface Benchmark

Date: 2026-03-15
Status: active

## Scope

This report covers the pure-JAX dense `arb_mat` / `acb_mat` matrix surface benchmark in:

- [benchmark_dense_matrix_surface.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_dense_matrix_surface.py)

It focuses on:

- direct solve
- LU-reuse solve
- cached matvec
- transpose / conjugate-transpose
- diagonal extraction

## Environment

- platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`
- backend: `cpu`
- GPU note: CUDA-enabled `jaxlib` was not installed, so JAX ran on CPU

## Results

### `n=16`, `warmup=1`, `runs=3`

- `arb_direct_solve_s`: `0.000121`
- `arb_lu_reuse_s`: `0.000366`
- `arb_cached_matvec_s`: `0.000044`
- `arb_transpose_s`: `0.000024`
- `arb_diag_s`: `0.000030`
- `acb_direct_solve_s`: `0.000056`
- `acb_lu_reuse_s`: `0.000086`
- `acb_cached_matvec_s`: `0.000245`
- `acb_transpose_s`: `0.000035`
- `acb_conjugate_transpose_s`: `0.000058`
- `acb_diag_s`: `0.000016`

### `n=32`, `warmup=1`, `runs=3`

- `arb_direct_solve_s`: `0.000249`
- `arb_lu_reuse_s`: `0.000158`
- `arb_cached_matvec_s`: `0.000238`
- `arb_transpose_s`: `0.000036`
- `arb_diag_s`: `0.000017`
- `acb_direct_solve_s`: `0.000215`
- `acb_lu_reuse_s`: `0.000087`
- `acb_cached_matvec_s`: `0.000308`
- `acb_transpose_s`: `0.000132`
- `acb_conjugate_transpose_s`: `0.000076`
- `acb_diag_s`: `0.000021`

## Notes

- The LU-reuse plan is not universally faster at very small sizes. On this CPU run, `arb_direct_solve` was still faster than LU-reuse at `n=16`.
- By `n=32`, LU-reuse was already faster than direct solve for both `arb_mat` and `acb_mat`.
- The current rigorous dense matrix layer is still wrapper/enclosure based for solve-like operations, but exact structural transforms now preserve interval/box information directly for:
  - permutation matrices
  - transpose / conjugate-transpose
  - submatrix extraction
  - diagonal extraction
  - diagonal-matrix construction

## Related Surfaces

- example notebook: [example_dense_matrix_surface.ipynb](/home/phili/projects/arbplusJAX/examples/example_dense_matrix_surface.ipynb)
- dense matrix tests:
  - [test_arb_mat_chassis.py](/home/phili/projects/arbplusJAX/tests/test_arb_mat_chassis.py)
  - [test_acb_mat_chassis.py](/home/phili/projects/arbplusJAX/tests/test_acb_mat_chassis.py)
  - [test_mat_modes.py](/home/phili/projects/arbplusJAX/tests/test_mat_modes.py)
