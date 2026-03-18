# Dense Matrix Surface Benchmark

Date: 2026-03-18
Status: active

## Scope

This report covers the pure-JAX dense `arb_mat` / `acb_mat` matrix surface benchmark in:

- [benchmark_dense_matrix_surface.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_dense_matrix_surface.py)

It focuses on:

- direct solve
- LU-reuse solve
- SPD / HPD structured solve
- SPD / HPD structured plan-reuse solve
- dense matvec plan preparation
- cached matvec
- cached matvec padded batch apply
- transpose / conjugate-transpose
- diagonal extraction

## Environment

- platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`
- backend: `cpu`
- GPU note: CUDA-enabled `jaxlib` was not installed, so JAX ran on CPU

## Results

### `n=16`, `warmup=1`, `runs=1`

- `arb_direct_solve_s`: `0.000076`
- `arb_lu_reuse_s`: `0.000163`
- `arb_dense_plan_prepare_s`: `0.000035`
- `arb_cached_matvec_s`: `0.000075`
- `arb_cached_matvec_padded_s`: `0.003993`
- `arb_spd_solve_s`: `0.000091`
- `arb_spd_plan_solve_s`: `0.000063`
- `arb_transpose_s`: `0.000038`
- `arb_diag_s`: `0.000028`
- `acb_direct_solve_s`: `0.000090`
- `acb_lu_reuse_s`: `0.000157`
- `acb_dense_plan_prepare_s`: `0.000023`
- `acb_cached_matvec_s`: `0.000109`
- `acb_cached_matvec_padded_s`: `0.015562`
- `acb_hpd_solve_s`: `0.000112`
- `acb_hpd_plan_solve_s`: `0.000057`
- `acb_transpose_s`: `0.000070`
- `acb_conjugate_transpose_s`: `0.000027`
- `acb_diag_s`: `0.000027`

## Notes

- Dense plan preparation itself is cheap on CPU at this size; the dominant extra cost is padded batch apply, not plan construction.
- The padded cached matvec path is materially slower than single-vector cached apply at `n=16`, which is expected because it trades raw latency for fixed-shape batching and reduced recompilation pressure.
- The structured dense paths are now competitive with the generic direct solve and cheaper once a Cholesky plan is reused.
- The current rigorous dense matrix layer is still wrapper/enclosure based for solve-like operations, but exact structural transforms now preserve interval/box information directly for:
  - permutation matrices
  - transpose / conjugate-transpose
  - symmetric / Hermitian part extraction
  - submatrix extraction
  - diagonal extraction
  - diagonal-matrix construction
  - dense matvec plan preparation
  - dense LU-reuse plan preparation
  - dense SPD / HPD plan preparation

## Related Surfaces

- example notebook: [example_dense_matrix_surface.ipynb](/home/phili/projects/arbplusJAX/examples/example_dense_matrix_surface.ipynb)
- dense matrix tests:
  - [test_arb_mat_chassis.py](/home/phili/projects/arbplusJAX/tests/test_arb_mat_chassis.py)
  - [test_acb_mat_chassis.py](/home/phili/projects/arbplusJAX/tests/test_acb_mat_chassis.py)
  - [test_mat_modes.py](/home/phili/projects/arbplusJAX/tests/test_mat_modes.py)
  - [test_dense_plan_modes.py](/home/phili/projects/arbplusJAX/tests/test_dense_plan_modes.py)
  - [test_dense_structured_modes.py](/home/phili/projects/arbplusJAX/tests/test_dense_structured_modes.py)
