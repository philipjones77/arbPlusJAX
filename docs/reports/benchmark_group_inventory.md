Last updated: 2026-03-22T00:00:00Z

# Benchmark Group Inventory

This report records the current benchmark grouping in the repo.

Policy and grouping rules live in:

- [benchmark_grouping_standard.md](/home/phili/projects/arbplusJAX/docs/standards/benchmark_grouping_standard.md)
- [benchmark_validation_policy.md](/home/phili/projects/arbplusJAX/docs/standards/benchmark_validation_policy.md)

The executable source of truth is:

- [taxonomy.py](/home/phili/projects/arbplusJAX/benchmarks/taxonomy.py)

## Official Benchmarks

- `core_accuracy` -> `bench_harness.py`
- `api_speed` -> `benchmark_api_surface.py`
- `matrix_speed` -> `benchmark_matrix_suite.py`
- `matrix_compile` -> `benchmark_matrix_stack_diagnostics.py`
- `matrix_ad` -> `benchmark_matrix_free_krylov.py`
- `matrix_backend_compare` -> `benchmark_matrix_backend_candidates.py`
- `transform_speed` -> `benchmark_fft_nufft.py`
- `transform_backend_compare` -> `benchmark_nufft_backends.py`
- `transform_gpu` -> `benchmark_fft_nufft.py`

## Intent Groups

### Accuracy

- `benchmark_gamma_compare.py`
- `benchmark_loggamma_compare.py`

### Performance

- `bench_harness.py`
- `benchmark_acb_calc.py`
- `benchmark_acb_dirichlet.py`
- `benchmark_acb_elliptic.py`
- `benchmark_acb_mat.py`
- `benchmark_acb_modular.py`
- `benchmark_acb_poly.py`
- `benchmark_acf.py`
- `benchmark_api_surface.py`
- `benchmark_arb_calc.py`
- `benchmark_arb_fmpz_poly.py`
- `benchmark_arb_fpwrap.py`
- `benchmark_arb_mat.py`
- `benchmark_arb_poly.py`
- `benchmark_arf.py`
- `benchmark_barnes_double_gamma.py`
- `benchmark_bernoulli.py`
- `benchmark_block_sparse_matrix_surface.py`
- `benchmark_bool_mat.py`
- `benchmark_dense_matrix_surface.py`
- `benchmark_dirichlet.py`
- `benchmark_dlog.py`
- `benchmark_fft_nufft.py`
- `benchmark_fmpr.py`
- `benchmark_fmpz_extras.py`
- `benchmark_fmpzi.py`
- `benchmark_hypgeom.py`
- `benchmark_hypgeom_extra.py`
- `benchmark_mag.py`
- `benchmark_matrix_suite.py`
- `benchmark_partitions.py`
- `benchmark_sparse_matrix_surface.py`
- `benchmark_vblock_sparse_matrix_surface.py`

### Compile

- `benchmark_matrix_stack_diagnostics.py`

### AD

- `benchmark_matrix_free_krylov.py`

### Comparison

- `benchmark_matrix_backend_candidates.py`
- `benchmark_nufft_backends.py`
- `compare_acb_calc.py`
- `compare_acb_core.py`
- `compare_acb_dirichlet.py`
- `compare_acb_elliptic.py`
- `compare_acb_mat.py`
- `compare_acb_modular.py`
- `compare_acb_poly.py`
- `compare_acf.py`
- `compare_arb_calc.py`
- `compare_arb_core.py`
- `compare_arb_fmpz_poly.py`
- `compare_arb_fpwrap.py`
- `compare_arb_mat.py`
- `compare_arb_poly.py`
- `compare_arf.py`
- `compare_ball_wrappers.py`
- `compare_bernoulli.py`
- `compare_bool_mat.py`
- `compare_dft.py`
- `compare_dirichlet.py`
- `compare_dlog.py`
- `compare_double_interval.py`
- `compare_fmpr.py`
- `compare_fmpz_extras.py`
- `compare_fmpzi.py`
- `compare_hypgeom.py`
- `compare_mag.py`
- `compare_mpmath.py`
- `compare_partitions.py`

## Functionality Groups

### API

- `benchmark_api_surface.py`

### Scalar

- `bench_harness.py`
- `benchmark_acb_calc.py`
- `benchmark_acb_poly.py`
- `benchmark_acf.py`
- `benchmark_arb_calc.py`
- `benchmark_arb_fmpz_poly.py`
- `benchmark_arb_fpwrap.py`
- `benchmark_arb_poly.py`
- `benchmark_arf.py`
- `benchmark_fmpr.py`
- `benchmark_fmpz_extras.py`
- `benchmark_fmpzi.py`
- `benchmark_mag.py`
- `compare_acb_calc.py`
- `compare_acb_core.py`
- `compare_acb_poly.py`
- `compare_acf.py`
- `compare_arb_calc.py`
- `compare_arb_core.py`
- `compare_arb_fmpz_poly.py`
- `compare_arb_fpwrap.py`
- `compare_arb_poly.py`
- `compare_arf.py`
- `compare_ball_wrappers.py`
- `compare_dft.py`
- `compare_double_interval.py`
- `compare_fmpr.py`
- `compare_fmpz_extras.py`
- `compare_fmpzi.py`
- `compare_mag.py`
- `compare_mpmath.py`

### Special

- `benchmark_acb_dirichlet.py`
- `benchmark_acb_elliptic.py`
- `benchmark_acb_modular.py`
- `benchmark_barnes_double_gamma.py`
- `benchmark_dirichlet.py`
- `benchmark_gamma_compare.py`
- `benchmark_hypgeom.py`
- `benchmark_hypgeom_extra.py`
- `benchmark_loggamma_compare.py`
- `compare_acb_dirichlet.py`
- `compare_acb_elliptic.py`
- `compare_acb_modular.py`
- `compare_dirichlet.py`
- `compare_hypgeom.py`

### Combinatorics

- `benchmark_bernoulli.py`
- `benchmark_dlog.py`
- `benchmark_partitions.py`
- `compare_bernoulli.py`
- `compare_dlog.py`
- `compare_partitions.py`

### Transform

- `benchmark_fft_nufft.py`

### Backend Matrix

- `benchmark_matrix_backend_candidates.py`

### Backend Transform

- `benchmark_nufft_backends.py`

### Matrix

- `benchmark_acb_mat.py`
- `benchmark_arb_mat.py`
- `benchmark_bool_mat.py`
- `benchmark_matrix_stack_diagnostics.py`
- `benchmark_matrix_suite.py`
- `compare_acb_mat.py`
- `compare_arb_mat.py`
- `compare_bool_mat.py`

### Matrix Dense

- `benchmark_dense_matrix_surface.py`

### Matrix Sparse

- `benchmark_block_sparse_matrix_surface.py`
- `benchmark_sparse_matrix_surface.py`
- `benchmark_vblock_sparse_matrix_surface.py`

### Matrix Free

- `benchmark_matrix_free_krylov.py`

## Device Groups

### CPU

All current benchmark scripts default to CPU except the transform GPU-optional group.

### GPU Optional

- `benchmark_fft_nufft.py`
- `benchmark_nufft_backends.py`

## Notes

- The current transform group is narrower than the matrix group. It is mostly FFT/NUFFT-focused today.
- The current compile group has one official matrix compile benchmark and no scalar/special compile-specific benchmark yet.
- The current AD benchmark group is matrix-free focused today.
- PETSc/SLEPc remain comparison-only through `benchmark_matrix_backend_candidates.py`.
