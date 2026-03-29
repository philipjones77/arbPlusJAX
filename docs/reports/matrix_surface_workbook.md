Last updated: 2026-03-20T00:00:00Z

# Matrix Surface Workbook

This workbook summarizes the benchmark surfaces for dense, sparse, and matrix-free matrix families.

It is intended to make the current comparison surface legible while the matrix API and execution-strategy hardening continues.

The workbook now compares dense, sparse, block-sparse, variable-block sparse, and matrix-free/operator-plan execution in one place.

## Compare and Contrast

| family | use when | fastest metrics |
| --- | --- | --- |
| dense | matrices are small/medium and direct kernels or cached dense plans are appropriate | `acb_dense_plan_prepare_s`=1.162e-04s, `arb_dense_plan_prepare_s`=4.280e-04s, `arb_transpose_s`=5.970e-02s, `arb_diag_s`=7.211e-02s |
| sparse | storage sparsity is meaningful and callers want sparse cached matvec/rmatvec reuse | `srb_csr_point_cached_matvec_s`=2.350e-03s, `scb_csr_point_cached_prepare_s`=2.538e-03s, `srb_csr_point_cached_prepare_s`=3.005e-03s, `scb_csr_point_cached_matvec_s`=3.796e-03s |
| block sparse | block structure is explicit and callers want block-native apply paths | `scb_block_adjoint_cached_s`=5.394e-02s, `srb_block_rmatvec_cached_s`=8.018e-02s, `scb_block_matvec_s`=1.375e-01s, `srb_block_matvec_s`=2.154e-01s |
| variable block sparse | partitions are irregular but structure should still be preserved | `srb_vblock_matvec_cached_s`=1.378e-01s, `scb_vblock_matvec_cached_s`=1.421e-01s, `srb_vblock_matvec_s`=1.771e-01s, `scb_vblock_matvec_s`=2.588e-01s |
| matrix free | operator plans, Krylov solves, logdet, or adapter-based execution matter more than explicit materialization | `real_apply_plan_warm_s`=1.015e-05s, `real_rapply_plan_warm_s`=1.075e-05s, `complex_apply_plan_warm_s`=1.156e-05s, `complex_rapply_plan_warm_s`=1.174e-05s |

Recommended visualizations in the canonical notebooks:

- dense: direct solve vs cached matvec/rmatvec vs operator-plan apply
- sparse: sparse vs block-sparse vs vblock matvec/rmatvec
- matrix-free: dense-adapted vs sparse-adapted operator plans and solve/logdet slices

## Dense Matrix Surface

Command:

```bash
python benchmarks/benchmark_dense_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `acb_cached_matvec_padded_s` | 1.500115e+00 |
| `acb_cached_matvec_s` | 6.602075e-01 |
| `acb_conjugate_transpose_s` | 1.272419e-01 |
| `acb_dense_plan_prepare_s` | 1.161860e-04 |
| `acb_diag_s` | 8.774424e-02 |
| `acb_direct_solve_s` | 6.543078e-01 |
| `acb_lu_reuse_s` | 3.143238e-01 |
| `acb_transpose_s` | 8.220398e-02 |
| `arb_cached_matvec_padded_s` | 3.592728e+00 |
| `arb_cached_matvec_s` | 3.460538e-01 |
| `arb_dense_plan_prepare_s` | 4.279970e-04 |
| `arb_diag_s` | 7.210607e-02 |
| `arb_direct_solve_s` | 4.776618e-01 |
| `arb_lu_reuse_s` | 3.690385e-01 |
| `arb_transpose_s` | 5.970279e-02 |

## Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_csr_basic_cached_matvec_s` | 6.097758e-02 |
| `scb_csr_basic_cached_prepare_s` | 5.580408e-03 |
| `scb_csr_basic_hpd_prepare_s` | 2.377895e-02 |
| `scb_csr_basic_lu_prepare_s` | 1.870117e-02 |
| `scb_csr_basic_matvec_s` | 1.696886e-01 |
| `scb_csr_basic_storage_prepare_s` | 1.136808e-02 |
| `scb_csr_point_cached_matvec_s` | 3.796382e-03 |
| `scb_csr_point_cached_prepare_s` | 2.537931e-03 |
| `scb_csr_point_hpd_prepare_s` | 4.440331e-02 |
| `scb_csr_point_lu_prepare_s` | 1.486896e-02 |
| `scb_csr_point_matvec_s` | 1.800177e-01 |
| `scb_csr_point_storage_prepare_s` | 1.086163e-02 |
| `srb_csr_basic_cached_matvec_s` | 1.933018e-02 |
| `srb_csr_basic_cached_prepare_s` | 5.479311e-03 |
| `srb_csr_basic_lu_prepare_s` | 1.365347e-02 |
| `srb_csr_basic_matvec_s` | 1.604138e+00 |
| `srb_csr_basic_spd_prepare_s` | 8.900453e-03 |
| `srb_csr_basic_storage_prepare_s` | 8.500440e-03 |
| `srb_csr_point_cached_matvec_s` | 2.349868e-03 |
| `srb_csr_point_cached_prepare_s` | 3.005147e-03 |
| `srb_csr_point_lu_prepare_s` | 2.043781e-02 |
| `srb_csr_point_matvec_s` | 2.036784e-01 |
| `srb_csr_point_spd_prepare_s` | 3.205204e-02 |
| `srb_csr_point_storage_prepare_s` | 7.519066e-03 |

## Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_block_sparse_matrix_surface.py --n-blocks 2 --block-size 2 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_block_adjoint_cached_s` | 5.393833e-02 |
| `scb_block_matvec_s` | 1.374534e-01 |
| `srb_block_matvec_s` | 2.153765e-01 |
| `srb_block_rmatvec_cached_s` | 8.018189e-02 |

## Variable-Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_vblock_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_vblock_matvec_cached_s` | 1.420713e-01 |
| `scb_vblock_matvec_s` | 2.587504e-01 |
| `srb_vblock_matvec_cached_s` | 1.377720e-01 |
| `srb_vblock_matvec_s` | 1.770709e-01 |

## Matrix-Free Surface

Command:

```bash
python benchmarks/benchmark_matrix_free_krylov.py
```

| metric | seconds |
| --- | ---: |
| `complex_action_cold_s` | 4.087453e-01 |
| `complex_action_plan_cold_s` | 6.171976e-01 |
| `complex_action_plan_precompile_s` | 3.806190e-04 |
| `complex_action_plan_warm_s` | 2.750620e-05 |
| `complex_adjoint_apply_plan_cold_s` | 1.125367e-01 |
| `complex_adjoint_apply_plan_warm_s` | 1.181720e-05 |
| `complex_apply_cold_s` | 1.326142e-01 |
| `complex_apply_plan_cold_s` | 1.073635e-01 |
| `complex_apply_plan_precompile_s` | 1.154490e-04 |
| `complex_apply_plan_warm_s` | 1.155980e-05 |
| `complex_det_cold_s` | 5.037143e-01 |
| `complex_det_plan_cold_s` | 5.155169e-01 |
| `complex_det_plan_precompile_s` | 1.204925e-03 |
| `complex_eigsh_cold_s` | 4.396854e-01 |
| `complex_eigsh_plan_cold_s` | 5.263248e-01 |
| `complex_eigsh_restarted_plan_compile_s` | 7.982512e-01 |
| `complex_eigsh_restarted_plan_execute_s` | 6.096370e-04 |
| `complex_grad_cold_s` | 4.574597e-01 |
| `complex_inverse_action_cold_s` | 3.402833e-01 |
| `complex_inverse_action_plan_cold_s` | 4.206346e-01 |
| `complex_inverse_action_plan_compile_s` | 1.239250e-04 |
| `complex_inverse_action_plan_execute_s` | 2.652100e-05 |
| `complex_inverse_action_plan_warm_s` | 1.459180e-05 |
| `complex_inverse_grad_plan_compile_s` | 4.645596e-01 |
| `complex_inverse_grad_plan_execute_s` | 6.728800e-05 |
| `complex_logdet_cold_s` | 6.441892e-01 |
| `complex_logdet_grad_cold_s` | 1.861164e+00 |
| `complex_logdet_grad_compile_s` | 4.050930e-04 |
| `complex_logdet_grad_execute_s` | 1.705060e-04 |
| `complex_logdet_plan_cold_s` | 4.676006e-01 |
| `complex_logdet_plan_precompile_s` | 1.449054e-03 |
| `complex_logdet_plan_warm_s` | 3.479540e-04 |
| `complex_logdet_solve_plan_cold_s` | 4.620355e+00 |
| `complex_logdet_solve_plan_compile_s` | 1.015686e+00 |
| `complex_logdet_solve_plan_execute_s` | 9.632797e-01 |
| `complex_logdet_solve_plan_precompile_s` | 1.110559e+00 |
| `complex_logdet_solve_plan_warm_s` | 8.917330e-01 |
| `complex_minres_plan_cold_s` | 7.428522e-01 |
| `complex_minres_plan_compile_s` | 4.763340e-04 |
| `complex_minres_plan_execute_s` | 1.730850e-04 |
| `complex_multi_shift_plan_compile_s` | 4.636305e-01 |
| `complex_multi_shift_plan_execute_s` | 1.849820e-04 |
| `complex_multi_shift_plan_warm_s` | 2.278500e-05 |
| `complex_rapply_plan_cold_s` | 1.513076e-01 |
| `complex_rapply_plan_warm_s` | 1.173660e-05 |
| `complex_restarted_action_cold_s` | 3.340723e-01 |
| `complex_restarted_action_plan_cold_s` | 3.973997e-01 |
| `complex_solve_action_cold_s` | 2.884748e-01 |
| `complex_solve_action_plan_cold_s` | 2.584666e-01 |
| `complex_solve_action_plan_compile_s` | 1.869710e-04 |
| `complex_solve_action_plan_execute_s` | 3.644600e-05 |
| `complex_solve_action_plan_warm_s` | 1.524020e-05 |
| `complex_solve_grad_plan_compile_s` | 5.857882e-01 |
| `complex_solve_grad_plan_execute_s` | 7.496900e-05 |
| `real_action_cold_s` | 6.343553e-01 |
| `real_action_plan_cold_s` | 4.435254e-01 |
| `real_action_plan_precompile_s` | 4.644680e-04 |
| `real_action_plan_warm_s` | 1.061950e-04 |
| `real_action_warm_s` | 7.826740e-05 |
| `real_apply_cold_s` | 1.117790e-01 |
| `real_apply_plan_cold_s` | 1.369601e-01 |
| `real_apply_plan_precompile_s` | 1.460820e-04 |
| `real_apply_plan_warm_s` | 1.014880e-05 |
| `real_apply_warm_s` | 1.175140e-05 |
| `real_det_cold_s` | 4.012137e-01 |
| `real_det_plan_cold_s` | 5.024946e-01 |
| `real_det_plan_precompile_s` | 4.092900e-04 |
| `real_eigsh_cold_s` | 2.532812e-01 |
| `real_eigsh_plan_cold_s` | 5.568796e-01 |
| `real_eigsh_restarted_plan_compile_s` | 5.875271e-01 |
| `real_eigsh_restarted_plan_execute_s` | 4.249940e-04 |
| `real_grad_cold_s` | 3.334576e-01 |
| `real_inverse_action_cold_s` | 2.836060e-01 |
| `real_inverse_action_plan_cold_s` | 2.442914e-01 |
| `real_inverse_action_plan_compile_s` | 1.977760e-04 |
| `real_inverse_action_plan_execute_s` | 3.205000e-05 |
| `real_inverse_action_plan_warm_s` | 1.782880e-05 |
| `real_inverse_grad_plan_compile_s` | 3.714198e-01 |
| `real_inverse_grad_plan_execute_s` | 6.640500e-05 |
| `real_logdet_cold_s` | 3.902084e-01 |
| `real_logdet_grad_cold_s` | 6.310755e-01 |
| `real_logdet_grad_compile_s` | 3.144100e-04 |
| `real_logdet_grad_execute_s` | 2.701040e-04 |
| `real_logdet_plan_cold_s` | 3.283926e-01 |
| `real_logdet_plan_precompile_s` | 6.496050e-04 |
| `real_logdet_plan_warm_s` | 9.594700e-05 |
| `real_logdet_solve_plan_cold_s` | 4.926701e+00 |
| `real_logdet_solve_plan_compile_s` | 8.251618e-01 |
| `real_logdet_solve_plan_execute_s` | 7.045489e-01 |
| `real_logdet_solve_plan_precompile_s` | 9.302337e-01 |
| `real_logdet_solve_plan_warm_s` | 6.324446e-01 |
| `real_logdet_warm_s` | 1.243160e-04 |
| `real_minres_plan_cold_s` | 5.628214e-01 |
| `real_minres_plan_compile_s` | 4.718600e-04 |
| `real_minres_plan_execute_s` | 2.159250e-04 |
| `real_multi_shift_plan_compile_s` | 3.424579e-01 |
| `real_multi_shift_plan_execute_s` | 2.960760e-04 |
| `real_multi_shift_plan_warm_s` | 2.588640e-05 |
| `real_rapply_plan_cold_s` | 1.270792e-01 |
| `real_rapply_plan_warm_s` | 1.075340e-05 |
| `real_restarted_action_cold_s` | 5.436374e-01 |
| `real_restarted_action_plan_cold_s` | 3.164798e-01 |
| `real_solve_action_cold_s` | 3.073523e-01 |
| `real_solve_action_plan_cold_s` | 3.120060e-01 |
| `real_solve_action_plan_compile_s` | 2.424280e-04 |
| `real_solve_action_plan_execute_s` | 4.814300e-05 |
| `real_solve_action_plan_warm_s` | 2.225020e-05 |
| `real_solve_grad_plan_compile_s` | 3.380409e-01 |
| `real_solve_grad_plan_execute_s` | 1.251340e-04 |
| `sparse_complex_action_plan_s` | 4.482644e-01 |
| `sparse_complex_action_s` | 4.575913e-01 |
| `sparse_complex_adjoint_apply_plan_s` | 1.298064e-01 |
| `sparse_complex_apply_plan_s` | 1.416069e-01 |
| `sparse_complex_apply_s` | 1.461259e-01 |
| `sparse_complex_det_plan_s` | 6.573357e-01 |
| `sparse_complex_det_s` | 7.081657e-01 |
| `sparse_complex_inverse_action_plan_s` | 3.633112e-01 |
| `sparse_complex_inverse_action_s` | 4.207348e-01 |
| `sparse_complex_logdet_plan_s` | 7.077795e-01 |
| `sparse_complex_logdet_s` | 1.676192e+00 |
| `sparse_complex_rapply_plan_s` | 1.715021e-01 |
| `sparse_complex_restarted_plan_s` | 5.077016e-01 |
| `sparse_complex_restarted_s` | 4.434487e-01 |
| `sparse_complex_solve_action_plan_s` | 3.022498e-01 |
| `sparse_complex_solve_action_s` | 3.800733e-01 |
| `sparse_real_apply_plan_s` | 1.389499e-01 |
| `sparse_real_apply_s` | 1.062244e-01 |
| `sparse_real_det_plan_s` | 4.079756e-01 |
| `sparse_real_det_s` | 3.770228e-01 |
| `sparse_real_inverse_action_plan_s` | 2.874505e-01 |
| `sparse_real_inverse_action_s` | 3.063118e-01 |
| `sparse_real_inverse_diag_corrected_s` | 1.691698e+00 |
| `sparse_real_inverse_diag_local_s` | 2.345163e-01 |
| `sparse_real_logdet_grad_s` | 5.689103e-01 |
| `sparse_real_logdet_leja_hutchpp_auto_s` | 2.112370e+00 |
| `sparse_real_logdet_leja_hutchpp_s` | 1.585879e+00 |
| `sparse_real_logdet_plan_s` | 5.475732e-01 |
| `sparse_real_logdet_s` | 4.540330e-01 |
| `sparse_real_rapply_plan_s` | 1.020325e-01 |
| `sparse_real_solve_action_plan_s` | 2.288292e-01 |
| `sparse_real_solve_action_s` | 2.771373e-01 |

## Matrix Backend Candidates

Command:

```bash
python benchmarks/benchmark_matrix_backend_candidates.py --n 4 --warmup 0 --runs 1
```

| metric | seconds |
| --- | ---: |
| `candidate_arbplusjax_sparse_cached_matvec_s` | 1.687203e-03 |
| `candidate_arbplusjax_sparse_fromdense_solve_s` | 8.131491e-01 |
| `candidate_arbplusjax_sparse_matvec_s` | 2.320852e-03 |
| `candidate_arbplusjax_sparse_spd_solve_s` | 4.661007e-01 |
| `candidate_jax_dense_eigh_s` | 1.079195e-01 |
| `candidate_jax_dense_matvec_s` | 6.813881e-02 |
| `candidate_jax_dense_solve_s` | 1.770659e-01 |
| `candidate_jax_experimental_sparse_cg_s` | 2.653300e-01 |
| `candidate_jax_experimental_sparse_matvec_s` | 1.069917e-01 |
| `candidate_jax_scipy_dense_solve_s` | 1.338760e-01 |
| `candidate_matfree_apply_s` | 2.711300e-03 |
| `candidate_matfree_logdet_slq_s` | 5.404283e-01 |
| `candidate_matfree_solve_action_s` | 1.902730e-04 |
| `candidate_petsc_available` | 0.000000e+00 |
| `candidate_scipy_dense_eigh_s` | 6.477149e-03 |
| `candidate_scipy_dense_matvec_s` | 1.277868e-03 |
| `candidate_scipy_dense_solve_s` | 4.714034e-03 |
| `candidate_scipy_linear_operator_cg_s` | 1.927580e-04 |
| `candidate_scipy_sparse_cg_s` | 1.168562e-03 |
| `candidate_scipy_sparse_eigsh_s` | 5.747787e-03 |
| `candidate_scipy_sparse_matvec_s` | 4.219144e-03 |
| `candidate_slepc_available` | 0.000000e+00 |
