Last updated: 2026-03-20T00:00:00Z

# Matrix Surface Workbook

This workbook summarizes the benchmark surfaces for dense, sparse, and matrix-free matrix families.

It is intended to make the current comparison surface legible while the matrix API and execution-strategy hardening continues.

The workbook now compares dense, sparse, block-sparse, variable-block sparse, and matrix-free/operator-plan execution in one place.

## Compare and Contrast

| family | use when | fastest metrics |
| --- | --- | --- |
| dense | matrices are small/medium and direct kernels or cached dense plans are appropriate | `arb_dense_plan_prepare_s`=1.486e-04s, `acb_dense_plan_prepare_s`=1.530e-04s, `acb_diag_s`=3.386e-02s, `acb_transpose_s`=4.365e-02s |
| sparse | storage sparsity is meaningful and callers want sparse cached matvec/rmatvec reuse | `srb_csr_point_cached_matvec_s`=1.611e-03s, `srb_coo_point_cached_matvec_s`=1.801e-03s, `scb_bcoo_point_cached_matvec_s`=2.910e-03s, `scb_csr_point_cached_matvec_s`=2.942e-03s |
| block sparse | block structure is explicit and callers want block-native apply paths | `srb_block_rmatvec_cached_s`=5.847e-02s, `scb_block_adjoint_cached_s`=8.494e-02s, `scb_block_matvec_s`=1.186e-01s, `srb_block_matvec_s`=3.869e-01s |
| variable block sparse | partitions are irregular but structure should still be preserved | `srb_vblock_matvec_cached_s`=1.086e-01s, `scb_vblock_matvec_cached_s`=1.320e-01s, `scb_vblock_matvec_s`=1.802e-01s, `srb_vblock_matvec_s`=1.928e-01s |
| matrix free | operator plans, Krylov solves, logdet, or adapter-based execution matter more than explicit materialization | `real_apply_plan_warm_s`=1.800e-05s, `real_apply_warm_s`=1.895e-05s, `complex_inverse_action_plan_warm_s`=2.060e-05s, `real_inverse_action_plan_warm_s`=2.102e-05s |

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
| `acb_cached_matvec_padded_s` | 6.743075e-01 |
| `acb_cached_matvec_s` | 1.079694e+00 |
| `acb_conjugate_transpose_s` | 4.503404e-02 |
| `acb_dense_plan_prepare_s` | 1.529520e-04 |
| `acb_diag_s` | 3.385952e-02 |
| `acb_direct_solve_s` | 1.032805e+00 |
| `acb_lu_reuse_s` | 6.588473e-01 |
| `acb_transpose_s` | 4.364825e-02 |
| `arb_cached_matvec_padded_s` | 2.620439e+00 |
| `arb_cached_matvec_s` | 1.474046e-01 |
| `arb_dense_plan_prepare_s` | 1.486310e-04 |
| `arb_diag_s` | 1.134494e-01 |
| `arb_direct_solve_s` | 1.997472e-01 |
| `arb_lu_reuse_s` | 1.287702e-01 |
| `arb_transpose_s` | 1.286523e-01 |

## Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_bcoo_basic_cached_matvec_s` | 4.720946e-02 |
| `scb_bcoo_basic_matvec_s` | 5.879287e-02 |
| `scb_bcoo_point_cached_matvec_s` | 2.909948e-03 |
| `scb_bcoo_point_matvec_s` | 3.424872e-03 |
| `scb_coo_basic_cached_matvec_s` | 5.504423e-02 |
| `scb_coo_basic_matvec_s` | 2.315646e-01 |
| `scb_coo_point_cached_matvec_s` | 2.962013e-03 |
| `scb_coo_point_matvec_s` | 4.911929e-01 |
| `scb_csr_basic_cached_matvec_s` | 6.387502e-02 |
| `scb_csr_basic_matvec_s` | 6.007980e-02 |
| `scb_csr_point_cached_matvec_s` | 2.941541e-03 |
| `scb_csr_point_matvec_s` | 7.409267e-03 |
| `srb_bcoo_basic_cached_matvec_s` | 2.979044e-02 |
| `srb_bcoo_basic_matvec_s` | 1.015864e-01 |
| `srb_bcoo_point_cached_matvec_s` | 3.487705e-03 |
| `srb_bcoo_point_matvec_s` | 1.195561e-02 |
| `srb_coo_basic_cached_matvec_s` | 5.213956e-03 |
| `srb_coo_basic_matvec_s` | 8.915470e-01 |
| `srb_coo_point_cached_matvec_s` | 1.801093e-03 |
| `srb_coo_point_matvec_s` | 1.480472e-01 |
| `srb_csr_basic_cached_matvec_s` | 8.382825e-03 |
| `srb_csr_basic_matvec_s` | 1.199771e-02 |
| `srb_csr_point_cached_matvec_s` | 1.611439e-03 |
| `srb_csr_point_matvec_s` | 5.016736e-03 |

## Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_block_sparse_matrix_surface.py --n-blocks 2 --block-size 2 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_block_adjoint_cached_s` | 8.493559e-02 |
| `scb_block_matvec_s` | 1.186069e-01 |
| `srb_block_matvec_s` | 3.869023e-01 |
| `srb_block_rmatvec_cached_s` | 5.847189e-02 |

## Variable-Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_vblock_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_vblock_matvec_cached_s` | 1.319967e-01 |
| `scb_vblock_matvec_s` | 1.801849e-01 |
| `srb_vblock_matvec_cached_s` | 1.086495e-01 |
| `srb_vblock_matvec_s` | 1.927985e-01 |

## Matrix-Free Surface

Command:

```bash
python benchmarks/benchmark_matrix_free_krylov.py
```

| metric | seconds |
| --- | ---: |
| `complex_action_cold_s` | 1.285480e+00 |
| `complex_action_plan_cold_s` | 1.170284e+00 |
| `complex_action_plan_precompile_s` | 1.818698e-03 |
| `complex_action_plan_warm_s` | 3.800682e-04 |
| `complex_apply_cold_s` | 3.172320e-01 |
| `complex_apply_plan_cold_s` | 3.784899e-01 |
| `complex_apply_plan_precompile_s` | 5.738480e-04 |
| `complex_apply_plan_warm_s` | 5.925860e-05 |
| `complex_det_cold_s` | 6.571954e-01 |
| `complex_det_plan_cold_s` | 1.067300e+00 |
| `complex_det_plan_precompile_s` | 1.640405e-03 |
| `complex_eigsh_cold_s` | 8.427846e-01 |
| `complex_eigsh_plan_cold_s` | 7.921426e-01 |
| `complex_eigsh_restarted_plan_compile_s` | 6.909143e-01 |
| `complex_eigsh_restarted_plan_execute_s` | 2.682000e-04 |
| `complex_grad_cold_s` | 7.073837e-01 |
| `complex_inverse_action_cold_s` | 8.873934e-01 |
| `complex_inverse_action_plan_cold_s` | 7.586897e-01 |
| `complex_inverse_action_plan_compile_s` | 7.929400e-05 |
| `complex_inverse_action_plan_execute_s` | 4.599800e-05 |
| `complex_inverse_action_plan_warm_s` | 2.059860e-05 |
| `complex_inverse_grad_plan_compile_s` | 3.336192e+00 |
| `complex_inverse_grad_plan_execute_s` | 2.924040e-04 |
| `complex_logdet_cold_s` | 8.389488e-01 |
| `complex_logdet_grad_cold_s` | 2.679248e+00 |
| `complex_logdet_grad_compile_s` | 6.064823e-03 |
| `complex_logdet_grad_execute_s` | 4.552210e-04 |
| `complex_logdet_plan_cold_s` | 4.602657e-01 |
| `complex_logdet_plan_precompile_s` | 5.045660e-03 |
| `complex_logdet_plan_warm_s` | 2.185052e-04 |
| `complex_minres_plan_cold_s` | 1.115692e+00 |
| `complex_minres_plan_compile_s` | 3.062660e-04 |
| `complex_minres_plan_execute_s` | 1.603740e-04 |
| `complex_multi_shift_plan_compile_s` | 8.755410e-01 |
| `complex_multi_shift_plan_execute_s` | 1.975380e-04 |
| `complex_multi_shift_plan_warm_s` | 3.636340e-05 |
| `complex_restarted_action_cold_s` | 1.060396e+00 |
| `complex_restarted_action_plan_cold_s` | 1.339841e+00 |
| `complex_solve_action_cold_s` | 5.699726e-01 |
| `complex_solve_action_plan_cold_s` | 7.658547e-01 |
| `complex_solve_action_plan_compile_s` | 1.521690e-04 |
| `complex_solve_action_plan_execute_s` | 5.533200e-05 |
| `complex_solve_action_plan_warm_s` | 2.177780e-05 |
| `complex_solve_grad_plan_compile_s` | 1.285117e+00 |
| `complex_solve_grad_plan_execute_s` | 1.822950e-04 |
| `real_action_cold_s` | 1.016479e+00 |
| `real_action_plan_cold_s` | 8.989407e-01 |
| `real_action_plan_precompile_s` | 1.898074e-03 |
| `real_action_plan_warm_s` | 6.496746e-04 |
| `real_action_warm_s` | 3.641160e-05 |
| `real_apply_cold_s` | 4.084621e-01 |
| `real_apply_plan_cold_s` | 4.554639e-01 |
| `real_apply_plan_precompile_s` | 2.209940e-04 |
| `real_apply_plan_warm_s` | 1.800000e-05 |
| `real_apply_warm_s` | 1.895340e-05 |
| `real_det_cold_s` | 8.319097e-01 |
| `real_det_plan_cold_s` | 8.451516e-01 |
| `real_det_plan_precompile_s` | 5.625993e-03 |
| `real_eigsh_cold_s` | 2.696494e-01 |
| `real_eigsh_plan_cold_s` | 2.417835e-01 |
| `real_eigsh_restarted_plan_compile_s` | 1.226006e+00 |
| `real_eigsh_restarted_plan_execute_s` | 6.135100e-04 |
| `real_grad_cold_s` | 1.151843e+00 |
| `real_inverse_action_cold_s` | 1.422943e-01 |
| `real_inverse_action_plan_cold_s` | 1.835770e-01 |
| `real_inverse_action_plan_compile_s` | 3.033520e-04 |
| `real_inverse_action_plan_execute_s` | 1.135960e-04 |
| `real_inverse_action_plan_warm_s` | 2.101820e-05 |
| `real_inverse_grad_plan_compile_s` | 1.687403e+00 |
| `real_inverse_grad_plan_execute_s` | 8.893600e-05 |
| `real_logdet_cold_s` | 8.511046e-01 |
| `real_logdet_grad_cold_s` | 3.702061e-01 |
| `real_logdet_grad_compile_s` | 2.063330e-04 |
| `real_logdet_grad_execute_s` | 2.671022e-03 |
| `real_logdet_plan_cold_s` | 7.280799e-01 |
| `real_logdet_plan_precompile_s` | 2.750550e-04 |
| `real_logdet_plan_warm_s` | 6.771280e-05 |
| `real_logdet_warm_s` | 4.430840e-05 |
| `real_minres_plan_cold_s` | 4.537163e-01 |
| `real_minres_plan_compile_s` | 2.091310e-04 |
| `real_minres_plan_execute_s` | 1.014210e-04 |
| `real_multi_shift_plan_compile_s` | 8.091046e-01 |
| `real_multi_shift_plan_execute_s` | 1.088470e-04 |
| `real_multi_shift_plan_warm_s` | 3.468620e-05 |
| `real_restarted_action_cold_s` | 1.039658e+00 |
| `real_restarted_action_plan_cold_s` | 8.921965e-01 |
| `real_solve_action_cold_s` | 5.491415e-01 |
| `real_solve_action_plan_cold_s` | 1.434593e-01 |
| `real_solve_action_plan_compile_s` | 7.664520e-04 |
| `real_solve_action_plan_execute_s` | 3.333770e-04 |
| `real_solve_action_plan_warm_s` | 1.025618e-04 |
| `real_solve_grad_plan_compile_s` | 1.856558e+00 |
| `real_solve_grad_plan_execute_s` | 8.168300e-05 |
| `sparse_complex_action_plan_s` | 9.962816e-01 |
| `sparse_complex_action_s` | 1.140329e+00 |
| `sparse_complex_apply_plan_s` | 4.469855e-01 |
| `sparse_complex_apply_s` | 4.127720e-01 |
| `sparse_complex_det_plan_s` | 5.677815e-01 |
| `sparse_complex_det_s` | 5.139312e-01 |
| `sparse_complex_inverse_action_plan_s` | 6.771838e-01 |
| `sparse_complex_inverse_action_s` | 7.413090e-01 |
| `sparse_complex_logdet_plan_s` | 4.324160e-01 |
| `sparse_complex_logdet_s` | 8.590080e-01 |
| `sparse_complex_restarted_plan_s` | 1.111584e+00 |
| `sparse_complex_restarted_s` | 1.091905e+00 |
| `sparse_complex_solve_action_plan_s` | 6.274333e-01 |
| `sparse_complex_solve_action_s` | 2.194730e-01 |
| `sparse_real_apply_plan_s` | 3.355031e-01 |
| `sparse_real_apply_s` | 3.789774e-01 |
| `sparse_real_det_plan_s` | 1.224565e+00 |
| `sparse_real_det_s` | 1.176319e+00 |
| `sparse_real_inverse_action_plan_s` | 4.425915e-01 |
| `sparse_real_inverse_action_s` | 4.278625e-01 |
| `sparse_real_inverse_diag_corrected_s` | 2.911636e+00 |
| `sparse_real_inverse_diag_local_s` | 5.053176e-01 |
| `sparse_real_logdet_grad_s` | 1.534757e+00 |
| `sparse_real_logdet_leja_hutchpp_auto_s` | 4.152162e+00 |
| `sparse_real_logdet_leja_hutchpp_s` | 1.100051e+00 |
| `sparse_real_logdet_plan_s` | 7.018206e-01 |
| `sparse_real_logdet_s` | 1.270319e+00 |
| `sparse_real_solve_action_plan_s` | 5.336904e-01 |
| `sparse_real_solve_action_s` | 5.502056e-01 |

## Matrix Backend Candidates

Command:

```bash
python benchmarks/benchmark_matrix_backend_candidates.py --n 4 --warmup 0 --runs 1
```

| metric | seconds |
| --- | ---: |
| `candidate_arbplusjax_sparse_cached_matvec_s` | 2.553915e-03 |
| `candidate_arbplusjax_sparse_fromdense_solve_s` | 1.556503e+00 |
| `candidate_arbplusjax_sparse_matvec_s` | 7.322914e-03 |
| `candidate_arbplusjax_sparse_spd_solve_s` | 1.188982e+00 |
| `candidate_jax_dense_eigh_s` | 2.470059e-01 |
| `candidate_jax_dense_matvec_s` | 9.312155e-02 |
| `candidate_jax_dense_solve_s` | 4.935775e-01 |
| `candidate_jax_experimental_sparse_cg_s` | 4.689772e-01 |
| `candidate_jax_experimental_sparse_matvec_s` | 2.955479e-01 |
| `candidate_jax_scipy_dense_solve_s` | 3.410598e-01 |
| `candidate_matfree_apply_s` | 4.825568e-03 |
| `candidate_matfree_logdet_slq_s` | 3.745362e-01 |
| `candidate_matfree_solve_action_s` | 4.499720e-04 |
| `candidate_petsc_available` | 0.000000e+00 |
| `candidate_scipy_dense_eigh_s` | 5.077289e-03 |
| `candidate_scipy_dense_matvec_s` | 9.796600e-05 |
| `candidate_scipy_dense_solve_s` | 5.045930e-03 |
| `candidate_scipy_linear_operator_cg_s` | 1.181300e-04 |
| `candidate_scipy_sparse_cg_s` | 8.906810e-04 |
| `candidate_scipy_sparse_eigsh_s` | 1.385147e-03 |
| `candidate_scipy_sparse_matvec_s` | 2.505506e-03 |
| `candidate_slepc_available` | 0.000000e+00 |
