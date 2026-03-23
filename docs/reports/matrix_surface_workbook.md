Last updated: 2026-03-20T00:00:00Z

# Matrix Surface Workbook

This workbook summarizes the benchmark surfaces for dense, sparse, and matrix-free matrix families.

It is intended to make the current comparison surface legible while the matrix API and execution-strategy hardening continues.

The workbook now compares dense, sparse, block-sparse, variable-block sparse, and matrix-free/operator-plan execution in one place.

## Compare and Contrast

| family | use when | fastest metrics |
| --- | --- | --- |
| dense | matrices are small/medium and direct kernels or cached dense plans are appropriate | `acb_dense_plan_prepare_s`=8.828e-05s, `arb_dense_plan_prepare_s`=3.448e-04s, `acb_diag_s`=2.493e-02s, `arb_diag_s`=2.806e-02s |
| sparse | storage sparsity is meaningful and callers want sparse cached matvec/rmatvec reuse | `srb_bcoo_point_cached_matvec_s`=8.478e-04s, `srb_csr_point_cached_matvec_s`=8.556e-04s, `srb_bcoo_point_matvec_s`=9.815e-04s, `scb_csr_point_cached_matvec_s`=1.024e-03s |
| block sparse | block structure is explicit and callers want block-native apply paths | `scb_block_adjoint_cached_s`=5.435e-02s, `srb_block_rmatvec_cached_s`=1.041e-01s, `srb_block_matvec_s`=1.741e-01s, `scb_block_matvec_s`=1.872e-01s |
| variable block sparse | partitions are irregular but structure should still be preserved | `scb_vblock_matvec_cached_s`=9.296e-02s, `scb_vblock_matvec_s`=1.394e-01s, `srb_vblock_matvec_cached_s`=1.478e-01s, `srb_vblock_matvec_s`=1.970e-01s |
| matrix free | operator plans, Krylov solves, logdet, or adapter-based execution matter more than explicit materialization | `real_apply_plan_warm_s`=6.312e-06s, `real_apply_warm_s`=6.515e-06s, `complex_apply_plan_warm_s`=8.351e-06s, `real_solve_action_plan_warm_s`=1.002e-05s |

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
| `acb_cached_matvec_padded_s` | 5.146898e-01 |
| `acb_cached_matvec_s` | 1.939264e-01 |
| `acb_conjugate_transpose_s` | 3.464039e-02 |
| `acb_dense_plan_prepare_s` | 8.827800e-05 |
| `acb_diag_s` | 2.493387e-02 |
| `acb_direct_solve_s` | 2.061470e-01 |
| `acb_lu_reuse_s` | 1.308222e-01 |
| `acb_transpose_s` | 3.311947e-02 |
| `arb_cached_matvec_padded_s` | 1.861919e+00 |
| `arb_cached_matvec_s` | 2.005157e-01 |
| `arb_dense_plan_prepare_s` | 3.447540e-04 |
| `arb_diag_s` | 2.805763e-02 |
| `arb_direct_solve_s` | 3.378641e-01 |
| `arb_lu_reuse_s` | 1.709141e-01 |
| `arb_transpose_s` | 2.822199e-02 |

## Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_bcoo_basic_cached_matvec_s` | 1.504881e-02 |
| `scb_bcoo_basic_matvec_s` | 1.631156e-02 |
| `scb_bcoo_point_cached_matvec_s` | 1.052623e-03 |
| `scb_bcoo_point_matvec_s` | 1.191802e-03 |
| `scb_coo_basic_cached_matvec_s` | 2.342180e-02 |
| `scb_coo_basic_matvec_s` | 1.306049e-01 |
| `scb_coo_point_cached_matvec_s` | 1.378967e-03 |
| `scb_coo_point_matvec_s` | 1.052524e-01 |
| `scb_csr_basic_cached_matvec_s` | 2.479492e-02 |
| `scb_csr_basic_matvec_s` | 4.576900e-02 |
| `scb_csr_point_cached_matvec_s` | 1.024294e-03 |
| `scb_csr_point_matvec_s` | 3.002426e-03 |
| `srb_bcoo_basic_cached_matvec_s` | 4.377832e-03 |
| `srb_bcoo_basic_matvec_s` | 4.586999e-03 |
| `srb_bcoo_point_cached_matvec_s` | 8.477500e-04 |
| `srb_bcoo_point_matvec_s` | 9.814970e-04 |
| `srb_coo_basic_cached_matvec_s` | 1.177221e-02 |
| `srb_coo_basic_matvec_s` | 8.509185e-01 |
| `srb_coo_point_cached_matvec_s` | 1.321769e-03 |
| `srb_coo_point_matvec_s` | 8.192436e-02 |
| `srb_csr_basic_cached_matvec_s` | 5.537944e-03 |
| `srb_csr_basic_matvec_s` | 9.641629e-03 |
| `srb_csr_point_cached_matvec_s` | 8.555530e-04 |
| `srb_csr_point_matvec_s` | 2.708920e-03 |

## Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_block_sparse_matrix_surface.py --n-blocks 2 --block-size 2 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_block_adjoint_cached_s` | 5.435336e-02 |
| `scb_block_matvec_s` | 1.871517e-01 |
| `srb_block_matvec_s` | 1.740984e-01 |
| `srb_block_rmatvec_cached_s` | 1.041051e-01 |

## Variable-Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_vblock_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_vblock_matvec_cached_s` | 9.296039e-02 |
| `scb_vblock_matvec_s` | 1.393804e-01 |
| `srb_vblock_matvec_cached_s` | 1.478031e-01 |
| `srb_vblock_matvec_s` | 1.969581e-01 |

## Matrix-Free Surface

Command:

```bash
python benchmarks/benchmark_matrix_free_krylov.py
```

| metric | seconds |
| --- | ---: |
| `complex_action_cold_s` | 3.884058e-01 |
| `complex_action_plan_cold_s` | 3.411443e-01 |
| `complex_action_plan_precompile_s` | 9.890900e-05 |
| `complex_action_plan_warm_s` | 2.779400e-05 |
| `complex_apply_cold_s` | 9.151797e-02 |
| `complex_apply_plan_cold_s` | 9.530623e-02 |
| `complex_apply_plan_precompile_s` | 5.116900e-05 |
| `complex_apply_plan_warm_s` | 8.351400e-06 |
| `complex_det_cold_s` | 4.009148e-01 |
| `complex_det_plan_cold_s` | 3.542324e-01 |
| `complex_det_plan_precompile_s` | 2.917990e-04 |
| `complex_eigsh_cold_s` | 1.919688e-01 |
| `complex_eigsh_plan_cold_s` | 2.029076e-01 |
| `complex_eigsh_restarted_plan_compile_s` | 4.289552e-01 |
| `complex_eigsh_restarted_plan_execute_s` | 2.491070e-04 |
| `complex_grad_cold_s` | 3.237500e-01 |
| `complex_inverse_action_cold_s` | 1.946775e-01 |
| `complex_inverse_action_plan_cold_s` | 1.649273e-01 |
| `complex_inverse_action_plan_compile_s` | 3.278700e-05 |
| `complex_inverse_action_plan_execute_s` | 1.135300e-05 |
| `complex_inverse_action_plan_warm_s` | 1.112080e-05 |
| `complex_inverse_grad_plan_compile_s` | 9.386834e-01 |
| `complex_inverse_grad_plan_execute_s` | 5.792200e-05 |
| `complex_logdet_cold_s` | 6.091455e-01 |
| `complex_logdet_grad_cold_s` | 8.124401e-01 |
| `complex_logdet_grad_compile_s` | 2.968400e-04 |
| `complex_logdet_grad_execute_s` | 2.769320e-04 |
| `complex_logdet_plan_cold_s` | 4.062078e-01 |
| `complex_logdet_plan_precompile_s` | 2.813010e-04 |
| `complex_logdet_plan_warm_s` | 1.970754e-04 |
| `complex_minres_plan_cold_s` | 2.538205e-01 |
| `complex_minres_plan_compile_s` | 2.287990e-04 |
| `complex_minres_plan_execute_s` | 1.820950e-04 |
| `complex_multi_shift_plan_compile_s` | 2.304086e-01 |
| `complex_multi_shift_plan_execute_s` | 6.885900e-05 |
| `complex_multi_shift_plan_warm_s` | 1.620160e-05 |
| `complex_restarted_action_cold_s` | 2.875868e-01 |
| `complex_restarted_action_plan_cold_s` | 3.326709e-01 |
| `complex_solve_action_cold_s` | 1.574561e-01 |
| `complex_solve_action_plan_cold_s` | 1.602728e-01 |
| `complex_solve_action_plan_compile_s` | 9.878100e-05 |
| `complex_solve_action_plan_execute_s` | 2.111100e-05 |
| `complex_solve_action_plan_warm_s` | 1.154840e-05 |
| `complex_solve_grad_plan_compile_s` | 8.184011e-01 |
| `complex_solve_grad_plan_execute_s` | 5.967600e-05 |
| `real_action_cold_s` | 3.401218e-01 |
| `real_action_plan_cold_s` | 2.086300e-01 |
| `real_action_plan_precompile_s` | 7.612800e-05 |
| `real_action_plan_warm_s` | 4.216560e-05 |
| `real_action_warm_s` | 4.544580e-05 |
| `real_apply_cold_s` | 7.637412e-02 |
| `real_apply_plan_cold_s` | 8.473924e-02 |
| `real_apply_plan_precompile_s` | 2.584900e-05 |
| `real_apply_plan_warm_s` | 6.312000e-06 |
| `real_apply_warm_s` | 6.515400e-06 |
| `real_det_cold_s` | 2.315384e-01 |
| `real_det_plan_cold_s` | 3.096650e-01 |
| `real_det_plan_precompile_s` | 8.617100e-05 |
| `real_eigsh_cold_s` | 3.669629e-01 |
| `real_eigsh_plan_cold_s` | 2.585503e-01 |
| `real_eigsh_restarted_plan_compile_s` | 2.743712e-01 |
| `real_eigsh_restarted_plan_execute_s` | 4.157430e-04 |
| `real_grad_cold_s` | 3.696152e-01 |
| `real_inverse_action_cold_s` | 1.408771e-01 |
| `real_inverse_action_plan_cold_s` | 2.042517e-01 |
| `real_inverse_action_plan_compile_s` | 3.874500e-05 |
| `real_inverse_action_plan_execute_s` | 1.331100e-05 |
| `real_inverse_action_plan_warm_s` | 1.054080e-05 |
| `real_inverse_grad_plan_compile_s` | 5.902881e-01 |
| `real_inverse_grad_plan_execute_s` | 4.815100e-05 |
| `real_logdet_cold_s` | 3.350975e-01 |
| `real_logdet_grad_cold_s` | 4.148136e-01 |
| `real_logdet_grad_compile_s` | 1.729940e-04 |
| `real_logdet_grad_execute_s` | 7.260400e-05 |
| `real_logdet_plan_cold_s` | 2.730634e-01 |
| `real_logdet_plan_precompile_s` | 1.507580e-04 |
| `real_logdet_plan_warm_s` | 6.560000e-05 |
| `real_logdet_warm_s` | 6.160260e-05 |
| `real_minres_plan_cold_s` | 2.733258e-01 |
| `real_minres_plan_compile_s` | 3.787160e-04 |
| `real_minres_plan_execute_s` | 2.769070e-04 |
| `real_multi_shift_plan_compile_s` | 1.916804e-01 |
| `real_multi_shift_plan_execute_s` | 1.030290e-04 |
| `real_multi_shift_plan_warm_s` | 1.504900e-05 |
| `real_restarted_action_cold_s` | 3.163186e-01 |
| `real_restarted_action_plan_cold_s` | 2.247907e-01 |
| `real_solve_action_cold_s` | 2.777106e-01 |
| `real_solve_action_plan_cold_s` | 2.113591e-01 |
| `real_solve_action_plan_compile_s` | 1.465230e-04 |
| `real_solve_action_plan_execute_s` | 3.752300e-05 |
| `real_solve_action_plan_warm_s` | 1.002180e-05 |
| `real_solve_grad_plan_compile_s` | 6.295960e-01 |
| `real_solve_grad_plan_execute_s` | 5.654100e-05 |
| `sparse_complex_action_plan_s` | 2.931183e-01 |
| `sparse_complex_action_s` | 2.794500e-01 |
| `sparse_complex_apply_plan_s` | 7.916854e-02 |
| `sparse_complex_apply_s` | 8.249440e-02 |
| `sparse_complex_det_plan_s` | 3.485588e-01 |
| `sparse_complex_det_s` | 4.202026e-01 |
| `sparse_complex_inverse_action_plan_s` | 2.277994e-01 |
| `sparse_complex_inverse_action_s` | 1.996503e-01 |
| `sparse_complex_logdet_plan_s` | 3.396399e-01 |
| `sparse_complex_logdet_s` | 4.428732e-01 |
| `sparse_complex_restarted_plan_s` | 2.960269e-01 |
| `sparse_complex_restarted_s` | 2.655255e-01 |
| `sparse_complex_solve_action_plan_s` | 2.194819e-01 |
| `sparse_complex_solve_action_s` | 2.312161e-01 |
| `sparse_real_apply_plan_s` | 6.380856e-02 |
| `sparse_real_apply_s` | 7.706972e-02 |
| `sparse_real_det_plan_s` | 3.255571e-01 |
| `sparse_real_det_s` | 3.686232e-01 |
| `sparse_real_inverse_action_plan_s` | 1.669780e-01 |
| `sparse_real_inverse_action_s` | 1.637841e-01 |
| `sparse_real_inverse_diag_corrected_s` | 1.378773e+00 |
| `sparse_real_inverse_diag_local_s` | 1.654376e-01 |
| `sparse_real_logdet_grad_s` | 5.864873e-01 |
| `sparse_real_logdet_leja_hutchpp_auto_s` | 1.522282e+00 |
| `sparse_real_logdet_leja_hutchpp_s` | 8.090853e-01 |
| `sparse_real_logdet_plan_s` | 2.626674e-01 |
| `sparse_real_logdet_s` | 3.198901e-01 |
| `sparse_real_solve_action_plan_s` | 1.862434e-01 |
| `sparse_real_solve_action_s` | 1.807798e-01 |

## Matrix Backend Candidates

Command:

```bash
python benchmarks/benchmark_matrix_backend_candidates.py --n 4 --warmup 0 --runs 1
```

| metric | seconds |
| --- | ---: |
| `candidate_arbplusjax_sparse_cached_matvec_s` | 1.628961e-03 |
| `candidate_arbplusjax_sparse_fromdense_solve_s` | 3.484293e-01 |
| `candidate_arbplusjax_sparse_matvec_s` | 2.425069e-03 |
| `candidate_arbplusjax_sparse_spd_solve_s` | 2.139934e-01 |
| `candidate_jax_dense_eigh_s` | 4.290183e-02 |
| `candidate_jax_dense_matvec_s` | 4.860500e-02 |
| `candidate_jax_dense_solve_s` | 1.106786e-01 |
| `candidate_jax_experimental_sparse_cg_s` | 1.304722e-01 |
| `candidate_jax_experimental_sparse_matvec_s` | 5.438793e-02 |
| `candidate_jax_scipy_dense_solve_s` | 6.698013e-02 |
| `candidate_matfree_apply_s` | 3.360019e-03 |
| `candidate_matfree_logdet_slq_s` | 2.772929e-01 |
| `candidate_matfree_solve_action_s` | 7.768700e-05 |
| `candidate_petsc_available` | 0.000000e+00 |
| `candidate_scipy_dense_eigh_s` | 2.893580e-04 |
| `candidate_scipy_dense_matvec_s` | 5.917200e-05 |
| `candidate_scipy_dense_solve_s` | 1.245956e-03 |
| `candidate_scipy_linear_operator_cg_s` | 1.516660e-04 |
| `candidate_scipy_sparse_cg_s` | 2.891030e-04 |
| `candidate_scipy_sparse_eigsh_s` | 2.250857e-03 |
| `candidate_scipy_sparse_matvec_s` | 9.907400e-05 |
| `candidate_slepc_available` | 0.000000e+00 |
