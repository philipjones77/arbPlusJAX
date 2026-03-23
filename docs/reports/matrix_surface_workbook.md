Last updated: 2026-03-20T00:00:00Z

# Matrix Surface Workbook

This workbook summarizes the benchmark surfaces for dense, sparse, and matrix-free matrix families.

It is intended to make the current comparison surface legible while the matrix API and execution-strategy hardening continues.

The workbook now compares dense, sparse, block-sparse, variable-block sparse, and matrix-free/operator-plan execution in one place.

## Compare and Contrast

| family | use when | fastest metrics |
| --- | --- | --- |
| dense | matrices are small/medium and direct kernels or cached dense plans are appropriate | `acb_dense_plan_prepare_s`=1.240e-04s, `arb_dense_plan_prepare_s`=2.226e-04s, `acb_diag_s`=3.562e-02s, `arb_diag_s`=3.648e-02s |
| sparse | storage sparsity is meaningful and callers want sparse cached matvec/rmatvec reuse | `scb_csr_point_cached_matvec_s`=9.086e-04s, `srb_csr_point_cached_matvec_s`=1.032e-03s, `scb_bcoo_point_cached_matvec_s`=1.090e-03s, `scb_bcoo_point_matvec_s`=1.278e-03s |
| block sparse | block structure is explicit and callers want block-native apply paths | `srb_block_rmatvec_cached_s`=6.487e-02s, `scb_block_adjoint_cached_s`=7.397e-02s, `scb_block_matvec_s`=1.650e-01s, `srb_block_matvec_s`=2.006e-01s |
| variable block sparse | partitions are irregular but structure should still be preserved | `scb_vblock_matvec_cached_s`=1.240e-01s, `srb_vblock_matvec_cached_s`=1.567e-01s, `scb_vblock_matvec_s`=2.110e-01s, `srb_vblock_matvec_s`=2.320e-01s |
| matrix free | operator plans, Krylov solves, logdet, or adapter-based execution matter more than explicit materialization | `real_apply_plan_warm_s`=8.923e-06s, `complex_inverse_action_plan_execute_s`=9.476e-06s, `real_apply_warm_s`=9.663e-06s, `real_inverse_action_plan_warm_s`=1.338e-05s |

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
| `acb_cached_matvec_padded_s` | 6.704204e-01 |
| `acb_cached_matvec_s` | 3.199395e-01 |
| `acb_conjugate_transpose_s` | 6.146854e-02 |
| `acb_dense_plan_prepare_s` | 1.239970e-04 |
| `acb_diag_s` | 3.562213e-02 |
| `acb_direct_solve_s` | 3.125810e-01 |
| `acb_lu_reuse_s` | 2.274354e-01 |
| `acb_transpose_s` | 7.138594e-02 |
| `arb_cached_matvec_padded_s` | 2.317676e+00 |
| `arb_cached_matvec_s` | 1.960772e-01 |
| `arb_dense_plan_prepare_s` | 2.225640e-04 |
| `arb_diag_s` | 3.648326e-02 |
| `arb_direct_solve_s` | 2.185177e-01 |
| `arb_lu_reuse_s` | 1.574163e-01 |
| `arb_transpose_s` | 4.153954e-02 |

## Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_bcoo_basic_cached_matvec_s` | 3.528848e-02 |
| `scb_bcoo_basic_matvec_s` | 4.365647e-02 |
| `scb_bcoo_point_cached_matvec_s` | 1.090091e-03 |
| `scb_bcoo_point_matvec_s` | 1.277706e-03 |
| `scb_coo_basic_cached_matvec_s` | 5.200838e-02 |
| `scb_coo_basic_matvec_s` | 1.313380e-01 |
| `scb_coo_point_cached_matvec_s` | 1.495995e-03 |
| `scb_coo_point_matvec_s` | 1.427029e-01 |
| `scb_csr_basic_cached_matvec_s` | 2.687658e-02 |
| `scb_csr_basic_matvec_s` | 2.891461e-02 |
| `scb_csr_point_cached_matvec_s` | 9.086030e-04 |
| `scb_csr_point_matvec_s` | 2.469113e-03 |
| `srb_bcoo_basic_cached_matvec_s` | 5.932081e-03 |
| `srb_bcoo_basic_matvec_s` | 7.167066e-03 |
| `srb_bcoo_point_cached_matvec_s` | 2.057630e-03 |
| `srb_bcoo_point_matvec_s` | 1.369316e-03 |
| `srb_coo_basic_cached_matvec_s` | 1.055633e-02 |
| `srb_coo_basic_matvec_s` | 9.653552e-01 |
| `srb_coo_point_cached_matvec_s` | 2.229768e-03 |
| `srb_coo_point_matvec_s` | 1.412715e-01 |
| `srb_csr_basic_cached_matvec_s` | 6.442008e-03 |
| `srb_csr_basic_matvec_s` | 1.239952e-02 |
| `srb_csr_point_cached_matvec_s` | 1.031966e-03 |
| `srb_csr_point_matvec_s` | 2.923125e-03 |

## Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_block_sparse_matrix_surface.py --n-blocks 2 --block-size 2 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_block_adjoint_cached_s` | 7.397016e-02 |
| `scb_block_matvec_s` | 1.649672e-01 |
| `srb_block_matvec_s` | 2.005757e-01 |
| `srb_block_rmatvec_cached_s` | 6.487436e-02 |

## Variable-Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_vblock_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_vblock_matvec_cached_s` | 1.239580e-01 |
| `scb_vblock_matvec_s` | 2.109943e-01 |
| `srb_vblock_matvec_cached_s` | 1.567416e-01 |
| `srb_vblock_matvec_s` | 2.320140e-01 |

## Matrix-Free Surface

Command:

```bash
python benchmarks/benchmark_matrix_free_krylov.py
```

| metric | seconds |
| --- | ---: |
| `complex_action_cold_s` | 4.983991e-01 |
| `complex_action_plan_cold_s` | 4.528985e-01 |
| `complex_action_plan_precompile_s` | 1.613920e-04 |
| `complex_action_plan_warm_s` | 2.269040e-05 |
| `complex_apply_cold_s` | 1.183211e-01 |
| `complex_apply_plan_cold_s` | 1.810144e-01 |
| `complex_apply_plan_precompile_s` | 9.235800e-05 |
| `complex_apply_plan_warm_s` | 1.856020e-05 |
| `complex_det_cold_s` | 6.586019e-01 |
| `complex_det_plan_cold_s` | 7.016476e-01 |
| `complex_det_plan_precompile_s` | 3.362720e-04 |
| `complex_eigsh_cold_s` | 4.371583e-01 |
| `complex_eigsh_plan_cold_s` | 4.354604e-01 |
| `complex_eigsh_restarted_plan_compile_s` | 6.342522e-01 |
| `complex_eigsh_restarted_plan_execute_s` | 3.805040e-04 |
| `complex_grad_cold_s` | 5.270967e-01 |
| `complex_inverse_action_cold_s` | 3.287102e-01 |
| `complex_inverse_action_plan_cold_s` | 2.831074e-01 |
| `complex_inverse_action_plan_compile_s` | 4.236600e-05 |
| `complex_inverse_action_plan_execute_s` | 9.476000e-06 |
| `complex_inverse_action_plan_warm_s` | 1.412200e-05 |
| `complex_inverse_grad_plan_compile_s` | 1.153372e+00 |
| `complex_inverse_grad_plan_execute_s` | 9.559500e-05 |
| `complex_logdet_cold_s` | 9.796256e-01 |
| `complex_logdet_grad_cold_s` | 1.120067e+00 |
| `complex_logdet_grad_compile_s` | 3.425780e-04 |
| `complex_logdet_grad_execute_s` | 3.362240e-04 |
| `complex_logdet_plan_cold_s` | 5.198483e-01 |
| `complex_logdet_plan_precompile_s` | 3.160670e-04 |
| `complex_logdet_plan_warm_s` | 2.021030e-04 |
| `complex_minres_plan_cold_s` | 5.149382e-01 |
| `complex_minres_plan_compile_s` | 2.443910e-04 |
| `complex_minres_plan_execute_s` | 1.973180e-04 |
| `complex_multi_shift_plan_compile_s` | 2.757147e-01 |
| `complex_multi_shift_plan_execute_s` | 7.903200e-05 |
| `complex_multi_shift_plan_warm_s` | 2.593500e-05 |
| `complex_restarted_action_cold_s` | 4.867772e-01 |
| `complex_restarted_action_plan_cold_s` | 4.370777e-01 |
| `complex_solve_action_cold_s` | 3.241474e-01 |
| `complex_solve_action_plan_cold_s` | 2.789139e-01 |
| `complex_solve_action_plan_compile_s` | 1.133130e-04 |
| `complex_solve_action_plan_execute_s` | 1.909900e-05 |
| `complex_solve_action_plan_warm_s` | 1.498380e-05 |
| `complex_solve_grad_plan_compile_s` | 1.132958e+00 |
| `complex_solve_grad_plan_execute_s` | 1.156370e-04 |
| `real_action_cold_s` | 3.624371e-01 |
| `real_action_plan_cold_s` | 2.987012e-01 |
| `real_action_plan_precompile_s` | 2.438690e-04 |
| `real_action_plan_warm_s` | 1.574254e-04 |
| `real_action_warm_s` | 2.126430e-04 |
| `real_apply_cold_s` | 1.037686e-01 |
| `real_apply_plan_cold_s` | 1.073488e-01 |
| `real_apply_plan_precompile_s` | 7.478700e-05 |
| `real_apply_plan_warm_s` | 8.923000e-06 |
| `real_apply_warm_s` | 9.662800e-06 |
| `real_det_cold_s` | 3.668413e-01 |
| `real_det_plan_cold_s` | 4.232932e-01 |
| `real_det_plan_precompile_s` | 1.744970e-04 |
| `real_eigsh_cold_s` | 4.359455e-01 |
| `real_eigsh_plan_cold_s` | 2.760115e-01 |
| `real_eigsh_restarted_plan_compile_s` | 4.974701e-01 |
| `real_eigsh_restarted_plan_execute_s` | 3.169510e-04 |
| `real_grad_cold_s` | 5.622054e-01 |
| `real_inverse_action_cold_s` | 2.799831e-01 |
| `real_inverse_action_plan_cold_s` | 2.302693e-01 |
| `real_inverse_action_plan_compile_s` | 4.466500e-05 |
| `real_inverse_action_plan_execute_s` | 1.689200e-05 |
| `real_inverse_action_plan_warm_s` | 1.337620e-05 |
| `real_inverse_grad_plan_compile_s` | 9.275620e-01 |
| `real_inverse_grad_plan_execute_s` | 1.218140e-04 |
| `real_logdet_cold_s` | 4.159505e-01 |
| `real_logdet_grad_cold_s` | 6.523295e-01 |
| `real_logdet_grad_compile_s` | 3.743150e-04 |
| `real_logdet_grad_execute_s` | 2.381120e-04 |
| `real_logdet_plan_cold_s` | 3.806513e-01 |
| `real_logdet_plan_precompile_s` | 2.399880e-04 |
| `real_logdet_plan_warm_s` | 2.263988e-04 |
| `real_logdet_warm_s` | 2.346490e-04 |
| `real_minres_plan_cold_s` | 4.315779e-01 |
| `real_minres_plan_compile_s` | 5.415920e-04 |
| `real_minres_plan_execute_s` | 3.009110e-04 |
| `real_multi_shift_plan_compile_s` | 4.962484e-01 |
| `real_multi_shift_plan_execute_s` | 1.557480e-04 |
| `real_multi_shift_plan_warm_s` | 1.981160e-05 |
| `real_restarted_action_cold_s` | 3.783568e-01 |
| `real_restarted_action_plan_cold_s` | 3.892620e-01 |
| `real_solve_action_cold_s` | 3.266893e-01 |
| `real_solve_action_plan_cold_s` | 3.134370e-01 |
| `real_solve_action_plan_compile_s` | 1.526090e-04 |
| `real_solve_action_plan_execute_s` | 2.937100e-05 |
| `real_solve_action_plan_warm_s` | 1.476280e-05 |
| `real_solve_grad_plan_compile_s` | 7.559697e-01 |
| `real_solve_grad_plan_execute_s` | 5.773700e-05 |
| `sparse_complex_action_plan_s` | 3.385786e-01 |
| `sparse_complex_action_s` | 4.512185e-01 |
| `sparse_complex_apply_plan_s` | 1.261877e-01 |
| `sparse_complex_apply_s` | 1.390782e-01 |
| `sparse_complex_det_plan_s` | 5.021815e-01 |
| `sparse_complex_det_s` | 5.772473e-01 |
| `sparse_complex_inverse_action_plan_s` | 2.608371e-01 |
| `sparse_complex_inverse_action_s` | 2.984467e-01 |
| `sparse_complex_logdet_plan_s` | 5.421530e-01 |
| `sparse_complex_logdet_s` | 5.885214e-01 |
| `sparse_complex_restarted_plan_s` | 4.079589e-01 |
| `sparse_complex_restarted_s` | 3.923100e-01 |
| `sparse_complex_solve_action_plan_s` | 2.578791e-01 |
| `sparse_complex_solve_action_s` | 3.447213e-01 |
| `sparse_real_apply_plan_s` | 1.065154e-01 |
| `sparse_real_apply_s` | 1.182639e-01 |
| `sparse_real_det_plan_s` | 4.142852e-01 |
| `sparse_real_det_s` | 4.083789e-01 |
| `sparse_real_inverse_action_plan_s` | 2.386235e-01 |
| `sparse_real_inverse_action_s` | 2.615517e-01 |
| `sparse_real_inverse_diag_corrected_s` | 1.960161e+00 |
| `sparse_real_inverse_diag_local_s` | 2.331618e-01 |
| `sparse_real_logdet_grad_s` | 7.221256e-01 |
| `sparse_real_logdet_leja_hutchpp_auto_s` | 2.012789e+00 |
| `sparse_real_logdet_leja_hutchpp_s` | 1.371617e+00 |
| `sparse_real_logdet_plan_s` | 4.388212e-01 |
| `sparse_real_logdet_s` | 5.240224e-01 |
| `sparse_real_solve_action_plan_s` | 2.399148e-01 |
| `sparse_real_solve_action_s` | 3.092897e-01 |

## Matrix Backend Candidates

Command:

```bash
python benchmarks/benchmark_matrix_backend_candidates.py --n 4 --warmup 0 --runs 1
```

| metric | seconds |
| --- | ---: |
| `candidate_arbplusjax_sparse_cached_matvec_s` | 1.365225e-03 |
| `candidate_arbplusjax_sparse_fromdense_solve_s` | 5.077302e-01 |
| `candidate_arbplusjax_sparse_matvec_s` | 2.002409e-03 |
| `candidate_arbplusjax_sparse_spd_solve_s` | 3.353935e-01 |
| `candidate_jax_dense_eigh_s` | 6.482004e-02 |
| `candidate_jax_dense_matvec_s` | 2.786409e-02 |
| `candidate_jax_dense_solve_s` | 1.223299e-01 |
| `candidate_jax_experimental_sparse_cg_s` | 1.428925e-01 |
| `candidate_jax_experimental_sparse_matvec_s` | 6.648403e-02 |
| `candidate_jax_scipy_dense_solve_s` | 9.200946e-02 |
| `candidate_matfree_apply_s` | 4.390593e-03 |
| `candidate_matfree_logdet_slq_s` | 2.914322e-01 |
| `candidate_matfree_solve_action_s` | 9.463300e-05 |
| `candidate_petsc_available` | 0.000000e+00 |
| `candidate_scipy_dense_eigh_s` | 2.056350e-04 |
| `candidate_scipy_dense_matvec_s` | 8.463800e-05 |
| `candidate_scipy_dense_solve_s` | 3.063200e-04 |
| `candidate_scipy_linear_operator_cg_s` | 1.204840e-04 |
| `candidate_scipy_sparse_cg_s` | 2.413350e-04 |
| `candidate_scipy_sparse_eigsh_s` | 8.675260e-04 |
| `candidate_scipy_sparse_matvec_s` | 9.025400e-05 |
| `candidate_slepc_available` | 0.000000e+00 |
