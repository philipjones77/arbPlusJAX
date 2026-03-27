Last updated: 2026-03-20T00:00:00Z

# Matrix Surface Workbook

This workbook summarizes the benchmark surfaces for dense, sparse, and matrix-free matrix families.

It is intended to make the current comparison surface legible while the matrix API and execution-strategy hardening continues.

The workbook now compares dense, sparse, block-sparse, variable-block sparse, and matrix-free/operator-plan execution in one place.

## Compare and Contrast

| family | use when | fastest metrics |
| --- | --- | --- |
| dense | matrices are small/medium and direct kernels or cached dense plans are appropriate | `arb_dense_plan_prepare_s`=1.502e-04s, `acb_dense_plan_prepare_s`=1.991e-04s, `acb_diag_s`=4.136e-02s, `arb_diag_s`=5.134e-02s |
| sparse | storage sparsity is meaningful and callers want sparse cached matvec/rmatvec reuse | `srb_coo_point_cached_matvec_s`=3.516e-03s, `srb_csr_point_cached_matvec_s`=4.759e-03s, `scb_coo_point_cached_matvec_s`=5.077e-03s, `scb_csr_point_cached_matvec_s`=5.440e-03s |
| block sparse | block structure is explicit and callers want block-native apply paths | `srb_block_rmatvec_cached_s`=7.660e-02s, `scb_block_adjoint_cached_s`=8.391e-02s, `srb_block_matvec_s`=1.634e-01s, `scb_block_matvec_s`=1.871e-01s |
| variable block sparse | partitions are irregular but structure should still be preserved | `srb_vblock_matvec_cached_s`=1.376e-01s, `srb_vblock_matvec_s`=1.923e-01s, `scb_vblock_matvec_cached_s`=2.229e-01s, `scb_vblock_matvec_s`=3.277e-01s |
| matrix free | operator plans, Krylov solves, logdet, or adapter-based execution matter more than explicit materialization | `complex_apply_plan_warm_s`=5.435e-04s, `complex_apply_plan_precompile_s`=6.649e-04s, `complex_solve_action_plan_warm_s`=1.003e-03s, `complex_inverse_action_plan_execute_s`=1.198e-03s |

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
| `acb_cached_matvec_padded_s` | 9.722192e-01 |
| `acb_cached_matvec_s` | 8.521303e-01 |
| `acb_conjugate_transpose_s` | 7.399357e-02 |
| `acb_dense_plan_prepare_s` | 1.991460e-04 |
| `acb_diag_s` | 4.136025e-02 |
| `acb_direct_solve_s` | 6.975206e-01 |
| `acb_lu_reuse_s` | 3.816874e-01 |
| `acb_transpose_s` | 1.275840e-01 |
| `arb_cached_matvec_padded_s` | 2.539964e+00 |
| `arb_cached_matvec_s` | 3.341338e-01 |
| `arb_dense_plan_prepare_s` | 1.502160e-04 |
| `arb_diag_s` | 5.134174e-02 |
| `arb_direct_solve_s` | 6.047727e-01 |
| `arb_lu_reuse_s` | 2.890418e-01 |
| `arb_transpose_s` | 8.231005e-02 |

## Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_bcoo_basic_cached_matvec_s` | 1.566262e-01 |
| `scb_bcoo_basic_matvec_s` | 1.410489e-01 |
| `scb_bcoo_point_cached_matvec_s` | 8.994255e-03 |
| `scb_bcoo_point_matvec_s` | 9.662719e-03 |
| `scb_coo_basic_cached_matvec_s` | 1.341054e-01 |
| `scb_coo_basic_matvec_s` | 2.476556e-01 |
| `scb_coo_point_cached_matvec_s` | 5.077445e-03 |
| `scb_coo_point_matvec_s` | 2.150204e-01 |
| `scb_csr_basic_cached_matvec_s` | 1.286944e-01 |
| `scb_csr_basic_matvec_s` | 2.149314e-01 |
| `scb_csr_point_cached_matvec_s` | 5.440380e-03 |
| `scb_csr_point_matvec_s` | 1.355970e-02 |
| `srb_bcoo_basic_cached_matvec_s` | 3.861901e-02 |
| `srb_bcoo_basic_matvec_s` | 4.649157e-02 |
| `srb_bcoo_point_cached_matvec_s` | 1.042491e-02 |
| `srb_bcoo_point_matvec_s` | 9.195826e-03 |
| `srb_coo_basic_cached_matvec_s` | 4.676209e-02 |
| `srb_coo_basic_matvec_s` | 1.230239e+00 |
| `srb_coo_point_cached_matvec_s` | 3.515984e-03 |
| `srb_coo_point_matvec_s` | 1.778291e-01 |
| `srb_csr_basic_cached_matvec_s` | 3.797244e-02 |
| `srb_csr_basic_matvec_s` | 4.304082e-02 |
| `srb_csr_point_cached_matvec_s` | 4.759457e-03 |
| `srb_csr_point_matvec_s` | 1.113417e-02 |

## Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_block_sparse_matrix_surface.py --n-blocks 2 --block-size 2 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_block_adjoint_cached_s` | 8.391464e-02 |
| `scb_block_matvec_s` | 1.871397e-01 |
| `srb_block_matvec_s` | 1.634468e-01 |
| `srb_block_rmatvec_cached_s` | 7.660352e-02 |

## Variable-Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_vblock_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_vblock_matvec_cached_s` | 2.229362e-01 |
| `scb_vblock_matvec_s` | 3.277373e-01 |
| `srb_vblock_matvec_cached_s` | 1.376151e-01 |
| `srb_vblock_matvec_s` | 1.923321e-01 |

## Matrix-Free Surface

Command:

```bash
python benchmarks/benchmark_matrix_free_krylov.py
```

| metric | seconds |
| --- | ---: |
| `complex_action_cold_s` | 1.436794e+00 |
| `complex_action_plan_cold_s` | 7.187770e-01 |
| `complex_action_plan_precompile_s` | 7.030209e-03 |
| `complex_action_plan_warm_s` | 7.647221e-03 |
| `complex_apply_cold_s` | 2.529315e-01 |
| `complex_apply_plan_cold_s` | 1.421152e-01 |
| `complex_apply_plan_precompile_s` | 6.648970e-04 |
| `complex_apply_plan_warm_s` | 5.434870e-04 |
| `complex_det_cold_s` | 9.373794e-01 |
| `complex_det_plan_cold_s` | 9.151074e-01 |
| `complex_det_plan_precompile_s` | 9.027238e-03 |
| `complex_eigsh_cold_s` | 8.055317e-01 |
| `complex_eigsh_plan_cold_s` | 4.885019e-01 |
| `complex_eigsh_restarted_plan_compile_s` | 1.025824e+00 |
| `complex_eigsh_restarted_plan_execute_s` | 2.596451e-02 |
| `complex_grad_cold_s` | 8.824336e-01 |
| `complex_inverse_action_cold_s` | 5.991104e-01 |
| `complex_inverse_action_plan_cold_s` | 4.323608e-01 |
| `complex_inverse_action_plan_compile_s` | 1.273994e-03 |
| `complex_inverse_action_plan_execute_s` | 1.197858e-03 |
| `complex_inverse_action_plan_warm_s` | 1.211351e-03 |
| `complex_inverse_grad_plan_compile_s` | 1.612970e+00 |
| `complex_inverse_grad_plan_execute_s` | 3.099673e-03 |
| `complex_logdet_cold_s` | 1.875918e+00 |
| `complex_logdet_grad_cold_s` | 1.832033e+00 |
| `complex_logdet_grad_compile_s` | 2.876696e-02 |
| `complex_logdet_grad_execute_s` | 3.400531e-02 |
| `complex_logdet_plan_cold_s` | 8.041452e-01 |
| `complex_logdet_plan_precompile_s` | 2.074024e-02 |
| `complex_logdet_plan_warm_s` | 1.155816e-02 |
| `complex_minres_plan_cold_s` | 1.110969e+00 |
| `complex_minres_plan_compile_s` | 1.245389e-02 |
| `complex_minres_plan_execute_s` | 2.200852e-02 |
| `complex_multi_shift_plan_compile_s` | 8.068535e-01 |
| `complex_multi_shift_plan_execute_s` | 1.574519e-03 |
| `complex_multi_shift_plan_warm_s` | 1.624071e-03 |
| `complex_restarted_action_cold_s` | 9.482435e-01 |
| `complex_restarted_action_plan_cold_s` | 8.002454e-01 |
| `complex_solve_action_cold_s` | 5.483303e-01 |
| `complex_solve_action_plan_cold_s` | 5.458937e-01 |
| `complex_solve_action_plan_compile_s` | 1.698657e-03 |
| `complex_solve_action_plan_execute_s` | 1.204953e-03 |
| `complex_solve_action_plan_warm_s` | 1.002999e-03 |
| `complex_solve_grad_plan_compile_s` | 2.154705e+00 |
| `complex_solve_grad_plan_execute_s` | 2.376495e-03 |
| `real_action_cold_s` | 1.284143e+00 |
| `real_action_plan_cold_s` | 5.667324e-01 |
| `real_action_plan_precompile_s` | 1.123375e-02 |
| `real_action_plan_warm_s` | 6.545483e-03 |
| `real_action_warm_s` | 7.294853e-03 |
| `real_apply_cold_s` | 2.311998e-01 |
| `real_apply_plan_cold_s` | 9.932569e-02 |
| `real_apply_plan_precompile_s` | 1.341907e-03 |
| `real_apply_plan_warm_s` | 2.450473e-03 |
| `real_apply_warm_s` | 1.259710e-03 |
| `real_det_cold_s` | 5.681703e-01 |
| `real_det_plan_cold_s` | 5.594371e-01 |
| `real_det_plan_precompile_s` | 3.328775e-03 |
| `real_eigsh_cold_s` | 6.065776e-01 |
| `real_eigsh_plan_cold_s` | 5.919467e-01 |
| `real_eigsh_restarted_plan_compile_s` | 7.255040e-01 |
| `real_eigsh_restarted_plan_execute_s` | 8.475185e-03 |
| `real_grad_cold_s` | 6.356557e-01 |
| `real_inverse_action_cold_s` | 3.008896e-01 |
| `real_inverse_action_plan_cold_s` | 3.329286e-01 |
| `real_inverse_action_plan_compile_s` | 9.380020e-03 |
| `real_inverse_action_plan_execute_s` | 2.671211e-03 |
| `real_inverse_action_plan_warm_s` | 1.898208e-03 |
| `real_inverse_grad_plan_compile_s` | 1.145269e+00 |
| `real_inverse_grad_plan_execute_s` | 8.577269e-03 |
| `real_logdet_cold_s` | 1.231878e+00 |
| `real_logdet_grad_cold_s` | 8.447755e-01 |
| `real_logdet_grad_compile_s` | 3.196367e-03 |
| `real_logdet_grad_execute_s` | 1.519726e-03 |
| `real_logdet_plan_cold_s` | 6.085317e-01 |
| `real_logdet_plan_precompile_s` | 3.258848e-03 |
| `real_logdet_plan_warm_s` | 2.668544e-03 |
| `real_logdet_warm_s` | 2.687086e-03 |
| `real_minres_plan_cold_s` | 1.092704e+00 |
| `real_minres_plan_compile_s` | 1.180744e-02 |
| `real_minres_plan_execute_s` | 1.060554e-02 |
| `real_multi_shift_plan_compile_s` | 7.021765e-01 |
| `real_multi_shift_plan_execute_s` | 1.606253e-03 |
| `real_multi_shift_plan_warm_s` | 1.613942e-03 |
| `real_restarted_action_cold_s` | 9.661381e-01 |
| `real_restarted_action_plan_cold_s` | 7.093354e-01 |
| `real_solve_action_cold_s` | 5.211469e-01 |
| `real_solve_action_plan_cold_s` | 3.502851e-01 |
| `real_solve_action_plan_compile_s` | 3.830812e-03 |
| `real_solve_action_plan_execute_s` | 2.471105e-03 |
| `real_solve_action_plan_warm_s` | 1.960867e-03 |
| `real_solve_grad_plan_compile_s` | 1.528887e+00 |
| `real_solve_grad_plan_execute_s` | 4.828934e-03 |
| `sparse_complex_action_plan_s` | 6.993121e-01 |
| `sparse_complex_action_s` | 1.008191e+00 |
| `sparse_complex_apply_plan_s` | 1.873610e-01 |
| `sparse_complex_apply_s` | 3.244521e-01 |
| `sparse_complex_det_plan_s` | 9.496599e-01 |
| `sparse_complex_det_s` | 1.034075e+00 |
| `sparse_complex_inverse_action_plan_s` | 4.474070e-01 |
| `sparse_complex_inverse_action_s` | 4.682012e-01 |
| `sparse_complex_logdet_plan_s` | 1.114636e+00 |
| `sparse_complex_logdet_s` | 1.224949e+00 |
| `sparse_complex_restarted_plan_s` | 1.138875e+00 |
| `sparse_complex_restarted_s` | 9.755596e-01 |
| `sparse_complex_solve_action_plan_s` | 5.300052e-01 |
| `sparse_complex_solve_action_s` | 4.854586e-01 |
| `sparse_real_apply_plan_s` | 1.082561e-01 |
| `sparse_real_apply_s` | 1.615485e-01 |
| `sparse_real_det_plan_s` | 7.011477e-01 |
| `sparse_real_det_s` | 6.572574e-01 |
| `sparse_real_inverse_action_plan_s` | 3.813791e-01 |
| `sparse_real_inverse_action_s` | 4.584532e-01 |
| `sparse_real_inverse_diag_corrected_s` | 2.138071e+00 |
| `sparse_real_inverse_diag_local_s` | 2.790244e-01 |
| `sparse_real_logdet_grad_s` | 1.100131e+00 |
| `sparse_real_logdet_leja_hutchpp_auto_s` | 4.521177e+00 |
| `sparse_real_logdet_leja_hutchpp_s` | 2.852763e+00 |
| `sparse_real_logdet_plan_s` | 7.284164e-01 |
| `sparse_real_logdet_s` | 1.381327e+00 |
| `sparse_real_solve_action_plan_s` | 4.784746e-01 |
| `sparse_real_solve_action_s` | 3.704928e-01 |

## Matrix Backend Candidates

Command:

```bash
python benchmarks/benchmark_matrix_backend_candidates.py --n 4 --warmup 0 --runs 1
```

| metric | seconds |
| --- | ---: |
| `candidate_arbplusjax_sparse_cached_matvec_s` | 6.165800e-03 |
| `candidate_arbplusjax_sparse_fromdense_solve_s` | 1.052248e+00 |
| `candidate_arbplusjax_sparse_matvec_s` | 6.124561e-03 |
| `candidate_arbplusjax_sparse_spd_solve_s` | 6.108898e-01 |
| `candidate_jax_dense_eigh_s` | 1.352385e-01 |
| `candidate_jax_dense_matvec_s` | 6.132035e-02 |
| `candidate_jax_dense_solve_s` | 1.318999e-01 |
| `candidate_jax_experimental_sparse_cg_s` | 4.419904e-01 |
| `candidate_jax_experimental_sparse_matvec_s` | 1.171842e-01 |
| `candidate_jax_scipy_dense_solve_s` | 1.373971e-01 |
| `candidate_matfree_apply_s` | 2.564160e-02 |
| `candidate_matfree_logdet_slq_s` | 5.417706e-01 |
| `candidate_matfree_solve_action_s` | 1.986904e-03 |
| `candidate_petsc_available` | 0.000000e+00 |
| `candidate_scipy_dense_eigh_s` | 1.519676e-03 |
| `candidate_scipy_dense_matvec_s` | 1.040910e-04 |
| `candidate_scipy_dense_solve_s` | 5.043209e-03 |
| `candidate_scipy_linear_operator_cg_s` | 1.427930e-04 |
| `candidate_scipy_sparse_cg_s` | 4.119930e-04 |
| `candidate_scipy_sparse_eigsh_s` | 3.606062e-03 |
| `candidate_scipy_sparse_matvec_s` | 9.852910e-04 |
| `candidate_slepc_available` | 0.000000e+00 |
