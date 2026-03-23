Last updated: 2026-03-20T00:00:00Z

# Matrix Surface Workbook

This workbook summarizes the benchmark surfaces for dense, sparse, and matrix-free matrix families.

It is intended to make the current comparison surface legible while the matrix API and execution-strategy hardening continues.

The workbook now compares dense, sparse, block-sparse, variable-block sparse, and matrix-free/operator-plan execution in one place.

## Compare and Contrast

| family | use when | fastest metrics |
| --- | --- | --- |
| dense | matrices are small/medium and direct kernels or cached dense plans are appropriate | `acb_dense_plan_prepare_s`=1.042e-04s, `arb_dense_plan_prepare_s`=1.280e-04s, `arb_transpose_s`=2.988e-02s, `acb_diag_s`=3.106e-02s |
| sparse | storage sparsity is meaningful and callers want sparse cached matvec/rmatvec reuse | `srb_bcoo_point_cached_matvec_s`=6.703e-04s, `srb_bcoo_point_matvec_s`=7.637e-04s, `scb_csr_point_cached_matvec_s`=1.091e-03s, `srb_csr_point_cached_matvec_s`=1.440e-03s |
| block sparse | block structure is explicit and callers want block-native apply paths | `srb_block_rmatvec_cached_s`=4.962e-02s, `scb_block_adjoint_cached_s`=5.445e-02s, `scb_block_matvec_s`=9.920e-02s, `srb_block_matvec_s`=1.162e-01s |
| variable block sparse | partitions are irregular but structure should still be preserved | `srb_vblock_matvec_cached_s`=9.207e-02s, `scb_vblock_matvec_cached_s`=1.418e-01s, `scb_vblock_matvec_s`=1.418e-01s, `srb_vblock_matvec_s`=1.457e-01s |
| matrix free | operator plans, Krylov solves, logdet, or adapter-based execution matter more than explicit materialization | `complex_apply_plan_warm_s`=5.164e-06s, `complex_inverse_action_plan_warm_s`=7.118e-06s, `real_apply_plan_warm_s`=7.225e-06s, `complex_inverse_action_plan_execute_s`=7.320e-06s |

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
| `acb_cached_matvec_padded_s` | 4.668409e-01 |
| `acb_cached_matvec_s` | 2.097631e-01 |
| `acb_conjugate_transpose_s` | 3.375074e-02 |
| `acb_dense_plan_prepare_s` | 1.041930e-04 |
| `acb_diag_s` | 3.105611e-02 |
| `acb_direct_solve_s` | 2.261352e-01 |
| `acb_lu_reuse_s` | 1.532212e-01 |
| `acb_transpose_s` | 3.351432e-02 |
| `arb_cached_matvec_padded_s` | 1.770698e+00 |
| `arb_cached_matvec_s` | 1.131189e-01 |
| `arb_dense_plan_prepare_s` | 1.280300e-04 |
| `arb_diag_s` | 3.978727e-02 |
| `arb_direct_solve_s` | 1.841502e-01 |
| `arb_lu_reuse_s` | 1.138828e-01 |
| `arb_transpose_s` | 2.987866e-02 |

## Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_bcoo_basic_cached_matvec_s` | 2.399740e-02 |
| `scb_bcoo_basic_matvec_s` | 1.437756e-02 |
| `scb_bcoo_point_cached_matvec_s` | 3.628367e-03 |
| `scb_bcoo_point_matvec_s` | 2.450274e-03 |
| `scb_coo_basic_cached_matvec_s` | 2.362385e-02 |
| `scb_coo_basic_matvec_s` | 1.111682e-01 |
| `scb_coo_point_cached_matvec_s` | 1.619663e-03 |
| `scb_coo_point_matvec_s` | 1.467508e-01 |
| `scb_csr_basic_cached_matvec_s` | 2.294980e-02 |
| `scb_csr_basic_matvec_s` | 3.993908e-02 |
| `scb_csr_point_cached_matvec_s` | 1.090832e-03 |
| `scb_csr_point_matvec_s` | 2.742431e-03 |
| `srb_bcoo_basic_cached_matvec_s` | 3.605523e-03 |
| `srb_bcoo_basic_matvec_s` | 4.253742e-03 |
| `srb_bcoo_point_cached_matvec_s` | 6.703260e-04 |
| `srb_bcoo_point_matvec_s` | 7.636500e-04 |
| `srb_coo_basic_cached_matvec_s` | 4.810711e-03 |
| `srb_coo_basic_matvec_s` | 6.449660e-01 |
| `srb_coo_point_cached_matvec_s` | 1.949493e-03 |
| `srb_coo_point_matvec_s` | 1.190646e-01 |
| `srb_csr_basic_cached_matvec_s` | 3.354957e-03 |
| `srb_csr_basic_matvec_s` | 4.753825e-03 |
| `srb_csr_point_cached_matvec_s` | 1.440185e-03 |
| `srb_csr_point_matvec_s` | 3.936453e-03 |

## Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_block_sparse_matrix_surface.py --n-blocks 2 --block-size 2 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_block_adjoint_cached_s` | 5.444849e-02 |
| `scb_block_matvec_s` | 9.920160e-02 |
| `srb_block_matvec_s` | 1.161870e-01 |
| `srb_block_rmatvec_cached_s` | 4.961819e-02 |

## Variable-Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_vblock_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_vblock_matvec_cached_s` | 1.418180e-01 |
| `scb_vblock_matvec_s` | 1.418402e-01 |
| `srb_vblock_matvec_cached_s` | 9.207287e-02 |
| `srb_vblock_matvec_s` | 1.456817e-01 |

## Matrix-Free Surface

Command:

```bash
python benchmarks/benchmark_matrix_free_krylov.py
```

| metric | seconds |
| --- | ---: |
| `complex_action_cold_s` | 3.521791e-01 |
| `complex_action_plan_cold_s` | 2.557927e-01 |
| `complex_action_plan_precompile_s` | 1.224630e-04 |
| `complex_action_plan_warm_s` | 1.777400e-05 |
| `complex_apply_cold_s` | 1.057569e-01 |
| `complex_apply_plan_cold_s` | 8.954571e-02 |
| `complex_apply_plan_precompile_s` | 4.359800e-05 |
| `complex_apply_plan_warm_s` | 5.163800e-06 |
| `complex_det_cold_s` | 3.641509e-01 |
| `complex_det_plan_cold_s` | 3.707087e-01 |
| `complex_det_plan_precompile_s` | 2.942000e-04 |
| `complex_eigsh_cold_s` | 2.630090e-01 |
| `complex_eigsh_plan_cold_s` | 2.186637e-01 |
| `complex_eigsh_restarted_plan_compile_s` | 4.016164e-01 |
| `complex_eigsh_restarted_plan_execute_s` | 1.539230e-04 |
| `complex_grad_cold_s` | 3.291804e-01 |
| `complex_inverse_action_cold_s` | 1.782412e-01 |
| `complex_inverse_action_plan_cold_s` | 1.908048e-01 |
| `complex_inverse_action_plan_compile_s` | 2.374500e-05 |
| `complex_inverse_action_plan_execute_s` | 7.320000e-06 |
| `complex_inverse_action_plan_warm_s` | 7.117600e-06 |
| `complex_inverse_grad_plan_compile_s` | 8.818453e-01 |
| `complex_inverse_grad_plan_execute_s` | 6.624200e-05 |
| `complex_logdet_cold_s` | 6.340803e-01 |
| `complex_logdet_grad_cold_s` | 7.609862e-01 |
| `complex_logdet_grad_compile_s` | 3.250090e-04 |
| `complex_logdet_grad_execute_s` | 2.882280e-04 |
| `complex_logdet_plan_cold_s` | 3.590839e-01 |
| `complex_logdet_plan_precompile_s` | 3.017770e-04 |
| `complex_logdet_plan_warm_s` | 1.082222e-04 |
| `complex_minres_plan_cold_s` | 2.942099e-01 |
| `complex_minres_plan_compile_s` | 2.241400e-04 |
| `complex_minres_plan_execute_s` | 1.053900e-04 |
| `complex_multi_shift_plan_compile_s` | 2.370652e-01 |
| `complex_multi_shift_plan_execute_s` | 6.575400e-05 |
| `complex_multi_shift_plan_warm_s` | 1.141080e-05 |
| `complex_restarted_action_cold_s` | 3.058926e-01 |
| `complex_restarted_action_plan_cold_s` | 2.818026e-01 |
| `complex_solve_action_cold_s` | 1.625636e-01 |
| `complex_solve_action_plan_cold_s` | 2.050119e-01 |
| `complex_solve_action_plan_compile_s` | 8.042800e-05 |
| `complex_solve_action_plan_execute_s` | 1.366800e-05 |
| `complex_solve_action_plan_warm_s` | 7.738400e-06 |
| `complex_solve_grad_plan_compile_s` | 7.224172e-01 |
| `complex_solve_grad_plan_execute_s` | 5.150400e-05 |
| `real_action_cold_s` | 2.341797e-01 |
| `real_action_plan_cold_s` | 2.147824e-01 |
| `real_action_plan_precompile_s` | 1.832490e-04 |
| `real_action_plan_warm_s` | 1.311458e-04 |
| `real_action_warm_s` | 8.099540e-05 |
| `real_apply_cold_s` | 7.017439e-02 |
| `real_apply_plan_cold_s` | 7.870341e-02 |
| `real_apply_plan_precompile_s` | 6.988800e-05 |
| `real_apply_plan_warm_s` | 7.225400e-06 |
| `real_apply_warm_s` | 7.793800e-06 |
| `real_det_cold_s` | 2.348105e-01 |
| `real_det_plan_cold_s` | 2.628946e-01 |
| `real_det_plan_precompile_s` | 1.404350e-04 |
| `real_eigsh_cold_s` | 1.963583e-01 |
| `real_eigsh_plan_cold_s` | 1.674485e-01 |
| `real_eigsh_restarted_plan_compile_s` | 2.845023e-01 |
| `real_eigsh_restarted_plan_execute_s` | 5.677010e-04 |
| `real_grad_cold_s` | 3.677596e-01 |
| `real_inverse_action_cold_s` | 1.398061e-01 |
| `real_inverse_action_plan_cold_s` | 1.569544e-01 |
| `real_inverse_action_plan_compile_s` | 3.158000e-05 |
| `real_inverse_action_plan_execute_s` | 9.360000e-06 |
| `real_inverse_action_plan_warm_s` | 1.040660e-05 |
| `real_inverse_grad_plan_compile_s` | 5.492623e-01 |
| `real_inverse_grad_plan_execute_s` | 6.015300e-05 |
| `real_logdet_cold_s` | 2.605526e-01 |
| `real_logdet_grad_cold_s` | 3.326548e-01 |
| `real_logdet_grad_compile_s` | 1.778540e-04 |
| `real_logdet_grad_execute_s` | 8.082100e-05 |
| `real_logdet_plan_cold_s` | 2.542553e-01 |
| `real_logdet_plan_precompile_s` | 2.356290e-04 |
| `real_logdet_plan_warm_s` | 9.295880e-05 |
| `real_logdet_warm_s` | 1.341390e-04 |
| `real_minres_plan_cold_s` | 2.442078e-01 |
| `real_minres_plan_compile_s` | 4.314620e-04 |
| `real_minres_plan_execute_s` | 3.505920e-04 |
| `real_multi_shift_plan_compile_s` | 1.995032e-01 |
| `real_multi_shift_plan_execute_s` | 7.520200e-05 |
| `real_multi_shift_plan_warm_s` | 1.904020e-05 |
| `real_restarted_action_cold_s` | 2.296876e-01 |
| `real_restarted_action_plan_cold_s` | 2.280972e-01 |
| `real_solve_action_cold_s` | 1.528801e-01 |
| `real_solve_action_plan_cold_s` | 1.490880e-01 |
| `real_solve_action_plan_compile_s` | 8.715500e-05 |
| `real_solve_action_plan_execute_s` | 2.017300e-05 |
| `real_solve_action_plan_warm_s` | 1.019260e-05 |
| `real_solve_grad_plan_compile_s` | 5.723914e-01 |
| `real_solve_grad_plan_execute_s` | 3.960300e-05 |
| `sparse_complex_action_plan_s` | 3.081899e-01 |
| `sparse_complex_action_s` | 2.797752e-01 |
| `sparse_complex_apply_plan_s` | 9.902910e-02 |
| `sparse_complex_apply_s` | 1.115458e-01 |
| `sparse_complex_det_plan_s` | 3.595749e-01 |
| `sparse_complex_det_s` | 4.389352e-01 |
| `sparse_complex_inverse_action_plan_s` | 2.345671e-01 |
| `sparse_complex_inverse_action_s` | 2.429814e-01 |
| `sparse_complex_logdet_plan_s` | 3.472174e-01 |
| `sparse_complex_logdet_s` | 3.707508e-01 |
| `sparse_complex_restarted_plan_s` | 2.832940e-01 |
| `sparse_complex_restarted_s` | 2.904426e-01 |
| `sparse_complex_solve_action_plan_s` | 2.134439e-01 |
| `sparse_complex_solve_action_s` | 1.854636e-01 |
| `sparse_real_apply_plan_s` | 7.745738e-02 |
| `sparse_real_apply_s` | 7.393870e-02 |
| `sparse_real_det_plan_s` | 2.410899e-01 |
| `sparse_real_det_s` | 2.991860e-01 |
| `sparse_real_inverse_action_plan_s` | 1.648481e-01 |
| `sparse_real_inverse_action_s` | 1.456034e-01 |
| `sparse_real_inverse_diag_corrected_s` | 1.288295e+00 |
| `sparse_real_inverse_diag_local_s` | 1.599136e-01 |
| `sparse_real_logdet_grad_s` | 4.394307e-01 |
| `sparse_real_logdet_leja_hutchpp_auto_s` | 1.268761e+00 |
| `sparse_real_logdet_leja_hutchpp_s` | 8.225543e-01 |
| `sparse_real_logdet_plan_s` | 2.636632e-01 |
| `sparse_real_logdet_s` | 3.299175e-01 |
| `sparse_real_solve_action_plan_s` | 1.586457e-01 |
| `sparse_real_solve_action_s` | 1.679090e-01 |

## Matrix Backend Candidates

Command:

```bash
python benchmarks/benchmark_matrix_backend_candidates.py --n 4 --warmup 0 --runs 1
```

| metric | seconds |
| --- | ---: |
| `candidate_arbplusjax_sparse_cached_matvec_s` | 8.845130e-04 |
| `candidate_arbplusjax_sparse_fromdense_solve_s` | 4.200861e-01 |
| `candidate_arbplusjax_sparse_matvec_s` | 1.682493e-03 |
| `candidate_arbplusjax_sparse_spd_solve_s` | 2.714739e-01 |
| `candidate_jax_dense_eigh_s` | 5.698707e-02 |
| `candidate_jax_dense_matvec_s` | 2.578832e-02 |
| `candidate_jax_dense_solve_s` | 9.687107e-02 |
| `candidate_jax_experimental_sparse_cg_s` | 1.222515e-01 |
| `candidate_jax_experimental_sparse_matvec_s` | 6.474080e-02 |
| `candidate_jax_scipy_dense_solve_s` | 7.571446e-02 |
| `candidate_matfree_apply_s` | 3.008150e-03 |
| `candidate_matfree_logdet_slq_s` | 2.853393e-01 |
| `candidate_matfree_solve_action_s` | 1.162440e-04 |
| `candidate_petsc_available` | 0.000000e+00 |
| `candidate_scipy_dense_eigh_s` | 1.463340e-04 |
| `candidate_scipy_dense_matvec_s` | 1.186400e-04 |
| `candidate_scipy_dense_solve_s` | 2.687180e-04 |
| `candidate_scipy_linear_operator_cg_s` | 9.438600e-05 |
| `candidate_scipy_sparse_cg_s` | 1.868440e-04 |
| `candidate_scipy_sparse_eigsh_s` | 4.213470e-04 |
| `candidate_scipy_sparse_matvec_s` | 7.111900e-05 |
| `candidate_slepc_available` | 0.000000e+00 |
