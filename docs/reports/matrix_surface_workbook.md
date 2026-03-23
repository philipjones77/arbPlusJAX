Last updated: 2026-03-20T00:00:00Z

# Matrix Surface Workbook

This workbook summarizes the benchmark surfaces for dense, sparse, and matrix-free matrix families.

It is intended to make the current comparison surface legible while the matrix API and execution-strategy hardening continues.

The workbook now compares dense, sparse, block-sparse, variable-block sparse, and matrix-free/operator-plan execution in one place.

## Compare and Contrast

| family | use when | fastest metrics |
| --- | --- | --- |
| dense | matrices are small/medium and direct kernels or cached dense plans are appropriate | `acb_dense_plan_prepare_s`=1.218e-04s, `arb_dense_plan_prepare_s`=1.315e-04s, `arb_diag_s`=3.489e-02s, `acb_diag_s`=3.553e-02s |
| sparse | storage sparsity is meaningful and callers want sparse cached matvec/rmatvec reuse | `srb_bcoo_point_cached_matvec_s`=9.333e-04s, `srb_bcoo_point_matvec_s`=1.108e-03s, `scb_coo_point_cached_matvec_s`=1.423e-03s, `scb_csr_point_cached_matvec_s`=1.429e-03s |
| block sparse | block structure is explicit and callers want block-native apply paths | `scb_block_adjoint_cached_s`=4.682e-02s, `srb_block_rmatvec_cached_s`=5.742e-02s, `scb_block_matvec_s`=1.076e-01s, `srb_block_matvec_s`=1.276e-01s |
| variable block sparse | partitions are irregular but structure should still be preserved | `scb_vblock_matvec_cached_s`=8.259e-02s, `srb_vblock_matvec_cached_s`=1.292e-01s, `srb_vblock_matvec_s`=1.668e-01s, `scb_vblock_matvec_s`=1.840e-01s |
| matrix free | operator plans, Krylov solves, logdet, or adapter-based execution matter more than explicit materialization | `complex_apply_plan_warm_s`=8.277e-06s, `real_apply_plan_warm_s`=9.241e-06s, `real_apply_warm_s`=1.009e-05s, `complex_inverse_action_plan_warm_s`=1.138e-05s |

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
| `acb_cached_matvec_padded_s` | 5.588916e-01 |
| `acb_cached_matvec_s` | 2.194513e-01 |
| `acb_conjugate_transpose_s` | 4.739950e-02 |
| `acb_dense_plan_prepare_s` | 1.218310e-04 |
| `acb_diag_s` | 3.552767e-02 |
| `acb_direct_solve_s` | 2.294816e-01 |
| `acb_lu_reuse_s` | 1.653208e-01 |
| `acb_transpose_s` | 4.301039e-02 |
| `arb_cached_matvec_padded_s` | 2.304145e+00 |
| `arb_cached_matvec_s` | 1.518638e-01 |
| `arb_dense_plan_prepare_s` | 1.315370e-04 |
| `arb_diag_s` | 3.488522e-02 |
| `arb_direct_solve_s` | 1.748759e-01 |
| `arb_lu_reuse_s` | 9.431088e-02 |
| `arb_transpose_s` | 3.758419e-02 |

## Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_bcoo_basic_cached_matvec_s` | 1.116964e-02 |
| `scb_bcoo_basic_matvec_s` | 1.395153e-02 |
| `scb_bcoo_point_cached_matvec_s` | 2.138223e-03 |
| `scb_bcoo_point_matvec_s` | 1.641426e-03 |
| `scb_coo_basic_cached_matvec_s` | 2.600686e-02 |
| `scb_coo_basic_matvec_s` | 1.417826e-01 |
| `scb_coo_point_cached_matvec_s` | 1.423302e-03 |
| `scb_coo_point_matvec_s` | 1.135548e-01 |
| `scb_csr_basic_cached_matvec_s` | 1.561703e-02 |
| `scb_csr_basic_matvec_s` | 2.605844e-02 |
| `scb_csr_point_cached_matvec_s` | 1.429469e-03 |
| `scb_csr_point_matvec_s` | 2.908141e-03 |
| `srb_bcoo_basic_cached_matvec_s` | 4.213686e-03 |
| `srb_bcoo_basic_matvec_s` | 4.470518e-03 |
| `srb_bcoo_point_cached_matvec_s` | 9.332890e-04 |
| `srb_bcoo_point_matvec_s` | 1.108204e-03 |
| `srb_coo_basic_cached_matvec_s` | 4.052683e-03 |
| `srb_coo_basic_matvec_s` | 7.041689e-01 |
| `srb_coo_point_cached_matvec_s` | 1.497857e-03 |
| `srb_coo_point_matvec_s` | 9.740457e-02 |
| `srb_csr_basic_cached_matvec_s` | 3.484345e-03 |
| `srb_csr_basic_matvec_s` | 5.496718e-03 |
| `srb_csr_point_cached_matvec_s` | 1.472139e-03 |
| `srb_csr_point_matvec_s` | 5.343896e-03 |

## Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_block_sparse_matrix_surface.py --n-blocks 2 --block-size 2 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_block_adjoint_cached_s` | 4.682102e-02 |
| `scb_block_matvec_s` | 1.075972e-01 |
| `srb_block_matvec_s` | 1.275584e-01 |
| `srb_block_rmatvec_cached_s` | 5.741942e-02 |

## Variable-Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_vblock_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_vblock_matvec_cached_s` | 8.259181e-02 |
| `scb_vblock_matvec_s` | 1.840287e-01 |
| `srb_vblock_matvec_cached_s` | 1.292497e-01 |
| `srb_vblock_matvec_s` | 1.667899e-01 |

## Matrix-Free Surface

Command:

```bash
python benchmarks/benchmark_matrix_free_krylov.py
```

| metric | seconds |
| --- | ---: |
| `complex_action_cold_s` | 3.119660e-01 |
| `complex_action_plan_cold_s` | 2.718491e-01 |
| `complex_action_plan_precompile_s` | 8.070500e-05 |
| `complex_action_plan_warm_s` | 2.572640e-05 |
| `complex_apply_cold_s` | 8.341750e-02 |
| `complex_apply_plan_cold_s` | 8.149062e-02 |
| `complex_apply_plan_precompile_s` | 5.956300e-05 |
| `complex_apply_plan_warm_s` | 8.276600e-06 |
| `complex_det_cold_s` | 3.389907e-01 |
| `complex_det_plan_cold_s` | 3.751346e-01 |
| `complex_det_plan_precompile_s` | 2.956460e-04 |
| `complex_eigsh_cold_s` | 2.129073e-01 |
| `complex_eigsh_plan_cold_s` | 2.027855e-01 |
| `complex_eigsh_restarted_plan_compile_s` | 4.020778e-01 |
| `complex_eigsh_restarted_plan_execute_s` | 2.502460e-04 |
| `complex_grad_cold_s` | 3.607845e-01 |
| `complex_inverse_action_cold_s` | 1.557067e-01 |
| `complex_inverse_action_plan_cold_s` | 1.888423e-01 |
| `complex_inverse_action_plan_compile_s` | 3.341600e-05 |
| `complex_inverse_action_plan_execute_s` | 1.497200e-05 |
| `complex_inverse_action_plan_warm_s` | 1.137880e-05 |
| `complex_inverse_grad_plan_compile_s` | 8.721297e-01 |
| `complex_inverse_grad_plan_execute_s` | 5.485300e-05 |
| `complex_logdet_cold_s` | 5.758502e-01 |
| `complex_logdet_grad_cold_s` | 7.255932e-01 |
| `complex_logdet_grad_compile_s` | 3.166430e-04 |
| `complex_logdet_grad_execute_s` | 2.646800e-04 |
| `complex_logdet_plan_cold_s` | 3.523685e-01 |
| `complex_logdet_plan_precompile_s` | 2.977430e-04 |
| `complex_logdet_plan_warm_s` | 1.750056e-04 |
| `complex_minres_plan_cold_s` | 2.733325e-01 |
| `complex_minres_plan_compile_s` | 1.836430e-04 |
| `complex_minres_plan_execute_s` | 1.533140e-04 |
| `complex_multi_shift_plan_compile_s` | 2.359097e-01 |
| `complex_multi_shift_plan_execute_s` | 7.108300e-05 |
| `complex_multi_shift_plan_warm_s` | 1.680000e-05 |
| `complex_restarted_action_cold_s` | 2.889466e-01 |
| `complex_restarted_action_plan_cold_s` | 3.022149e-01 |
| `complex_solve_action_cold_s` | 1.783524e-01 |
| `complex_solve_action_plan_cold_s` | 1.527104e-01 |
| `complex_solve_action_plan_compile_s` | 1.048670e-04 |
| `complex_solve_action_plan_execute_s` | 2.303600e-05 |
| `complex_solve_action_plan_warm_s` | 1.182840e-05 |
| `complex_solve_grad_plan_compile_s` | 8.023430e-01 |
| `complex_solve_grad_plan_execute_s` | 6.599700e-05 |
| `real_action_cold_s` | 2.584254e-01 |
| `real_action_plan_cold_s` | 2.166873e-01 |
| `real_action_plan_precompile_s` | 3.028130e-04 |
| `real_action_plan_warm_s` | 8.336580e-05 |
| `real_action_warm_s` | 6.088420e-05 |
| `real_apply_cold_s` | 6.170124e-02 |
| `real_apply_plan_cold_s` | 8.278685e-02 |
| `real_apply_plan_precompile_s` | 8.373900e-05 |
| `real_apply_plan_warm_s` | 9.241400e-06 |
| `real_apply_warm_s` | 1.008760e-05 |
| `real_det_cold_s` | 2.835375e-01 |
| `real_det_plan_cold_s` | 2.588153e-01 |
| `real_det_plan_precompile_s` | 2.707670e-04 |
| `real_eigsh_cold_s` | 2.119273e-01 |
| `real_eigsh_plan_cold_s` | 2.116096e-01 |
| `real_eigsh_restarted_plan_compile_s` | 3.434562e-01 |
| `real_eigsh_restarted_plan_execute_s` | 2.460140e-04 |
| `real_grad_cold_s` | 3.317071e-01 |
| `real_inverse_action_cold_s` | 1.589653e-01 |
| `real_inverse_action_plan_cold_s` | 1.736864e-01 |
| `real_inverse_action_plan_compile_s` | 3.761000e-05 |
| `real_inverse_action_plan_execute_s` | 1.734100e-05 |
| `real_inverse_action_plan_warm_s` | 1.570540e-05 |
| `real_inverse_grad_plan_compile_s` | 6.380686e-01 |
| `real_inverse_grad_plan_execute_s` | 6.313000e-05 |
| `real_logdet_cold_s` | 2.679952e-01 |
| `real_logdet_grad_cold_s` | 3.829312e-01 |
| `real_logdet_grad_compile_s` | 3.186690e-04 |
| `real_logdet_grad_execute_s` | 2.964980e-04 |
| `real_logdet_plan_cold_s` | 2.152403e-01 |
| `real_logdet_plan_precompile_s` | 2.867440e-04 |
| `real_logdet_plan_warm_s` | 1.840176e-04 |
| `real_logdet_warm_s` | 1.223962e-04 |
| `real_minres_plan_cold_s` | 2.652853e-01 |
| `real_minres_plan_compile_s` | 2.567750e-04 |
| `real_minres_plan_execute_s` | 3.111000e-04 |
| `real_multi_shift_plan_compile_s` | 2.204967e-01 |
| `real_multi_shift_plan_execute_s` | 7.701800e-05 |
| `real_multi_shift_plan_warm_s` | 2.544360e-05 |
| `real_restarted_action_cold_s` | 2.397967e-01 |
| `real_restarted_action_plan_cold_s` | 2.492924e-01 |
| `real_solve_action_cold_s` | 1.485377e-01 |
| `real_solve_action_plan_cold_s` | 1.557309e-01 |
| `real_solve_action_plan_compile_s` | 1.333950e-04 |
| `real_solve_action_plan_execute_s` | 3.877000e-05 |
| `real_solve_action_plan_warm_s` | 2.126180e-05 |
| `real_solve_grad_plan_compile_s` | 5.490322e-01 |
| `real_solve_grad_plan_execute_s` | 6.315000e-05 |
| `sparse_complex_action_plan_s` | 3.236579e-01 |
| `sparse_complex_action_s` | 3.572248e-01 |
| `sparse_complex_apply_plan_s` | 9.351706e-02 |
| `sparse_complex_apply_s` | 9.592507e-02 |
| `sparse_complex_det_plan_s` | 4.340947e-01 |
| `sparse_complex_det_s` | 4.287548e-01 |
| `sparse_complex_inverse_action_plan_s` | 1.861637e-01 |
| `sparse_complex_inverse_action_s` | 1.949061e-01 |
| `sparse_complex_logdet_plan_s` | 3.514028e-01 |
| `sparse_complex_logdet_s` | 4.276844e-01 |
| `sparse_complex_restarted_plan_s` | 2.880844e-01 |
| `sparse_complex_restarted_s` | 2.887076e-01 |
| `sparse_complex_solve_action_plan_s` | 2.003548e-01 |
| `sparse_complex_solve_action_s` | 2.265693e-01 |
| `sparse_real_apply_plan_s` | 8.645925e-02 |
| `sparse_real_apply_s` | 8.058667e-02 |
| `sparse_real_det_plan_s` | 2.769802e-01 |
| `sparse_real_det_s` | 3.117777e-01 |
| `sparse_real_inverse_action_plan_s` | 1.611684e-01 |
| `sparse_real_inverse_action_s` | 1.276134e-01 |
| `sparse_real_inverse_diag_corrected_s` | 1.316709e+00 |
| `sparse_real_inverse_diag_local_s` | 1.789338e-01 |
| `sparse_real_logdet_grad_s` | 5.006315e-01 |
| `sparse_real_logdet_leja_hutchpp_auto_s` | 1.281736e+00 |
| `sparse_real_logdet_leja_hutchpp_s` | 8.828997e-01 |
| `sparse_real_logdet_plan_s` | 2.865104e-01 |
| `sparse_real_logdet_s` | 4.004520e-01 |
| `sparse_real_solve_action_plan_s` | 1.599730e-01 |
| `sparse_real_solve_action_s` | 1.509873e-01 |

## Matrix Backend Candidates

Command:

```bash
python benchmarks/benchmark_matrix_backend_candidates.py --n 4 --warmup 0 --runs 1
```

| metric | seconds |
| --- | ---: |
| `candidate_arbplusjax_sparse_cached_matvec_s` | 9.733260e-04 |
| `candidate_arbplusjax_sparse_fromdense_solve_s` | 4.159340e-01 |
| `candidate_arbplusjax_sparse_matvec_s` | 2.046718e-03 |
| `candidate_arbplusjax_sparse_spd_solve_s` | 2.473608e-01 |
| `candidate_jax_dense_eigh_s` | 5.604154e-02 |
| `candidate_jax_dense_matvec_s` | 2.108117e-02 |
| `candidate_jax_dense_solve_s` | 8.524335e-02 |
| `candidate_jax_experimental_sparse_cg_s` | 1.181293e-01 |
| `candidate_jax_experimental_sparse_matvec_s` | 6.188982e-02 |
| `candidate_jax_scipy_dense_solve_s` | 7.630302e-02 |
| `candidate_matfree_apply_s` | 3.182955e-03 |
| `candidate_matfree_logdet_slq_s` | 2.346271e-01 |
| `candidate_matfree_solve_action_s` | 4.994100e-05 |
| `candidate_petsc_available` | 0.000000e+00 |
| `candidate_scipy_dense_eigh_s` | 1.005663e-03 |
| `candidate_scipy_dense_matvec_s` | 2.325738e-03 |
| `candidate_scipy_dense_solve_s` | 1.125340e-03 |
| `candidate_scipy_linear_operator_cg_s` | 1.269570e-04 |
| `candidate_scipy_sparse_cg_s` | 7.099280e-04 |
| `candidate_scipy_sparse_eigsh_s` | 1.422838e-03 |
| `candidate_scipy_sparse_matvec_s` | 1.033720e-03 |
| `candidate_slepc_available` | 0.000000e+00 |
