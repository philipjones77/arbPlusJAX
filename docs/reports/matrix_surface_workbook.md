Last updated: 2026-03-20T00:00:00Z

# Matrix Surface Workbook

This workbook summarizes the benchmark surfaces for dense, sparse, and matrix-free matrix families.

It is intended to make the current comparison surface legible while the matrix API and execution-strategy hardening continues.

The workbook now compares dense, sparse, block-sparse, variable-block sparse, and matrix-free/operator-plan execution in one place.

## Compare and Contrast

| family | use when | fastest metrics |
| --- | --- | --- |
| dense | matrices are small/medium and direct kernels or cached dense plans are appropriate | `acb_dense_plan_prepare_s`=8.304e-05s, `arb_dense_plan_prepare_s`=1.380e-04s, `arb_transpose_s`=3.278e-02s, `acb_diag_s`=3.313e-02s |
| sparse | storage sparsity is meaningful and callers want sparse cached matvec/rmatvec reuse | `scb_csr_point_cached_matvec_s`=8.734e-04s, `srb_csr_point_cached_matvec_s`=1.240e-03s, `srb_bcoo_point_cached_matvec_s`=1.442e-03s, `srb_coo_point_cached_matvec_s`=1.848e-03s |
| block sparse | block structure is explicit and callers want block-native apply paths | `scb_block_adjoint_cached_s`=6.039e-02s, `srb_block_rmatvec_cached_s`=6.212e-02s, `scb_block_matvec_s`=1.231e-01s, `srb_block_matvec_s`=1.309e-01s |
| variable block sparse | partitions are irregular but structure should still be preserved | `srb_vblock_matvec_cached_s`=1.093e-01s, `scb_vblock_matvec_cached_s`=1.147e-01s, `srb_vblock_matvec_s`=1.623e-01s, `scb_vblock_matvec_s`=1.715e-01s |
| matrix free | operator plans, Krylov solves, logdet, or adapter-based execution matter more than explicit materialization | `real_apply_plan_warm_s`=6.480e-06s, `complex_apply_plan_warm_s`=7.298e-06s, `real_apply_warm_s`=8.823e-06s, `complex_inverse_action_plan_warm_s`=9.119e-06s |

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
| `acb_cached_matvec_padded_s` | 5.065365e-01 |
| `acb_cached_matvec_s` | 2.233829e-01 |
| `acb_conjugate_transpose_s` | 4.418868e-02 |
| `acb_dense_plan_prepare_s` | 8.304500e-05 |
| `acb_diag_s` | 3.313325e-02 |
| `acb_direct_solve_s` | 2.214591e-01 |
| `acb_lu_reuse_s` | 1.684281e-01 |
| `acb_transpose_s` | 3.974443e-02 |
| `arb_cached_matvec_padded_s` | 1.900509e+00 |
| `arb_cached_matvec_s` | 1.451563e-01 |
| `arb_dense_plan_prepare_s` | 1.379920e-04 |
| `arb_diag_s` | 4.811159e-02 |
| `arb_direct_solve_s` | 2.215061e-01 |
| `arb_lu_reuse_s` | 1.159537e-01 |
| `arb_transpose_s` | 3.278383e-02 |

## Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_bcoo_basic_cached_matvec_s` | 1.804021e-02 |
| `scb_bcoo_basic_matvec_s` | 1.751101e-02 |
| `scb_bcoo_point_cached_matvec_s` | 2.317443e-03 |
| `scb_bcoo_point_matvec_s` | 1.900614e-03 |
| `scb_coo_basic_cached_matvec_s` | 2.428177e-02 |
| `scb_coo_basic_matvec_s` | 1.226414e-01 |
| `scb_coo_point_cached_matvec_s` | 1.964275e-03 |
| `scb_coo_point_matvec_s` | 1.278651e-01 |
| `scb_csr_basic_cached_matvec_s` | 1.733833e-02 |
| `scb_csr_basic_matvec_s` | 2.200375e-02 |
| `scb_csr_point_cached_matvec_s` | 8.733790e-04 |
| `scb_csr_point_matvec_s` | 2.539479e-03 |
| `srb_bcoo_basic_cached_matvec_s` | 9.500396e-03 |
| `srb_bcoo_basic_matvec_s` | 1.012433e-02 |
| `srb_bcoo_point_cached_matvec_s` | 1.441874e-03 |
| `srb_bcoo_point_matvec_s` | 1.933427e-03 |
| `srb_coo_basic_cached_matvec_s` | 7.591840e-03 |
| `srb_coo_basic_matvec_s` | 8.863359e-01 |
| `srb_coo_point_cached_matvec_s` | 1.847904e-03 |
| `srb_coo_point_matvec_s` | 1.646150e-01 |
| `srb_csr_basic_cached_matvec_s` | 7.031799e-03 |
| `srb_csr_basic_matvec_s` | 1.793177e-02 |
| `srb_csr_point_cached_matvec_s` | 1.239533e-03 |
| `srb_csr_point_matvec_s` | 3.829410e-03 |

## Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_block_sparse_matrix_surface.py --n-blocks 2 --block-size 2 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_block_adjoint_cached_s` | 6.039450e-02 |
| `scb_block_matvec_s` | 1.230870e-01 |
| `srb_block_matvec_s` | 1.309448e-01 |
| `srb_block_rmatvec_cached_s` | 6.212182e-02 |

## Variable-Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_vblock_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_vblock_matvec_cached_s` | 1.147152e-01 |
| `scb_vblock_matvec_s` | 1.714558e-01 |
| `srb_vblock_matvec_cached_s` | 1.092893e-01 |
| `srb_vblock_matvec_s` | 1.623408e-01 |

## Matrix-Free Surface

Command:

```bash
python benchmarks/benchmark_matrix_free_krylov.py
```

| metric | seconds |
| --- | ---: |
| `complex_action_cold_s` | 4.040178e-01 |
| `complex_action_plan_cold_s` | 4.113455e-01 |
| `complex_action_plan_precompile_s` | 1.039920e-04 |
| `complex_action_plan_warm_s` | 1.690700e-05 |
| `complex_apply_cold_s` | 9.915735e-02 |
| `complex_apply_plan_cold_s` | 1.182559e-01 |
| `complex_apply_plan_precompile_s` | 4.111000e-05 |
| `complex_apply_plan_warm_s` | 7.298400e-06 |
| `complex_det_cold_s` | 5.159846e-01 |
| `complex_det_plan_cold_s` | 6.729449e-01 |
| `complex_det_plan_precompile_s` | 2.745970e-04 |
| `complex_eigsh_cold_s` | 2.575538e-01 |
| `complex_eigsh_plan_cold_s` | 3.446223e-01 |
| `complex_eigsh_restarted_plan_compile_s` | 6.159356e-01 |
| `complex_eigsh_restarted_plan_execute_s` | 1.638720e-04 |
| `complex_grad_cold_s` | 4.766313e-01 |
| `complex_inverse_action_cold_s` | 2.708236e-01 |
| `complex_inverse_action_plan_cold_s` | 2.432282e-01 |
| `complex_inverse_action_plan_compile_s` | 9.144700e-05 |
| `complex_inverse_action_plan_execute_s` | 4.883000e-05 |
| `complex_inverse_action_plan_warm_s` | 9.119400e-06 |
| `complex_inverse_grad_plan_compile_s` | 1.202511e+00 |
| `complex_inverse_grad_plan_execute_s` | 5.051400e-05 |
| `complex_logdet_cold_s` | 8.827566e-01 |
| `complex_logdet_grad_cold_s` | 1.279593e+00 |
| `complex_logdet_grad_compile_s` | 2.613560e-04 |
| `complex_logdet_grad_execute_s` | 2.120210e-04 |
| `complex_logdet_plan_cold_s` | 5.424063e-01 |
| `complex_logdet_plan_precompile_s` | 2.291080e-04 |
| `complex_logdet_plan_warm_s` | 2.427218e-04 |
| `complex_minres_plan_cold_s` | 4.115787e-01 |
| `complex_minres_plan_compile_s` | 2.254910e-04 |
| `complex_minres_plan_execute_s` | 1.573030e-04 |
| `complex_multi_shift_plan_compile_s` | 3.750336e-01 |
| `complex_multi_shift_plan_execute_s` | 8.235200e-05 |
| `complex_multi_shift_plan_warm_s` | 2.335500e-05 |
| `complex_restarted_action_cold_s` | 3.689029e-01 |
| `complex_restarted_action_plan_cold_s` | 3.912313e-01 |
| `complex_solve_action_cold_s` | 2.779179e-01 |
| `complex_solve_action_plan_cold_s` | 2.778845e-01 |
| `complex_solve_action_plan_compile_s` | 1.917230e-04 |
| `complex_solve_action_plan_execute_s` | 7.247300e-05 |
| `complex_solve_action_plan_warm_s` | 1.043820e-05 |
| `complex_solve_grad_plan_compile_s` | 1.148366e+00 |
| `complex_solve_grad_plan_execute_s` | 1.203570e-04 |
| `real_action_cold_s` | 2.622594e-01 |
| `real_action_plan_cold_s` | 2.432905e-01 |
| `real_action_plan_precompile_s` | 1.109830e-04 |
| `real_action_plan_warm_s` | 5.975740e-05 |
| `real_action_warm_s` | 5.514360e-05 |
| `real_apply_cold_s` | 8.159445e-02 |
| `real_apply_plan_cold_s` | 8.669242e-02 |
| `real_apply_plan_precompile_s` | 8.927800e-05 |
| `real_apply_plan_warm_s` | 6.479600e-06 |
| `real_apply_warm_s` | 8.823400e-06 |
| `real_det_cold_s` | 3.188969e-01 |
| `real_det_plan_cold_s` | 3.286085e-01 |
| `real_det_plan_precompile_s` | 9.339500e-05 |
| `real_eigsh_cold_s` | 2.749327e-01 |
| `real_eigsh_plan_cold_s` | 3.196408e-01 |
| `real_eigsh_restarted_plan_compile_s` | 4.378653e-01 |
| `real_eigsh_restarted_plan_execute_s` | 4.303690e-04 |
| `real_grad_cold_s` | 3.965302e-01 |
| `real_inverse_action_cold_s` | 1.775269e-01 |
| `real_inverse_action_plan_cold_s` | 2.170926e-01 |
| `real_inverse_action_plan_compile_s` | 2.980900e-05 |
| `real_inverse_action_plan_execute_s` | 1.139300e-05 |
| `real_inverse_action_plan_warm_s` | 1.040720e-05 |
| `real_inverse_grad_plan_compile_s` | 7.913006e-01 |
| `real_inverse_grad_plan_execute_s` | 5.700800e-05 |
| `real_logdet_cold_s` | 3.630113e-01 |
| `real_logdet_grad_cold_s` | 4.563099e-01 |
| `real_logdet_grad_compile_s` | 1.946140e-04 |
| `real_logdet_grad_execute_s` | 9.901600e-05 |
| `real_logdet_plan_cold_s` | 3.135958e-01 |
| `real_logdet_plan_precompile_s` | 1.336440e-04 |
| `real_logdet_plan_warm_s` | 1.422922e-04 |
| `real_logdet_warm_s` | 7.599140e-05 |
| `real_minres_plan_cold_s` | 3.502002e-01 |
| `real_minres_plan_compile_s` | 3.608350e-04 |
| `real_minres_plan_execute_s` | 2.960390e-04 |
| `real_multi_shift_plan_compile_s` | 2.649586e-01 |
| `real_multi_shift_plan_execute_s` | 8.155100e-05 |
| `real_multi_shift_plan_warm_s` | 1.513180e-05 |
| `real_restarted_action_cold_s` | 2.734017e-01 |
| `real_restarted_action_plan_cold_s` | 2.603520e-01 |
| `real_solve_action_cold_s` | 1.960775e-01 |
| `real_solve_action_plan_cold_s` | 2.005324e-01 |
| `real_solve_action_plan_compile_s` | 1.245260e-04 |
| `real_solve_action_plan_execute_s` | 3.305500e-05 |
| `real_solve_action_plan_warm_s` | 1.029100e-05 |
| `real_solve_grad_plan_compile_s` | 7.358362e-01 |
| `real_solve_grad_plan_execute_s` | 1.184530e-04 |
| `sparse_complex_action_plan_s` | 3.525812e-01 |
| `sparse_complex_action_s` | 4.324199e-01 |
| `sparse_complex_apply_plan_s` | 1.637571e-01 |
| `sparse_complex_apply_s` | 1.423651e-01 |
| `sparse_complex_det_plan_s` | 5.201460e-01 |
| `sparse_complex_det_s` | 7.031942e-01 |
| `sparse_complex_inverse_action_plan_s` | 2.875551e-01 |
| `sparse_complex_inverse_action_s` | 3.402363e-01 |
| `sparse_complex_logdet_plan_s` | 5.844197e-01 |
| `sparse_complex_logdet_s` | 6.560907e-01 |
| `sparse_complex_restarted_plan_s` | 3.897293e-01 |
| `sparse_complex_restarted_s` | 4.065593e-01 |
| `sparse_complex_solve_action_plan_s` | 2.837651e-01 |
| `sparse_complex_solve_action_s` | 3.456184e-01 |
| `sparse_real_apply_plan_s` | 1.222358e-01 |
| `sparse_real_apply_s` | 1.228492e-01 |
| `sparse_real_det_plan_s` | 3.332223e-01 |
| `sparse_real_det_s` | 3.779029e-01 |
| `sparse_real_inverse_action_plan_s` | 2.245275e-01 |
| `sparse_real_inverse_action_s` | 2.155542e-01 |
| `sparse_real_inverse_diag_corrected_s` | 1.828591e+00 |
| `sparse_real_inverse_diag_local_s` | 1.774326e-01 |
| `sparse_real_logdet_grad_s` | 5.327252e-01 |
| `sparse_real_logdet_leja_hutchpp_auto_s` | 1.692769e+00 |
| `sparse_real_logdet_leja_hutchpp_s` | 1.282930e+00 |
| `sparse_real_logdet_plan_s` | 3.795407e-01 |
| `sparse_real_logdet_s` | 4.654471e-01 |
| `sparse_real_solve_action_plan_s` | 3.648940e-01 |
| `sparse_real_solve_action_s` | 2.104105e-01 |

## Matrix Backend Candidates

Command:

```bash
python benchmarks/benchmark_matrix_backend_candidates.py --n 4 --warmup 0 --runs 1
```

| metric | seconds |
| --- | ---: |
| `candidate_arbplusjax_sparse_cached_matvec_s` | 1.629561e-03 |
| `candidate_arbplusjax_sparse_fromdense_solve_s` | 5.598420e-01 |
| `candidate_arbplusjax_sparse_matvec_s` | 2.853764e-03 |
| `candidate_arbplusjax_sparse_spd_solve_s` | 5.326112e-01 |
| `candidate_jax_dense_eigh_s` | 6.282072e-02 |
| `candidate_jax_dense_matvec_s` | 5.472111e-02 |
| `candidate_jax_dense_solve_s` | 1.245307e-01 |
| `candidate_jax_experimental_sparse_cg_s` | 2.662823e-01 |
| `candidate_jax_experimental_sparse_matvec_s` | 1.367673e-01 |
| `candidate_jax_scipy_dense_solve_s` | 9.158801e-02 |
| `candidate_matfree_apply_s` | 6.479904e-03 |
| `candidate_matfree_logdet_slq_s` | 4.148925e-01 |
| `candidate_matfree_solve_action_s` | 9.170000e-05 |
| `candidate_petsc_available` | 0.000000e+00 |
| `candidate_scipy_dense_eigh_s` | 1.022288e-03 |
| `candidate_scipy_dense_matvec_s` | 1.497170e-04 |
| `candidate_scipy_dense_solve_s` | 1.350081e-03 |
| `candidate_scipy_linear_operator_cg_s` | 3.415120e-04 |
| `candidate_scipy_sparse_cg_s` | 5.780230e-04 |
| `candidate_scipy_sparse_eigsh_s` | 1.292459e-03 |
| `candidate_scipy_sparse_matvec_s` | 1.676300e-04 |
| `candidate_slepc_available` | 0.000000e+00 |
