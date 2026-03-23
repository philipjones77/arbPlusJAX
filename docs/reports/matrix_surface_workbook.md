Last updated: 2026-03-20T00:00:00Z

# Matrix Surface Workbook

This workbook summarizes the benchmark surfaces for dense, sparse, and matrix-free matrix families.

It is intended to make the current comparison surface legible while the matrix API and execution-strategy hardening continues.

The workbook now compares dense, sparse, block-sparse, variable-block sparse, and matrix-free/operator-plan execution in one place.

## Compare and Contrast

| family | use when | fastest metrics |
| --- | --- | --- |
| dense | matrices are small/medium and direct kernels or cached dense plans are appropriate | `acb_dense_plan_prepare_s`=8.033e-05s, `arb_dense_plan_prepare_s`=1.054e-04s, `acb_transpose_s`=2.782e-02s, `arb_diag_s`=2.949e-02s |
| sparse | storage sparsity is meaningful and callers want sparse cached matvec/rmatvec reuse | `scb_csr_point_cached_matvec_s`=6.371e-04s, `scb_bcoo_point_cached_matvec_s`=8.097e-04s, `scb_bcoo_point_matvec_s`=8.239e-04s, `srb_bcoo_point_cached_matvec_s`=9.356e-04s |
| block sparse | block structure is explicit and callers want block-native apply paths | `scb_block_adjoint_cached_s`=4.423e-02s, `srb_block_rmatvec_cached_s`=4.646e-02s, `scb_block_matvec_s`=9.986e-02s, `srb_block_matvec_s`=1.215e-01s |
| variable block sparse | partitions are irregular but structure should still be preserved | `srb_vblock_matvec_cached_s`=8.325e-02s, `scb_vblock_matvec_cached_s`=9.021e-02s, `scb_vblock_matvec_s`=1.215e-01s, `srb_vblock_matvec_s`=1.270e-01s |
| matrix free | operator plans, Krylov solves, logdet, or adapter-based execution matter more than explicit materialization | `complex_apply_plan_warm_s`=5.561e-06s, `real_apply_plan_warm_s`=6.194e-06s, `real_apply_warm_s`=7.075e-06s, `complex_inverse_action_plan_warm_s`=7.499e-06s |

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
| `acb_cached_matvec_padded_s` | 4.238790e-01 |
| `acb_cached_matvec_s` | 1.858423e-01 |
| `acb_conjugate_transpose_s` | 3.220265e-02 |
| `acb_dense_plan_prepare_s` | 8.032900e-05 |
| `acb_diag_s` | 3.708005e-02 |
| `acb_direct_solve_s` | 1.752255e-01 |
| `acb_lu_reuse_s` | 1.546849e-01 |
| `acb_transpose_s` | 2.781926e-02 |
| `arb_cached_matvec_padded_s` | 1.572152e+00 |
| `arb_cached_matvec_s` | 9.974235e-02 |
| `arb_dense_plan_prepare_s` | 1.054140e-04 |
| `arb_diag_s` | 2.948604e-02 |
| `arb_direct_solve_s` | 1.606767e-01 |
| `arb_lu_reuse_s` | 1.070683e-01 |
| `arb_transpose_s` | 3.462801e-02 |

## Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_bcoo_basic_cached_matvec_s` | 1.466856e-02 |
| `scb_bcoo_basic_matvec_s` | 2.447295e-02 |
| `scb_bcoo_point_cached_matvec_s` | 8.097180e-04 |
| `scb_bcoo_point_matvec_s` | 8.238810e-04 |
| `scb_coo_basic_cached_matvec_s` | 1.944551e-02 |
| `scb_coo_basic_matvec_s` | 9.148074e-02 |
| `scb_coo_point_cached_matvec_s` | 1.245786e-03 |
| `scb_coo_point_matvec_s` | 9.169603e-02 |
| `scb_csr_basic_cached_matvec_s` | 1.140426e-02 |
| `scb_csr_basic_matvec_s` | 2.203525e-02 |
| `scb_csr_point_cached_matvec_s` | 6.370610e-04 |
| `scb_csr_point_matvec_s` | 1.804650e-03 |
| `srb_bcoo_basic_cached_matvec_s` | 4.350597e-03 |
| `srb_bcoo_basic_matvec_s` | 5.470142e-03 |
| `srb_bcoo_point_cached_matvec_s` | 9.355690e-04 |
| `srb_bcoo_point_matvec_s` | 1.045123e-03 |
| `srb_coo_basic_cached_matvec_s` | 5.272565e-03 |
| `srb_coo_basic_matvec_s` | 6.576780e-01 |
| `srb_coo_point_cached_matvec_s` | 1.216750e-03 |
| `srb_coo_point_matvec_s` | 1.013353e-01 |
| `srb_csr_basic_cached_matvec_s` | 4.757788e-03 |
| `srb_csr_basic_matvec_s` | 6.658176e-03 |
| `srb_csr_point_cached_matvec_s` | 1.221102e-03 |
| `srb_csr_point_matvec_s` | 2.475799e-03 |

## Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_block_sparse_matrix_surface.py --n-blocks 2 --block-size 2 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_block_adjoint_cached_s` | 4.423067e-02 |
| `scb_block_matvec_s` | 9.986199e-02 |
| `srb_block_matvec_s` | 1.215414e-01 |
| `srb_block_rmatvec_cached_s` | 4.646058e-02 |

## Variable-Block Sparse Matrix Surface

Command:

```bash
python benchmarks/benchmark_vblock_sparse_matrix_surface.py --n 4 --warmup 0 --runs 1 --dtype float64 --smoke
```

| metric | seconds |
| --- | ---: |
| `scb_vblock_matvec_cached_s` | 9.021463e-02 |
| `scb_vblock_matvec_s` | 1.215449e-01 |
| `srb_vblock_matvec_cached_s` | 8.324801e-02 |
| `srb_vblock_matvec_s` | 1.269606e-01 |

## Matrix-Free Surface

Command:

```bash
python benchmarks/benchmark_matrix_free_krylov.py
```

| metric | seconds |
| --- | ---: |
| `complex_action_cold_s` | 2.969981e-01 |
| `complex_action_plan_cold_s` | 2.601386e-01 |
| `complex_action_plan_precompile_s` | 7.212600e-05 |
| `complex_action_plan_warm_s` | 1.996640e-05 |
| `complex_apply_cold_s` | 9.109244e-02 |
| `complex_apply_plan_cold_s` | 8.870064e-02 |
| `complex_apply_plan_precompile_s` | 2.344400e-05 |
| `complex_apply_plan_warm_s` | 5.561200e-06 |
| `complex_det_cold_s` | 3.592282e-01 |
| `complex_det_plan_cold_s` | 3.446682e-01 |
| `complex_det_plan_precompile_s` | 1.596170e-04 |
| `complex_eigsh_cold_s` | 2.157348e-01 |
| `complex_eigsh_plan_cold_s` | 2.054601e-01 |
| `complex_eigsh_restarted_plan_compile_s` | 3.831667e-01 |
| `complex_eigsh_restarted_plan_execute_s` | 2.117730e-04 |
| `complex_grad_cold_s` | 3.604235e-01 |
| `complex_inverse_action_cold_s` | 1.999706e-01 |
| `complex_inverse_action_plan_cold_s` | 2.016802e-01 |
| `complex_inverse_action_plan_compile_s` | 2.511800e-05 |
| `complex_inverse_action_plan_execute_s` | 8.201000e-06 |
| `complex_inverse_action_plan_warm_s` | 7.499000e-06 |
| `complex_inverse_grad_plan_compile_s` | 8.012567e-01 |
| `complex_inverse_grad_plan_execute_s` | 3.977800e-05 |
| `complex_logdet_cold_s` | 6.183087e-01 |
| `complex_logdet_grad_cold_s` | 7.119775e-01 |
| `complex_logdet_grad_compile_s` | 2.465880e-04 |
| `complex_logdet_grad_execute_s` | 8.949500e-05 |
| `complex_logdet_plan_cold_s` | 3.298154e-01 |
| `complex_logdet_plan_precompile_s` | 1.932640e-04 |
| `complex_logdet_plan_warm_s` | 8.134480e-05 |
| `complex_minres_plan_cold_s` | 3.081224e-01 |
| `complex_minres_plan_compile_s` | 1.502190e-04 |
| `complex_minres_plan_execute_s` | 1.878080e-04 |
| `complex_multi_shift_plan_compile_s` | 2.413502e-01 |
| `complex_multi_shift_plan_execute_s` | 6.697400e-05 |
| `complex_multi_shift_plan_warm_s` | 1.107720e-05 |
| `complex_restarted_action_cold_s` | 2.678221e-01 |
| `complex_restarted_action_plan_cold_s` | 2.911600e-01 |
| `complex_solve_action_cold_s` | 1.450001e-01 |
| `complex_solve_action_plan_cold_s` | 2.015451e-01 |
| `complex_solve_action_plan_compile_s` | 7.651800e-05 |
| `complex_solve_action_plan_execute_s` | 1.255500e-05 |
| `complex_solve_action_plan_warm_s` | 8.008000e-06 |
| `complex_solve_grad_plan_compile_s` | 6.558138e-01 |
| `complex_solve_grad_plan_execute_s` | 4.130700e-05 |
| `real_action_cold_s` | 2.542954e-01 |
| `real_action_plan_cold_s` | 2.307712e-01 |
| `real_action_plan_precompile_s` | 2.198300e-04 |
| `real_action_plan_warm_s` | 5.012600e-05 |
| `real_action_warm_s` | 6.473580e-05 |
| `real_apply_cold_s` | 8.769890e-02 |
| `real_apply_plan_cold_s` | 9.941116e-02 |
| `real_apply_plan_precompile_s` | 1.214490e-04 |
| `real_apply_plan_warm_s` | 6.193800e-06 |
| `real_apply_warm_s` | 7.075000e-06 |
| `real_det_cold_s` | 1.979996e-01 |
| `real_det_plan_cold_s` | 2.215419e-01 |
| `real_det_plan_precompile_s` | 1.366370e-04 |
| `real_eigsh_cold_s` | 2.022429e-01 |
| `real_eigsh_plan_cold_s` | 1.811258e-01 |
| `real_eigsh_restarted_plan_compile_s` | 2.712214e-01 |
| `real_eigsh_restarted_plan_execute_s` | 3.329510e-04 |
| `real_grad_cold_s` | 2.720730e-01 |
| `real_inverse_action_cold_s` | 1.350562e-01 |
| `real_inverse_action_plan_cold_s` | 1.448046e-01 |
| `real_inverse_action_plan_compile_s` | 9.743900e-05 |
| `real_inverse_action_plan_execute_s` | 2.346000e-05 |
| `real_inverse_action_plan_warm_s` | 9.919400e-06 |
| `real_inverse_grad_plan_compile_s` | 6.644563e-01 |
| `real_inverse_grad_plan_execute_s` | 1.042210e-04 |
| `real_logdet_cold_s` | 2.365365e-01 |
| `real_logdet_grad_cold_s` | 2.851831e-01 |
| `real_logdet_grad_compile_s` | 1.938080e-04 |
| `real_logdet_grad_execute_s` | 9.029300e-05 |
| `real_logdet_plan_cold_s` | 2.098460e-01 |
| `real_logdet_plan_precompile_s` | 1.634900e-04 |
| `real_logdet_plan_warm_s` | 8.162480e-05 |
| `real_logdet_warm_s` | 6.092720e-05 |
| `real_minres_plan_cold_s` | 2.507006e-01 |
| `real_minres_plan_compile_s` | 1.825050e-04 |
| `real_minres_plan_execute_s` | 2.477750e-04 |
| `real_multi_shift_plan_compile_s` | 1.812174e-01 |
| `real_multi_shift_plan_execute_s` | 6.367500e-05 |
| `real_multi_shift_plan_warm_s` | 1.495860e-05 |
| `real_restarted_action_cold_s` | 2.008667e-01 |
| `real_restarted_action_plan_cold_s` | 2.106661e-01 |
| `real_solve_action_cold_s` | 1.388257e-01 |
| `real_solve_action_plan_cold_s` | 1.420096e-01 |
| `real_solve_action_plan_compile_s` | 9.975700e-05 |
| `real_solve_action_plan_execute_s` | 2.192700e-05 |
| `real_solve_action_plan_warm_s` | 1.010540e-05 |
| `real_solve_grad_plan_compile_s` | 5.153149e-01 |
| `real_solve_grad_plan_execute_s` | 5.080900e-05 |
| `sparse_complex_action_plan_s` | 2.715057e-01 |
| `sparse_complex_action_s` | 2.961253e-01 |
| `sparse_complex_apply_plan_s` | 8.701920e-02 |
| `sparse_complex_apply_s` | 8.710970e-02 |
| `sparse_complex_det_plan_s` | 3.553748e-01 |
| `sparse_complex_det_s` | 3.964406e-01 |
| `sparse_complex_inverse_action_plan_s` | 1.882911e-01 |
| `sparse_complex_inverse_action_s` | 1.726287e-01 |
| `sparse_complex_logdet_plan_s` | 3.484414e-01 |
| `sparse_complex_logdet_s` | 3.759796e-01 |
| `sparse_complex_restarted_plan_s` | 2.536773e-01 |
| `sparse_complex_restarted_s` | 2.977958e-01 |
| `sparse_complex_solve_action_plan_s` | 2.258494e-01 |
| `sparse_complex_solve_action_s` | 1.869132e-01 |
| `sparse_real_apply_plan_s` | 7.616323e-02 |
| `sparse_real_apply_s` | 8.059587e-02 |
| `sparse_real_det_plan_s` | 2.406366e-01 |
| `sparse_real_det_s` | 2.946812e-01 |
| `sparse_real_inverse_action_plan_s` | 1.697439e-01 |
| `sparse_real_inverse_action_s` | 1.648694e-01 |
| `sparse_real_inverse_diag_corrected_s` | 1.205220e+00 |
| `sparse_real_inverse_diag_local_s` | 1.673623e-01 |
| `sparse_real_logdet_grad_s` | 4.296443e-01 |
| `sparse_real_logdet_leja_hutchpp_auto_s` | 1.196568e+00 |
| `sparse_real_logdet_leja_hutchpp_s` | 7.594973e-01 |
| `sparse_real_logdet_plan_s` | 2.349319e-01 |
| `sparse_real_logdet_s` | 2.929648e-01 |
| `sparse_real_solve_action_plan_s` | 1.732246e-01 |
| `sparse_real_solve_action_s` | 1.401309e-01 |

## Matrix Backend Candidates

Command:

```bash
python benchmarks/benchmark_matrix_backend_candidates.py --n 4 --warmup 0 --runs 1
```

| metric | seconds |
| --- | ---: |
| `candidate_arbplusjax_sparse_cached_matvec_s` | 1.368416e-03 |
| `candidate_arbplusjax_sparse_fromdense_solve_s` | 3.870823e-01 |
| `candidate_arbplusjax_sparse_matvec_s` | 1.580229e-03 |
| `candidate_arbplusjax_sparse_spd_solve_s` | 2.725112e-01 |
| `candidate_jax_dense_eigh_s` | 5.033759e-02 |
| `candidate_jax_dense_matvec_s` | 2.458814e-02 |
| `candidate_jax_dense_solve_s` | 8.456442e-02 |
| `candidate_jax_experimental_sparse_cg_s` | 1.202031e-01 |
| `candidate_jax_experimental_sparse_matvec_s` | 5.961724e-02 |
| `candidate_jax_scipy_dense_solve_s` | 6.754875e-02 |
| `candidate_matfree_apply_s` | 2.587817e-03 |
| `candidate_matfree_logdet_slq_s` | 2.533430e-01 |
| `candidate_matfree_solve_action_s` | 1.297780e-04 |
| `candidate_petsc_available` | 0.000000e+00 |
| `candidate_scipy_dense_eigh_s` | 1.433940e-04 |
| `candidate_scipy_dense_matvec_s` | 4.028900e-05 |
| `candidate_scipy_dense_solve_s` | 2.483300e-04 |
| `candidate_scipy_linear_operator_cg_s` | 1.784700e-04 |
| `candidate_scipy_sparse_cg_s` | 2.591950e-04 |
| `candidate_scipy_sparse_eigsh_s` | 3.622970e-04 |
| `candidate_scipy_sparse_matvec_s` | 7.344500e-05 |
| `candidate_slepc_available` | 0.000000e+00 |
