Last updated: 2026-03-29T00:00:00Z

# Matrix-Free Krylov Benchmark Report

## Command

```bash
python benchmarks/benchmark_matrix_free_krylov.py
```

Runtime note from the retained run:

- CPU and GPU retained runs now both exist in the current WSL `jax` environment
- `JAX_PLATFORMS=cuda` is the stable matrix-free GPU path; the runtime still emits a non-fatal driver-version parse warning, but retained GPU benchmark and notebook execution now complete
- `--startup-prewarm` is now available when the benchmark should represent steady-state reuse rather than first-hit compile cost

## Results

### Real dense

- `real_apply_cold_s`: `0.106085`
- `real_apply_plan_cold_s`: `0.099339`
- `real_rapply_plan_cold_s`: `0.088194`
- `real_action_cold_s`: `1.037866`
- `real_action_plan_cold_s`: `0.395755`
- `real_logdet_cold_s`: `0.458148`
- `real_logdet_plan_cold_s`: `0.351403`
- `real_solve_action_plan_cold_s`: `0.254500`
- `real_apply_plan_warm_s`: `0.000142`
- `real_rapply_plan_warm_s`: `0.000081`
- `real_action_plan_warm_s`: `0.000134`
- `real_solve_action_plan_execute_s`: `0.000066`
- `real_multi_shift_plan_execute_s`: `0.000155`

### Sparse real

- `sparse_real_apply_s`: `0.127974`
- `sparse_real_apply_plan_s`: `0.106800`
- `sparse_real_rapply_plan_s`: `0.117296`
- `sparse_real_logdet_s`: `0.590590`
- `sparse_real_logdet_plan_s`: `0.377632`
- `sparse_real_solve_action_plan_s`: `0.261134`
- `sparse_real_inverse_action_plan_s`: `0.231177`
- `sparse_real_logdet_leja_hutchpp_s`: `1.990019`
- `sparse_real_logdet_leja_hutchpp_auto_s`: `1.952072`
- `sparse_real_inverse_diag_corrected_s`: `2.309461`

### Complex dense

- `complex_apply_cold_s`: `0.120995`
- `complex_apply_plan_cold_s`: `0.148162`
- `complex_rapply_plan_cold_s`: `0.224946`
- `complex_adjoint_apply_plan_cold_s`: `0.114168`
- `complex_action_plan_cold_s`: `0.547630`
- `complex_logdet_plan_cold_s`: `0.470936`
- `complex_solve_action_plan_cold_s`: `0.321451`
- `complex_apply_plan_warm_s`: `0.000090`
- `complex_rapply_plan_warm_s`: `0.000034`
- `complex_adjoint_apply_plan_warm_s`: `0.000024`
- `complex_multi_shift_plan_execute_s`: `0.000230`

### Sparse complex

- `sparse_complex_apply_s`: `0.169481`
- `sparse_complex_apply_plan_s`: `0.212636`
- `sparse_complex_rapply_plan_s`: `0.138673`
- `sparse_complex_adjoint_apply_plan_s`: `0.135303`
- `sparse_complex_action_plan_s`: `0.494469`
- `sparse_complex_logdet_plan_s`: `0.569058`
- `sparse_complex_solve_action_plan_s`: `0.343557`

## Main observations

- prepared operator plans remain the preferred repeated-call path for apply, transpose / adjoint apply, solve, inverse, and SLQ-style logdet / det runs under `jit`
- transpose / adjoint plan reuse belongs in the same practical guidance layer as forward apply; it should not be treated as an unowned side path
- fused bundle paths such as `*_logdet_solve_point_jit(...)` are now part of the benchmarked reuse story and should be preferred over separate solve-plus-logdet calls when both outputs are required
- compile / execute splits are required to interpret matrix-free backend-realized performance correctly
- dense real and dense complex workloads at the retained benchmark sizes remain CPU-favored in warm execution
- sparse complex operator-plan workloads are the first current matrix-free slice where GPU wins show up consistently
- startup prewarm and stable plan reuse reduce practical GPU pain, but they do not eliminate the dense real multi-shift compile cliff
- sparse Leja plus Hutch++ and corrected inverse diagonal remain the heaviest sparse-real estimator paths in this sweep

## AD And Recompile Notes

During this tranche, the benchmark exposed a JAX issue:

- callable-oriented `custom_vjp` kernels were valid for Python operator callables
- the same kernels were not valid for dynamic `OperatorPlan` payloads under `jit`

The fix was to split the plan JIT wrappers for:

- real `logdet`
- real `det`
- complex `logdet`
- complex `det`

onto plan-safe kernels that do not pass traced operator plans through `nondiff_argnums`.

That change removes the tracer failure from the benchmarked plan path and gives a stable public entrypoint for repeated plan reuse.
