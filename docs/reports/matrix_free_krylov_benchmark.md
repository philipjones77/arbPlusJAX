Last updated: 2026-03-20T06:25:37Z

# Matrix-Free Krylov Benchmark Report

## Command

```bash
python benchmarks/benchmark_matrix_free_krylov.py
```

Runtime note from the run:

- NVIDIA GPU detected on machine, but CUDA-enabled `jaxlib` was not installed
- benchmark therefore ran on CPU

## Results

### Real dense

- `real_apply_s`: `0.163271`
- `real_apply_plan_s`: `0.161324`
- `real_action_s`: `0.728492`
- `real_action_plan_s`: `0.351882`
- `real_restarted_action_s`: `0.388415`
- `real_restarted_action_plan_s`: `0.409817`
- `real_grad_s`: `0.509196`
- `real_logdet_s`: `0.505909`
- `real_logdet_plan_s`: `0.412185`
- `real_det_s`: `0.390425`
- `real_det_plan_s`: `0.465138`
- `real_solve_action_s`: `0.290295`
- `real_solve_action_plan_s`: `0.325073`
- `real_inverse_action_s`: `0.388134`
- `real_inverse_action_plan_s`: `0.303748`
- `real_logdet_grad_s`: `0.608802`

### Sparse real

- `sparse_real_apply_s`: `0.294542`
- `sparse_real_apply_plan_s`: `0.157471`
- `sparse_real_logdet_s`: `0.637975`
- `sparse_real_logdet_plan_s`: `0.544391`
- `sparse_real_det_s`: `0.430223`
- `sparse_real_det_plan_s`: `0.394791`
- `sparse_real_solve_action_s`: `0.422631`
- `sparse_real_solve_action_plan_s`: `0.258872`
- `sparse_real_inverse_action_s`: `0.260272`
- `sparse_real_inverse_action_plan_s`: `0.249686`
- `sparse_real_logdet_leja_hutchpp_s`: `1.867329`
- `sparse_real_logdet_leja_hutchpp_auto_s`: `2.295081`
- `sparse_real_logdet_grad_s`: `0.676664`
- `sparse_real_inverse_diag_local_s`: `0.295543`
- `sparse_real_inverse_diag_corrected_s`: `1.992431`

### Complex dense

- `complex_apply_s`: `0.199005`
- `complex_apply_plan_s`: `0.186385`
- `complex_action_s`: `0.449733`
- `complex_action_plan_s`: `0.457754`
- `complex_restarted_action_s`: `0.404683`
- `complex_restarted_action_plan_s`: `0.446212`
- `complex_grad_s`: `0.553290`
- `complex_logdet_s`: `0.737944`
- `complex_logdet_plan_s`: `0.526566`
- `complex_det_s`: `0.485178`
- `complex_det_plan_s`: `0.466185`
- `complex_solve_action_s`: `0.483535`
- `complex_solve_action_plan_s`: `0.507092`
- `complex_inverse_action_s`: `0.524849`
- `complex_inverse_action_plan_s`: `0.478005`
- `complex_logdet_grad_s`: `1.103818`

### Sparse complex

- `sparse_complex_apply_s`: `0.132671`
- `sparse_complex_apply_plan_s`: `0.117930`
- `sparse_complex_action_s`: `0.481187`
- `sparse_complex_action_plan_s`: `0.586282`
- `sparse_complex_restarted_s`: `0.430856`
- `sparse_complex_restarted_plan_s`: `0.423077`
- `sparse_complex_logdet_s`: `0.594024`
- `sparse_complex_logdet_plan_s`: `0.528615`
- `sparse_complex_det_s`: `0.506190`
- `sparse_complex_det_plan_s`: `0.481444`
- `sparse_complex_solve_action_s`: `0.279245`
- `sparse_complex_solve_action_plan_s`: `0.301718`
- `sparse_complex_inverse_action_s`: `0.335327`
- `sparse_complex_inverse_action_plan_s`: `0.293488`

## Main observations

- prepared operator plans are now consistently usable for repeated `logdet` and `det` runs under `jit`
- the strongest reuse gains appear on sparse apply, sparse solve, and several logdet paths
- complex structured logdet remains materially more expensive than real symmetric logdet
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
