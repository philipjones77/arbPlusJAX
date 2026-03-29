Last updated: 2026-03-29T00:00:00Z

# Sparse Matrices

## Scope

This page owns the practical sparse-only calling guidance for the current sparse
real/complex matrix tranche.

The hardened sparse-native operational contract in this pass is:

- `point` sparse `matvec`
- `point` sparse `rmatvec`
- `point` sparse cached `prepare/apply`
- `basic` sparse `matvec`
- `basic` sparse `rmatvec`
- `basic` sparse cached `prepare/apply`
- compiled point/basic batch binders for cached sparse apply

These surfaces are explicitly tested to avoid dense fallback.

## Recommended Calling Pattern

Use sparse cached plans for repeated traffic:

```python
from arbplusjax import api, mat_wrappers

point_plan = mat_wrappers.srb_mat_matvec_cached_prepare_mode(sparse, impl="point")
basic_plan = mat_wrappers.srb_mat_matvec_cached_prepare_mode(sparse, impl="basic")

point_apply = api.bind_point_batch_jit(
    "srb_mat_matvec_cached_apply",
    dtype="float64",
    pad_to=8,
    backend="cpu",
)
basic_apply = api.bind_interval_batch_jit(
    "srb_mat_matvec_cached_apply",
    mode="basic",
    dtype="float64",
    pad_to=8,
    backend="cpu",
)

point_out = point_apply(point_plan, rhs_batch)
basic_out = basic_apply(basic_plan, rhs_batch)
```

Operational rules:

- prepare once, apply many times
- keep RHS batch shapes padded/stable
- use `point` for direct sparse numeric throughput
- use `basic` when interval/box enclosure is required on the same sparse-native
  apply path
- keep diagnostics outside the hot path by using diagnostics-bearing wrappers
  only at validation boundaries

## CPU / GPU Guidance

Current retained sparse operational benchmarking is on:

- [benchmark_sparse_operational_surface.py](/benchmarks/benchmark_sparse_operational_surface.py)
- [benchmark_sparse_operational_surface_cpu_refresh.json](/benchmarks/results/benchmark_sparse_operational_surface/benchmark_sparse_operational_surface_cpu_refresh.json)
- [benchmark_sparse_operational_surface_gpu_refresh.json](/benchmarks/results/benchmark_sparse_operational_surface/benchmark_sparse_operational_surface_gpu_refresh.json)

Practical interpretation:

- CPU remains the default for small sparse operational workloads.
- GPU should be considered for repeated, stable-shape sparse batches.
- Point/basic cached apply is the main sparse operational path to measure first.
- Do not infer sparse solve/factor behavior from these results; this page is about
  sparse-native operational apply surfaces only.

Retained `csr`, `n=32`, `float64` signal:

- real point cached apply:
  - CPU `0.00217s`
  - GPU `0.01180s`
- real point compiled cached batch apply:
  - CPU `0.00048s`
  - GPU `0.00087s`
- complex point compiled cached batch apply:
  - CPU `0.00055s`
  - GPU `0.00062s`

So the current sparse operational rule is:

- CPU first for ordinary sparse apply traffic
- GPU only after you have a genuinely repeated, stable-shape compiled batch path
- basic interval/box sparse apply is available and sparse-native, but it is still
  slower than point on both CPU and GPU in the retained slice

## Fast JAX vs Operational JAX

Sparse point/basic cached apply satisfies the structural fast-JAX contract when:

- the plan is prepared outside the hot loop
- the batch binder is reused
- shapes are stable
- no diagnostics formatting happens in the compiled path

Operationally, use:

- `bind_point_batch_jit(...)` for `point`
- `bind_interval_batch_jit(...)` for `basic`
- `pad_to` or shape bucketing for repeated service workloads

## No-Dense-Fallback Boundary

This practical page is intentionally narrow.

The sparse-native no-dense-fallback contract currently applies to the owned
operational surfaces listed above. It does not claim that every sparse `basic`
surface is sparse-native today.

In particular, some sparse `basic` solve/factor paths still lift to dense
interval/box implementations. Those are tracked separately in:

- [sparse_completion_plan.md](/docs/status/sparse_completion_plan.md)
- [sparse_matrix_status.md](/docs/reports/sparse_matrix_status.md)
