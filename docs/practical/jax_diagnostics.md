Last updated: 2026-03-20T00:00:00Z

# JAX Diagnostics

This repo now has an optional JAX diagnostics path in [jax_diagnostics.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jax_diagnostics.py).

## Design Goal

Diagnostics must be off by default.

That means:
- no tracing overhead unless explicitly enabled
- no extra compilation artifacts unless explicitly requested
- no benchmark slowdowns in normal test or user execution

## Environment Controls

Prefix: `ARBPLUSJAX_JAX_DIAGNOSTICS_`

Flags:
- `ENABLED=1`
- `JAXPR=1`
- `HLO=1`
- `TRACE=1`
- `TRACE_DIR=/path/to/output`

## What It Captures

- compile latency
- steady-state execution latency
- recompile latency for a new shape
- process peak RSS delta
- device memory delta when JAX reports memory stats
- optional JAXPR capture
- optional HLO capture
- optional trace emission through `jax.profiler.trace`

## Main APIs

- `config_from_env()`
- `collect_compilation_artifacts(...)`
- `profile_jitted_function(...)`
- `profile_function_suite(...)`
- `write_profile_report(...)`

## Recommended Workflow

1. Run correctness tests first.
2. Run the normal benchmark for the subsystem:
   - dense: [benchmark_dense_matrix_surface.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_dense_matrix_surface.py)
   - sparse: [benchmark_sparse_matrix_surface.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_sparse_matrix_surface.py)
   - matrix-free: [benchmark_matrix_free_krylov.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_matrix_free_krylov.py)
3. Run the diagnostics benchmark when you want compile/recompile or memory visibility:
   - [benchmark_matrix_stack_diagnostics.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_matrix_stack_diagnostics.py)

Example:

```bash
ARBPLUSJAX_JAX_DIAGNOSTICS_ENABLED=1 \
ARBPLUSJAX_JAX_DIAGNOSTICS_JAXPR=1 \
python benchmarks/benchmark_matrix_stack_diagnostics.py --n 8 --repeats 4
```

Default report path:
- `experiments/benchmarks/outputs/diagnostics/matrix_stack_profile.json`

## Interpretation

- high `compile_ms` with low steady-state time means compile amortization matters
- high `recompile_new_shape_ms` means shape instability is expensive
- high RSS or device-memory deltas suggest plan payload size or captured constants are too large
- if plan-based entrypoints do not improve `steady_ms_median`, the workload is dominated by kernel math rather than Python-side wrapping

## Coverage

Diagnostics coverage is validated in:
- [test_jax_diagnostics.py](/home/phili/projects/arbplusJAX/tests/test_jax_diagnostics.py)
