Last updated: 2026-03-17T00:00:00Z

# Numerical Guidance

This page is for the practical question: what tends to work numerically in this codebase right now.

## Current guidance

- keep matvecs cheap when needed, but keep reductions and recurrence scalars in `float64` or `complex128`
- treat non-finite outputs as a tripwire for fallback, widened bounds, or higher precision
- separate correctness validation from performance benchmarking; both matter, but they answer different questions
- prefer explicit, opt-in use of soft surrogate operators rather than mixing them into the main Arb-like runtime surface

## JAX runtime practice

- pin `jax` and `jaxlib` deliberately when working on custom-VJP or matrix-free estimator paths
- rerun gradient and microbenchmark checks after JAX upgrades, especially for `jit(grad(...))` paths
- treat tracer/runtime and sharding changes as execution-path changes even when the public API looks stable
- this repo does not currently depend on `jax.shard_map`, so explicit-sharding `PartitionSpec` checks are not an immediate runtime dependency here
- matrix-free adjoint and SLQ-style workflows should keep regression coverage on custom-VJP, JIT, and probe-gradient behavior

## External ecosystem note

- `matfree` is a relevant design reference for differentiable Lanczos and Arnoldi workflows
- `traceax` is a relevant design reference for stochastic trace-estimation workflows
- they are not current runtime dependencies of this repository, but they are useful comparison points for practical AD and estimator design

## Detailed references

- [precision_guardrails_gpu.md](/docs/implementation/precision_guardrails_gpu.md)
- [matrix_logdet_landscape.md](/docs/implementation/matrix_logdet_landscape.md)
- [soft_ops_optional.md](/docs/implementation/soft_ops_optional.md)
