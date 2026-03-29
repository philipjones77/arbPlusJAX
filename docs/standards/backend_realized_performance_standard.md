Last updated: 2026-03-28T00:00:00Z

# Backend Realized Performance Standard

Status: active

## Purpose

This document defines the second-layer performance standard that applies after a
surface is already structurally `fast JAX`.

`Fast JAX` is necessary but not sufficient for good runtime behavior on a real
backend. A surface can be:

- `jit` compilable
- `vmap` compatible
- shape-stable for a compiled instance

and still perform poorly on CPU or GPU because of compile/startup cost, shape
churn, tiny-kernel overhead, transfer/setup cost, or weak batching policy.

This standard exists to govern that practical layer explicitly.

It should remain reusable across JAX-first numerical libraries. arbPlusJAX then
specializes it through its current binder controls, prepared-plan surfaces,
benchmark artifacts, and notebook guidance.

It is a companion to:

- [api_surface_kinds_standard.md](/docs/standards/api_surface_kinds_standard.md)
- [fast_jax_standard.md](/docs/standards/fast_jax_standard.md)
- [startup_compile_standard.md](/docs/standards/startup_compile_standard.md)
- [caching_recompilation_standard.md](/docs/standards/caching_recompilation_standard.md)
- [benchmark_validation_policy_standard.md](/docs/standards/benchmark_validation_policy_standard.md)

## Core Distinction

The repo uses two different concepts:

1. `fast JAX`
   A structural kernel/runtime contract:
   - JIT-safe
   - VMAP-safe
   - shape-stable for a compiled instance
   - no host objects/callbacks in the hot path

2. `backend-realized performance`
   A practical backend-runtime contract:
   - compile/startup cost is governed
   - repeated calls avoid unnecessary recompiles
   - shape variation is bucketed or padded intentionally
   - batching is large enough to amortize backend overhead
   - the backend is actually a win, or the repo documents when it is not

The repo must not silently collapse these into one claim.

Generic JAX-library rule:

- structural JAX readiness and realized backend speed are different standards

arbPlusJAX specialization:

- the current public binder controls, diagnostics fields, and prepared-plan
  surfaces are the repo's concrete way of satisfying that distinction

The surface-kind names referenced in this document, such as bound service,
compiled bound, diagnostics-bearing, prepared, raw, or lightly wrapped paths,
are owned by:

- [api_surface_kinds_standard.md](/docs/standards/api_surface_kinds_standard.md)

## Scope

Apply this standard to:

- public service-style batch binders
- compiled public point-batch surfaces
- interval/basic batch surfaces when they claim repeated execution value
- dense, sparse, and matrix-free prepared-plan apply surfaces
- benchmark harnesses and executed notebooks that teach production-style use
- CPU and GPU rollout work

## Required Policy

### 1. Separate structural readiness from realized speed

A surface may be marked `fast JAX` without being marked backend-efficient.

Backend-efficient claims require direct evidence on the target backend.

### 2. Measure cold, warm, and recompile separately

Every practical backend-performance claim must distinguish:

- cold compile + first execution
- warm repeated execution
- recompile or invalidation cost after a shape/static-control change

Do not use one blended metric.

### 3. Stable-shape paths are required for repeated execution

If a surface is intended for repeated service-style use, it must expose one or
more stable-shape strategies such as:

- explicit `pad_to`
- shape buckets
- chunking over stable bucket sizes
- prepare/apply split with stable apply shapes

### 4. Backend choice must be workload-aware

The repo must not assume GPU is always faster.

For tiny kernels or low arithmetic intensity, CPU may remain the preferred
backend even when a GPU path exists.

The repo should explicitly document:

- when GPU is expected to help
- when CPU remains preferable
- which batching/padding thresholds matter

### 5. Compile ownership and reuse must be explicit

Repeated batch execution should prefer:

- binder reuse
- cache reuse
- compiled callable reuse across nearby shapes via buckets or fixed padding

The public API should not force users to rediscover compile-stable calling
patterns through trial and error.

### 6. Optimization must target the actual bottleneck class

Backend-performance work should classify regressions into one of:

- compile/startup dominated
- shape-churn / recompilation dominated
- transfer/setup dominated
- tiny-kernel launch dominated
- true device-compute dominated

Fixes should match the bottleneck class rather than assuming all slowdowns are
numerical-kernel problems.

## Required Public API Surface

When a family claims repeated execution value on CPU or GPU, the public API
should expose the backend-realized performance controls explicitly rather than
forcing callers to rediscover them ad hoc.

The preferred public surface is:

### 1. Backend-aware binding options

Repeated-execution binders should prefer an explicit policy surface such as:

- `backend="cpu" | "gpu" | "auto"`
- `shape_bucket_multiple=...`
- `pad_to=...`
- `chunk_size=...`
- `prewarm=True` or an equivalent warmup hook
- `min_gpu_batch_size=...` when a family has a backend crossover rule

The repo may stage this in tranches, but the intended public ownership should
be clear in the API rather than being left entirely to notebook-local helper
code.

### 2. Prepared service handles

If a surface is used repeatedly with stable semantics, the repo should prefer a
prepare/apply or bind/reuse pattern over repeated one-shot orchestration.

Examples include:

- prepared point-batch handles
- prepared matrix or operator plans
- cached special-function batch handles
- retained benchmark/example binders that mirror production usage

### 3. Performance diagnostics on binders

The API should make backend-realized execution observable.

Preferred diagnostics include:

- chosen backend
- effective padded or bucketed size
- chunking policy
- whether a compile or cache miss occurred
- the relevant reuse boundary: binder reuse, plan reuse, or both

These diagnostics may be optional metadata or a companion diagnostics call, but
they should not remain opaque.

### 4. Auto policy helpers

The repo should provide reusable helpers that recommend a stable repeated-call
policy from workload facts such as:

- batch size
- dtype
- backend
- shape variability
- arithmetic intensity

Typical examples:

- `choose_point_batch_policy(...)`
- `choose_matrix_batch_policy(...)`

These helpers are recommendations, not silent hidden heuristics.

### 5. Family-level warmup entrypoints

Hot repeated-execution families should expose an intentional warmup surface for
services, notebooks, and benchmark setup.

Typical examples:

- `prewarm_core_point_kernels(...)`
- `prewarm_special_function_point_kernels(...)`
- matrix or sparse family warmup helpers

Warmup should target the governed stable-shape path rather than compiling an
arbitrary one-off workload.

### 6. Explicit backend-realized benchmark surfaces

Benchmark helpers and retained artifacts should mirror the public repeated-call
API and record the main backend-realized distinctions:

- cold versus warm
- recompile-sensitive changes
- padded versus unpadded
- bucketed versus exact-size
- prepared versus raw
- CPU versus GPU

These benchmark surfaces are the evidence layer for the policy above, not an
unrelated measurement sidecar.

## Required Evidence

For a backend-realized performance claim, the repo should provide:

- at least one benchmark artifact with cold/warm/recompile separation
- at least one repeated-call surface using an explicit stable-shape policy
- at least one notebook/example that teaches the stable-shape pattern
- test coverage proving the stable-shape/bucketing path is semantically correct

## Anti-patterns

- calling a surface “fast” only because it is JIT-safe
- teaching GPU execution on tiny scalar kernels without noting amortization
- measuring only warm time while ignoring compile/recompile cost
- letting repeated user-sized shape churn define the default service path
- adding padding manually in notebooks while the public binder lacks a governed
  stable-shape option

## Current Repo Mapping

This standard currently maps most directly to:

- public point-batch service binders in [api.py](/src/arbplusjax/api.py)
- startup/compile governance in
  [startup_compile_standard.md](/docs/standards/startup_compile_standard.md)
- cache/recompile governance in
  [caching_recompilation_standard.md](/docs/standards/caching_recompilation_standard.md)
- scalar and matrix service benchmarks under [benchmarks](/benchmarks)

In the current GPU rollout, the first scalar tranche showed:

- tiny scalar kernels can remain slower on GPU in steady state
- compile/startup cost is still large even for small probes
- stable-shape batch policy materially affects warm GPU behavior

That is exactly the class of issue this standard governs.
