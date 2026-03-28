Last updated: 2026-03-28T00:00:00Z

# Backend-Realized Performance Usage

This note explains how to use the public API when you care about practical
CPU/GPU speed rather than only structural `fast JAX` compatibility.

Use this together with:

- [backend_realized_performance_standard.md](/docs/standards/backend_realized_performance_standard.md)
- [api_usability_standard.md](/docs/standards/api_usability_standard.md)
- [core_scalar_service_calling_standard.md](/docs/standards/core_scalar_service_calling_standard.md)

## Core Distinction

`Fast JAX` means a surface is structurally suitable for JAX:

- `jit` compatible
- `vmap` compatible
- shape-stable for a compiled instance
- no host callbacks in the hot path

That does not automatically mean it is the fastest way to run the workload on a
real backend.

Backend-realized performance depends on:

- compile/startup cost
- recompile churn
- device transfer/setup cost
- kernel launch overhead
- batch size and arithmetic intensity
- whether the call pattern actually reuses compiled work

## When GPU Is Usually Better

GPU is usually better when all of the following are true:

- the workload is repeated, not one-off
- the batch is large enough to amortize launch overhead
- the shape is stable across calls
- the compiled executable is reused
- the math dominates the orchestration around it

This is much more common for:

- dense matrix work
- sparse or matrix-free operator work
- heavy special-function batches
- repeated service-style batch evaluation

It is much less common for:

- tiny scalar calls
- highly variable tiny batches
- notebook cells that make many unrelated one-off calls

## What “Raw Or Lightly Wrapped Compiled Path” Means

The phrase means the useful numeric work reaches the compiled JAX kernel
quickly, with very little extra orchestration around it.

In practice, that means:

- one compiled kernel or a small compiled batch pipeline
- stable shapes
- no Python chunk loop around every tiny call
- minimal padding, copy, or setup overhead
- minimal binder or routing overhead
- no repeated plan construction
- no repeated compile-policy rediscovery on every call

Good examples in this repo are:

- direct repeated use of `api.bind_point_batch_jit(...)`
- direct repeated use of `api.bind_point_batch_jit_with_diagnostics(...)`
- repeated plan `prepare/apply` surfaces for matrix and operator families

Poor examples are:

- rebinding every call
- changing shapes constantly without padding or bucketing
- sending tiny scalar jobs to GPU and expecting wins
- wrapping every repeated call in fresh notebook-local orchestration

## Recommended Calling Order

For repeated workloads, use this order:

1. Choose the workload class.
   Decide whether the surface is tiny scalar, large scalar batch, dense matrix,
   sparse matrix, or matrix-free/operator.

2. Pick the backend intentionally.
   Use CPU for tiny scalar work unless the batch is large enough to justify GPU.

3. Pick a stable-shape policy.
   Prefer one of:
   - `pad_to`
   - `shape_bucket_multiple`
   - prepare/apply with stable apply shapes

4. Bind or prepare once.
   Reuse the bound callable or prepared plan.

5. Prewarm if the workload is service-like.
   Compile the representative stable-shape path before steady-state traffic.

6. Inspect diagnostics and retained benchmarks.
   Use the actual benchmark artifacts to see whether the chosen path is helping.

## Scalar API Pattern

For core scalar repeated calls, the recommended public API pattern is:

```python
from arbplusjax import api

policy = api.choose_point_batch_policy(
    batch_size=4096,
    dtype="float64",
    backend="auto",
    shape_bucket_multiple=128,
    min_gpu_batch_size=2048,
)

api.prewarm_core_point_kernels(
    dtype="float64",
    backend=policy.chosen_backend,
    batch_size=4096,
    shape_bucket_multiple=128,
    min_gpu_batch_size=2048,
)

bound = api.bind_point_batch_jit_with_diagnostics(
    "arb_fpwrap_double_exp",
    dtype="float64",
    shape_bucket_multiple=128,
    backend="auto",
    min_gpu_batch_size=2048,
)

values, diagnostics = bound(x)
```

This gives you:

- an explicit backend policy
- a stable-shape execution path
- binder reuse
- optional prewarm
- call diagnostics describing the chosen backend and effective padded shape

## Practical Scalar Guidance

For the current scalar tranche, the retained benchmark evidence says:

- CPU is still often better for tiny scalar warm calls
- GPU helps more for repeated larger batches
- stable padding can help GPU materially on some raw batch paths
- the service-style binder layer is still more mixed on GPU than the raw padded
  batch path

So for scalar functions:

- default to CPU for tiny interactive work
- use GPU only for repeated batch-heavy scalar workloads
- prefer padded or bucketed stable shapes
- prefer the compiled binder path over repeated one-off direct calls

## What To Inspect

When performance is not what you expect, inspect:

- chosen backend
- effective padded size
- whether the call compiled on first use
- warm versus cold timing
- padded versus unpadded timing
- binder/service path versus raw batch path

In the current scalar API, the diagnostics-bearing binders expose this
information directly.

## Common Mistakes

- assuming `jit` safety automatically means GPU speed
- rebinding inside a loop
- using GPU for tiny scalar batches
- changing batch sizes constantly without padding or bucketing
- mixing dtype policy across repeated calls
- measuring only warm time and ignoring compile cost

## Current Scope

This note is written first for the scalar/API tranche, but the same calling
pattern extends naturally to:

- interval/basic repeated batch surfaces
- dense cached matrix surfaces
- sparse prepared-plan surfaces
- matrix-free prepared-plan surfaces

As those tranches land, the same practical rule remains:

- expose the stable repeated-call mechanism in the API
- teach the intended use in notebooks and practical docs
- verify the backend behavior with retained benchmark artifacts

## Interval Mode Update

The interval/basic/adaptive/rigorous service tranche now follows the same
public repeated-call model:

- `api.choose_interval_batch_policy(...)`
- `api.bind_interval_batch(...)`
- `api.bind_interval_batch_with_diagnostics(...)`
- `api.bind_interval_batch_jit(...)`
- `api.bind_interval_batch_jit_with_diagnostics(...)`
- `api.prewarm_interval_mode_kernels(...)`

The current retained benchmark signal for this category is backend-sensitive:

- on CPU, the direct padded interval batch path can still beat the richer
  diagnostics-bearing interval service binder for tiny batches
- on GPU, the compiled interval service binder can beat the direct padded
  interval batch path once the workload is stable and repeated

So the practical rule for interval mode is:

- prefer the direct padded batch path on CPU when the workload is tiny and
  latency-sensitive
- prefer the compiled interval service binder on GPU for repeated stable-shape
  interval traffic
- keep `mode`, `prec_bits`, `dps`, `dtype`, and stable-shape policy fixed per
  worker
