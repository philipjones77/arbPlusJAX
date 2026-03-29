Last updated: 2026-03-29T00:00:00Z

# Fast JAX Standard

Status: active

## Purpose

This document defines what counts as a structurally `fast JAX` execution path
for a JAX-first numerical library.

It governs structural JAX readiness, not realized backend speed. Practical
CPU/GPU behavior after compile/startup, padding, bucketing, and batch
amortization is governed separately by:

- [backend_realized_performance_standard.md](/docs/standards/backend_realized_performance_standard.md)

Interpret this document in two layers:

- general JAX-library rule:
  define one structural acceptance standard for hot JAX kernels separate from
  realized backend speed
- arbPlusJAX specialization:
  the repo is currently applying that rule most aggressively to public `point`
  surfaces across the six top-level functionality categories

## Definition

A path qualifies as `fast JAX` only if all of the following are true:

- it is `jax.jit` compilable
- it is shape-stable for a compiled instance
- it is `vmap` compatible over its governed batch axis
- the hot path uses only JAX arrays and JAX primitives
- the hot path contains no Python-side adaptive loops
- the hot path contains no host callbacks
- the hot path touches no Arb, mpmath, or arbitrary-precision objects
- the hot path is written as array algebra rather than Python accumulation

If any one of those fails, the implementation is not yet `fast JAX`.

## Scope

This standard applies to hot execution surfaces across the repo's major
functionality categories:

- 1. core numeric scalars
- 2. interval / box / precision modes
- 3. dense matrix functionality
- 4. sparse / block-sparse / vblock functionality
- 5. matrix-free / operator functionality
- 6. special functions

It is relevant to:

- scalar kernels
- point-mode wrappers
- interval/basic repeated-call kernels where a true JAX path exists
- matrix and operator apply/solve helpers
- surrogate or approximant-backed evaluators

It does not redefine rigorous/adaptive/precision-reference semantics.

## Hot-Path Rules

### 1. No Python control flow in the kernel

Do not use Python `for` loops, Python accumulation, or Python `if/elif`
branches on array values in the fast kernel.

Use:

- `jax.lax.cond`
- `jax.lax.switch`
- `jax.lax.scan`
- vectorized array operations

### 2. Static shapes

Compiled fast kernels should use fixed array shapes per compiled instance.

Do not:

- change contour length or stencil size inside the JIT kernel
- allocate arrays whose size depends on runtime diagnostics
- pass Python lists of poles, residues, or work items through the hot path

Do:

- use fixed-shape work arrays
- choose a small family of predeclared sizes outside the kernel
- dispatch to those sizes before entering the compiled numeric path

### 3. Vectorized evaluation

Node-based, stencil-based, or batched operator evaluation should be vectorized.

Prefer:

- one integrand call over all nodes
- one reduction over weighted values
- `vmap` over external batches

Avoid:

- scalar loops over nodes
- per-item Python dispatch
- repeated host-side accumulation

### 4. Log-domain preference

Where products, Gamma-family factors, or Mellin-Barnes style terms appear, the
fast path should prefer log-domain algebra whenever practical.

This is both a numerical-stability rule and a fast-path design rule.

### 5. JAX-safe special functions only

Special functions used on the fast path must be JAX-safe.

Use one of:

- JAX-native special functions
- fixed-order recurrence kernels written in JAX
- approximant-backed JAX evaluators
- asymptotic-plus-correction kernels written in JAX

## Required Separation Of Layers

Engineering should separate three conceptual layers.

### Precise/reference layer

This layer may use:

- Arb
- mpmath
- adaptive logic
- variable precision

It is the truth engine, not the hot path.

### Fast-JAX layer

This layer should use:

- JAX arrays
- `float64` or `float32`
- fixed-shape kernels
- region-safe approximants or recurrences where needed

This is the production bulk-evaluation layer.

### Dispatch/diagnostics layer

This layer decides:

- whether the fast path is safe enough
- whether to fall back to the precise path
- what metadata or diagnostics to report

Do not mix these three responsibilities into one opaque function.

## Diagnostics Rule

Diagnostics that decide whether to use the fast path should live outside the
hot kernel.

Typical diagnostics:

- pole distance
- cancellation ratio
- tail indicator
- phase or oscillation indicator
- regime classifier
- convergence or residual summary for operator families

Those diagnostics may gate the fast path, but they must not reintroduce
Python-heavy adaptive behavior into the compiled kernel.

## Acceptance Criteria

A family or surface is `fast JAX ready` only when all of the following are
true:

- the public service surface exposes a compiled repeated-call callable when the
  family is intended for repeated API-level execution
- `jax.jit` works on the fast evaluator
- `jax.vmap` works on the fast evaluator
- no Python loops remain in the numerical hot path
- no Arb or mpmath objects are touched in the hot path
- safe-region values agree with the precise backend to the documented target
  tolerance
- runtime is dominated by compiled array operations rather than Python

## Relationship To Other Standards

- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)
  owns the broader JAX runtime contract
- [operational_jax_standard.md](/docs/standards/operational_jax_standard.md)
  owns stable-shape, minimal-load, and teaching-path policy for operational
  surfaces
- [engineering_standard.md](/docs/standards/engineering_standard.md)
  owns family hardening expectations

## arbPlusJAX Specialization

In this repo, the current first major rollout of this standard is the
point-fast JAX program, because `point` is the correct first target for turning
broad public function families into production-quality JAX kernels before
attempting full precision/adaptive acceleration.
