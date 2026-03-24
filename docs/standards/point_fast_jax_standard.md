Last updated: 2026-03-23T00:00:00Z

# Point Fast JAX Standard

Status: active

## Purpose

This document defines what counts as a `fast JAX` point-mode path in
arbPlusJAX.

It exists because `point` mode is the correct first target for turning broad
function families into production-quality JAX kernels before attempting full
precision/adaptive acceleration.

This standard does not require every public point function to already satisfy
the fast-path contract. It defines the target state and the acceptance rule.

## Definition

A function qualifies as a `fast JAX` point kernel only if all of the following
are true:

- it is `jax.jit` compilable
- it is shape-stable for a compiled instance
- it is `vmap` compatible over evaluation points
- the hot path uses only JAX arrays and JAX primitives
- the hot path contains no Python-side adaptive loops
- the hot path contains no host callbacks
- the hot path touches no Arb, mpmath, or arbitrary-precision objects
- the hot path is written as array algebra rather than Python accumulation

If any one of those fails, the implementation is not yet `fast JAX`.

## Scope

This standard applies first to public `point` mode surfaces across all six
top-level repo functionality categories defined in
[test_coverage_matrix.md](/docs/status/test_coverage_matrix.md).

The six required category scopes are:

- 1. core numeric scalars
- 2. interval / box / precision modes
- 3. dense matrix functionality
- 4. sparse / block-sparse / vblock functionality
- 5. matrix-free / operator functionality
- 6. special functions

It is especially relevant to:

- scalar special functions
- point-mode wrappers
- matrix and operator point helpers where a true JAX path exists
- surrogate or approximant-backed point evaluators

It does not redefine the rigorous/adaptive/precision-reference layers.

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

Compiled fast-point kernels should use fixed array shapes per compiled
instance.

Do not:

- change contour length or stencil size inside the JIT kernel
- allocate arrays whose size depends on runtime diagnostics
- pass Python lists of poles, residues, or work items through the hot path

Do:

- use fixed-shape work arrays
- choose a small family of predeclared sizes outside the kernel
- dispatch to those sizes before entering the compiled numeric path

### 3. Vectorized evaluation

Node-based or stencil-based point evaluation should be vectorized.

Prefer:

- one integrand call over all nodes
- one reduction over weighted values
- `vmap` over external point batches

Avoid:

- scalar loops over nodes
- per-node Python dispatch
- repeated host-side accumulation

### 4. Log-domain preference

Where products, Gamma-family factors, or Mellin-Barnes style terms appear, the
fast-point path should prefer log-domain algebra whenever practical.

This is both a numerical-stability rule and a fast-path design rule.

### 5. JAX-safe special functions only

Special functions used on the fast-point path must be JAX-safe.

If a special-function call is not array-safe and JIT-safe, it does not belong
in the fast kernel.

Use one of:

- JAX-native special functions
- fixed-order recurrence kernels written in JAX
- approximant-backed JAX evaluators
- asymptotic-plus-correction kernels written in JAX

## Required Separation Of Layers

Point-mode engineering should use three conceptual layers.

### Precise/reference layer

This layer may use:

- Arb
- mpmath
- adaptive logic
- variable precision

It is the truth engine, not the hot path.

### Point-fast layer

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

## Allowed Fast Strategies

Each point-mode family should prefer one primary fast strategy per regime:

- direct fp64 JAX formula
- fixed-iteration recurrence via `lax.scan` or `lax.fori_loop`
- piecewise Chebyshev or rational approximant
- asymptotic main term times fitted correction

Do not begin with large mixed strategy stacks inside one first-pass kernel.

## Diagnostics Rule

Diagnostics that decide whether to use the fast-point path should live outside
the hot kernel.

Typical diagnostics:

- pole distance
- cancellation ratio
- tail indicator
- phase or oscillation indicator
- regime classifier

Those diagnostics may gate the fast path, but they must not reintroduce
Python-heavy adaptive behavior into the compiled kernel.

## Acceptance Criteria

A point-mode family is `fast JAX ready` only when all of the following are
true:

- the public service surface exposes a compiled point-batch callable, normally
  through `api.bind_point_batch_jit(...)`, when that family is intended for
  repeated API-level point evaluation
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
- [engineering_standard.md](/docs/standards/engineering_standard.md)
  owns family hardening expectations
- this document specializes those two standards for the repo-wide `point fast`
  conversion tranche
