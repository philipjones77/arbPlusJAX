Last updated: 2026-03-28T00:00:00Z

# API Surface Kinds Standard

Status: active

## Purpose

This document defines the canonical public API surface kinds used in this repo.

Several standards refer to direct calls, compiled binders, diagnostics-bearing
variants, prepared plans, and policy helpers. This document exists so those
categories are defined once, in one place, rather than being redefined
partially across multiple standards.

It owns the taxonomy.

The taxonomy should remain reusable across JAX-first numerical libraries even
when their concrete exported names differ. arbPlusJAX then specializes that
taxonomy through its own public API.

Neighboring standards then own:

- runtime semantics and dtype/AD/diagnostics rules:
  [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)
- structural fast-JAX requirements for point mode:
  [fast_jax_standard.md](/docs/standards/fast_jax_standard.md)
- backend-realized performance policy:
  [backend_realized_performance_standard.md](/docs/standards/backend_realized_performance_standard.md)
- practical teaching and canonical use:
  [api_usability_standard.md](/docs/standards/api_usability_standard.md)

## Core Rule

Public API kinds must be explicit and composable.

Generic JAX-library rule:

- separate mathematical mode, execution style, observability, and policy
  surfaces

arbPlusJAX specialization:

- use the named categories in this document to describe the concrete public
  surfaces exported through `arbplusjax.api` and the matrix/helper entry
  modules

The repo should not blur:

- mode selection such as `point` versus `basic`
- execution kind such as direct versus compiled versus prepared
- observability kind such as plain return versus diagnostics-bearing
- policy kind such as explicit backend-policy helpers
- matrix kind such as dense versus sparse versus block-sparse versus
  matrix-free/operator
- structure subtype such as symmetric versus Hermitian versus SPD/HPD

These are different axes and should remain different axes.

## Matrix Family Interpretation Rule

For matrix libraries, the public API should be legible across three separate
questions:

1. matrix kind
   - dense
   - sparse
   - block-sparse / vblock
   - matrix-free / operator
2. structure subtype
   - symmetric / Hermitian
   - SPD / HPD
   - triangular / banded / permutation / similar
3. surface kind
   - direct
   - bound service
   - compiled bound
   - diagnostics-bearing
   - prepared / plan
   - policy helper

The API should not force users to infer all three from one overloaded name or
one giant auto-routing layer.

## Canonical Public Surface Kinds

### 1. Direct surface

The direct surface is the minimal ordinary public call.

Examples:

- `api.eval_point(...)`
- `api.eval_interval(...)`
- `jrb_mat_logdet_slq_point(...)`

Purpose:

- correctness-oriented ordinary usage
- explicit public semantics without requiring preparation

This is the default mathematical API, not necessarily the fastest repeated-call
surface.

### 2. Light wrapper surface

The light wrapper surface adds only thin public orchestration around the direct
surface.

Typical examples:

- dtype normalization
- stable routing between a small number of explicit public variants
- lightweight output normalization

A light wrapper must not add heavy repeated-call orchestration, hidden planning,
or expensive diagnostics by default.

### 3. Bound service surface

The bound service surface fixes repeated-call policy once and reuses it.

Examples:

- `api.bind_point_batch(...)`
- `api.bind_interval_batch(...)`

Typical responsibilities:

- bind dtype policy
- bind shape policy such as padding or buckets
- bind chunking policy when relevant
- reuse the same public operation selection across repeated calls

This is the canonical repeated-call service surface when prepare/apply is not
the right abstraction.

### 4. Compiled bound surface

The compiled bound surface is the repeated-call binder that owns the compiled
entrypoint explicitly.

Examples:

- `api.bind_point_batch_jit(...)`

Typical responsibilities:

- stable compiled entrypoint
- static routing policy
- compile reuse across repeated calls

This is distinct from the direct surface and from the general bound service
surface.

### 5. Diagnostics-bearing surface

The diagnostics-bearing surface returns structured metadata together with the
main value.

Examples:

- `*_with_diagnostics(...)`
- `api.bind_point_batch_with_diagnostics(...)`
- `api.bind_point_batch_jit_with_diagnostics(...)`

Responsibilities:

- expose execution strategy and numerical metadata
- expose compile/reuse metadata where the API supports it
- remain opt-in rather than making every normal call pay the same overhead

### 6. Prepared or plan surface

The prepared surface separates structure preparation from repeated application.

Examples:

- `*_plan_prepare(...)`
- `*_cached_prepare(...)`
- `*_cached_apply(...)`
- matrix/operator prepare/apply APIs

Responsibilities:

- make structural reuse explicit
- separate prepare cost from repeated apply cost
- give repeated-call workloads a stable reuse boundary

### 7. Policy helper surface

The policy helper surface recommends how to use another public surface.

Examples:

- `choose_point_batch_policy(...)`
- family-level warmup helpers such as `prewarm_core_point_kernels(...)`

Responsibilities:

- select or recommend backend, padding, bucketing, chunking, or warmup policy
- remain explicit recommendations, not hidden side effects

### 8. Auto-routing surface

The auto-routing surface chooses among explicit public variants.

Examples:

- `api.evaluate(...)`
- high-level mode/family routing where documented

Auto-routing is allowed, but it must not erase the explicit variants beneath
it. It is a discoverability and convenience layer, not the only API.

## Orthogonal Mode Axis

The following are mode kinds, not API surface kinds:

- `point`
- `basic`
- `adaptive`
- `rigorous`

Those modes may appear on several surface kinds above.

For example:

- direct point surface
- compiled bound point surface
- direct basic surface
- diagnostics-bearing basic surface

Do not confuse mode taxonomy with surface taxonomy.

## Naming Rule

Preferred naming signals:

- direct: ordinary public name
- compiled binder: `bind_*_jit`
- diagnostics-bearing: `*_with_diagnostics`
- prepared: `*_prepare`, `*_cached_prepare`, `*_plan_prepare`
- repeated apply: `*_apply`, `*_cached_apply`
- policy helpers: `choose_*_policy`, `prewarm_*`

If a family needs a new public kind, it should either follow this naming model
or document why the variant is materially different.

## Ownership Rule

Standards and docs should reference these terms consistently:

- `direct`
- `light wrapper`
- `bound service`
- `compiled bound`
- `diagnostics-bearing`
- `prepared/plan`
- `policy helper`
- `auto-routing`

Other standards may impose rules on these kinds, but they should not redefine
the taxonomy independently.
