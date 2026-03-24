Last updated: 2026-03-23T00:00:00Z

# Point Fast JAX Plan

Status: `in_progress`

This plan tracks the repo-wide conversion of public `point` mode into true
`fast JAX` across all six top-level functionality categories.

## Definition

`Fast JAX` means:

- `jax.jit` compilable
- `vmap` compatible
- shape-stable
- no Python control flow in the hot path
- no Arb or mpmath objects in the hot path
- no host callbacks

Current repo-wide public-surface status:

- all public point functions now have a compiled single-call surface through
  `api.eval_point(..., jit=True)`
- all public point functions now have a compiled public batch surface through
  `api.bind_point_batch_jit(...)`
- all public point functions now have a family-owned direct batch kernel
  registered in the point-batch API layer
- remaining work is deeper per-function numerical proof coverage and continued
  family-level hardening beyond the public API contract

The governing standard is:

- [point_fast_jax_standard.md](/docs/standards/point_fast_jax_standard.md)

The implementation program is:

- [point_fast_jax_implementation.md](/docs/implementation/point_fast_jax_implementation.md)

## Required Category Coverage

- `1. core numeric scalars`: `representative proof tranche`
- `2. interval / box / precision modes`: `representative proof tranche`
- `3. dense matrix functionality`: `representative proof tranche`
- `4. sparse / block-sparse / vblock functionality`: `representative proof tranche`
- `5. matrix-free / operator functionality`: `representative proof tranche`
- `6. special functions`: `representative proof tranche`

## Proof Rule

Each category must eventually have tests that prove:

- a compiled public point-batch or family-owned JIT surface exists
- the category point-fast path is `jit` compatible
- the category point-fast path is `vmap` compatible
- the category safe-box values agree with the precise path to the documented
  tolerance

This proof should land in category-owned pytest modules rather than one giant
cross-category smoke file.

## Immediate Next Step

Expand the representative six-category proof tranche into broader per-family
coverage and keep a living audit/registry that classifies existing point-mode
surfaces as:

- `direct_fast`
- `recurrence_fast`
- `approx_fast`
- `precise_only_for_now`
