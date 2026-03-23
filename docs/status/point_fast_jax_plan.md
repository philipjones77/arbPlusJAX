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

The governing standard is:

- [point_fast_jax_standard.md](/docs/standards/point_fast_jax_standard.md)

The implementation program is:

- [point_fast_jax_implementation.md](/docs/implementation/point_fast_jax_implementation.md)

## Required Category Coverage

- `1. core numeric scalars`: `planned`
- `2. interval / box / precision modes`: `planned`
- `3. dense matrix functionality`: `planned`
- `4. sparse / block-sparse / vblock functionality`: `planned`
- `5. matrix-free / operator functionality`: `planned`
- `6. special functions`: `planned`

## Proof Rule

Each category must eventually have tests that prove:

- the category point-fast path is `jit` compatible
- the category point-fast path is `vmap` compatible
- the category safe-box values agree with the precise path to the documented
  tolerance

This proof should land in category-owned pytest modules rather than one giant
cross-category smoke file.

## Immediate Next Step

Create the per-category audit/registry that classifies existing point-mode
surfaces as:

- `direct_fast`
- `recurrence_fast`
- `approx_fast`
- `precise_only_for_now`
