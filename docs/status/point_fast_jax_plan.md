Last updated: 2026-03-27T00:00:00Z

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
- all public point functions now have a direct registered point-batch route in
  the point-batch API layer; the public point-fast coverage gap is `0`
- a family-level point/basic verification ledger now exists in
  [point_basic_surface_status.md](/docs/reports/point_basic_surface_status.md)
- an explicit parameterized public AD audit now exists in
  [parameterized_ad_verification.md](/docs/reports/parameterized_ad_verification.md)
  and complements the family-level ledger with checked runtime proof cases
- remaining work is deeper numerical/performance proof coverage, continued
  family-level hardening beyond the public API contract, broader public
  `basic` exposure outside the currently enclosure-oriented families, and
  machine-readable capability classification of the landed point-fast set

The governing standard is:

- [fast_jax_standard.md](/docs/standards/fast_jax_standard.md)

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

Use the living audit layer in
[point_basic_surface_status.md](/docs/reports/point_basic_surface_status.md),
the per-function ledger in
[point_basic_function_verification.md](/docs/reports/point_basic_function_verification.md),
and the parameterized public AD audit in
[parameterized_ad_verification.md](/docs/reports/parameterized_ad_verification.md)
to keep widening proof quality beyond public-surface availability, and keep the
registry classifying existing point-mode surfaces as:

- `direct_fast`
- `recurrence_fast`
- `approx_fast`
- `precise_only_for_now`
