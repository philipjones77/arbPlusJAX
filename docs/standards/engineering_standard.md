Last updated: 2026-03-17T00:00:00Z

# Engineering Standard

## Scope

This standard applies to canonical Arb-like functions, alternative implementations, and repo-defined mathematical families.

## Engineering contract

- Public functions should expose the expected mode surface for their family (`point`, `basic`, and where appropriate `adaptive` / `rigorous`).
- Functions should obey the repo dtype rules. Family-specific algorithms do not get a separate dtype policy.
- Families should share batching/padding/dispatch infrastructure while keeping separate numerical kernels for `point`, `basic`, and tighter interval modes. Point paths should not be forced through interval/box kernels just to reuse plumbing.
- Runtime implementations should stay on the public JAX surface defined in [jax_surface_policy_standard.md](/home/phili/projects/arbplusJAX/docs/standards/jax_surface_policy_standard.md); SciPy-derived implementation paths are for benchmark/reference use only.
- Batch execution should stay shape-stable where possible, and padding-friendly where practical.
- Unnecessary Python-side value extraction and control flow should be removed from performance-sensitive paths.
- Automatic differentiation compatibility is a target, but current status must be reported honestly per implementation family.
- Tightening and hardening status should be tracked separately from provenance and naming.

## Status interpretation

- `pure_jax` is an aspiration-oriented field, not a binary admission rule.
- `dtype` reports current conformance to repo dtype expectations.
- `kernel_split` reports whether a family uses shared dispatch with separate per-mode kernels, or still mixes point and interval implementation layers.
- `batch` reports current batch stability, not an idealized future state.
- `ad` reports current compatibility expectations, not theoretical differentiability.
- `hardening` reports the current numerical-hardening level of the implementation path.

## Methodology

- Engineering status is generated from `arbplusjax.function_provenance` using the same inventory/provenance source as the naming reports.
- Machine-readable downstream routing data is generated from `arbplusjax.capability_registry` and written to `docs/reports/function_capability_registry.json`.
- New public families inherit a conservative default status from their category and module lineage.
- Known families with stronger or weaker guarantees use explicit overrides in `arbplusjax.function_provenance`.
- Regenerate with `python tools/check_generated_reports.py`.
- If a function family changes materially, update the engineering-status override logic before committing regenerated reports.
