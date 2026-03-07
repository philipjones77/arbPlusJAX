Last updated: 2026-03-07T00:00:00Z

# Engineering Policy

## Scope

This policy applies to canonical Arb-like functions, alternative implementations, and repo-defined mathematical families.

## Engineering contract

- Public functions should expose the expected mode surface for their family (`point`, `basic`, and where appropriate `adaptive` / `rigorous`).
- Functions should obey the repo dtype rules. Family-specific algorithms do not get a separate dtype policy.
- Batch execution should stay shape-stable where possible, and padding-friendly where practical.
- Unnecessary Python-side value extraction and control flow should be removed from performance-sensitive paths.
- Automatic differentiation compatibility is a target, but current status must be reported honestly per implementation family.
- Tightening and hardening status should be tracked separately from provenance and naming.

## Status interpretation

- `pure_jax` is an aspiration-oriented field, not a binary admission rule.
- `dtype` reports current conformance to repo dtype expectations.
- `batch` reports current batch stability, not an idealized future state.
- `ad` reports current compatibility expectations, not theoretical differentiability.
- `hardening` reports the current numerical-hardening level of the implementation path.

## Methodology

- Engineering status is generated from `arbplusjax.function_provenance` using the same inventory/provenance source as the naming reports.
- New public families inherit a conservative default status from their category and module lineage.
- Known families with stronger or weaker guarantees use explicit overrides in `arbplusjax.function_provenance`.
- Regenerate with `python tools/check_generated_reports.py`.
- If a function family changes materially, update the engineering-status override logic before committing regenerated reports.
