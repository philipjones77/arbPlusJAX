Last updated: 2026-03-17T00:00:00Z

# Engineering Standard

## Scope

This standard applies to canonical Arb-like functions, alternative implementations, and repo-defined mathematical families.

Code-local documentation quality for those surfaces is governed by:

- [code_documentation_standard.md](/docs/standards/code_documentation_standard.md)

## Engineering contract

- Public functions should expose the expected mode surface for their family (`point`, `basic`, and where appropriate `adaptive` / `rigorous`).
- Functions should obey the repo dtype rules. Family-specific algorithms do not get a separate dtype policy.
- Families should share batching/padding/dispatch infrastructure while keeping separate numerical kernels for `point`, `basic`, and tighter interval modes. Point paths should not be forced through interval/box kernels just to reuse plumbing.
- CPU is the current required execution and validation tranche in this repo state.
- Public surfaces, examples, tests, and benchmarks should still be written so they remain GPU-compatible unless a limitation is documented explicitly in metadata, diagnostics, or status reports.
- Validation ownership should be explicit about the current execution slice: CPU is required here, while GPU portability is a contract that should be preserved and surfaced through runtime parameterization rather than assumed.
- Runtime implementations should stay on the public JAX surface defined in [jax_surface_policy_standard.md](/docs/standards/jax_surface_policy_standard.md); SciPy-derived implementation paths are for benchmark/reference use only.
- Batch execution should stay shape-stable where possible, and padding-friendly where practical.
- Unnecessary Python-side value extraction and control flow should be removed from performance-sensitive paths.
- Automatic differentiation compatibility is a target, but current status must be reported honestly per implementation family.
- Tightening and hardening status should be tracked separately from provenance and naming.
- Production-facing families should expose metadata and diagnostics that make method selection, execution strategy, parameterization, and current hardening level inspectable from the public surface.
- Public hand-written surfaces should also satisfy the code-local documentation
  rules for module docstrings, public function docstrings, and non-obvious
  inline comments.

## Production calling contract

- Canonical example notebooks should teach the intended production calling style for their family.
- Repeated-call surfaces should prefer binder reuse, cached prepare/apply flows, or both where relevant.
- Variable-size service traffic should use stable dtype/mode controls and optional padding or chunking when that materially reduces recompiles.
- Cache-aware public surfaces should be tracked in generated reports rather than left implicit.
- Benchmarks should separate cold, warm, and recompile behavior when JAX compilation cost is part of the practical calling contract.
- Canonical tests, benchmarks, and examples should expose runtime selection through explicit `float32`/`float64` and CPU/GPU parameterization even when only one execution slice is exercised in the current environment.
- Cache-aware production behavior should follow [caching_recompilation_standard.md](/docs/standards/caching_recompilation_standard.md).

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
