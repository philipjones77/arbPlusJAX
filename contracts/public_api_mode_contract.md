Last updated: 2026-03-17T00:00:00Z

# Public API And Mode Contract

## Scope

This contract covers the shared public dispatch surface in `arbplusjax.api` and downstream wrappers that intentionally reuse the same mode vocabulary, including `arbplusjax.stable_kernels`.

## Standard mode set

The repository standard mode names are:

- `point`
- `basic`
- `adaptive`
- `rigorous`

These names are the stable public vocabulary for mode-routed scalar and interval-aware kernels.

## Mode semantics

- `point`: point evaluation only, with no enclosure semantics
- `basic`: midpoint evaluation plus outward boxing
- `adaptive`: fixed-shape heuristic enclosure tightening around midpoint evaluation
- `rigorous`: analytic or Jacobian/Lipschitz-style enclosure path when implemented

The semantic meaning of the modes is stable. The exact internal algorithm used for a mode may change if the public semantics remain the same.

## Dispatch contract

- `api.eval_point(...)` and `api.eval_point_batch(...)` are the canonical point-dispatch entry points.
- `api.eval_interval(...)` and `api.eval_interval_batch(...)` are the canonical interval-dispatch entry points.
- Downstream wrappers that expose `mode=...` must use the standard mode vocabulary above.
- Invalid mode names must be rejected rather than silently reinterpreted.

## Per-function support contract

- Not every public function is required to support every mode.
- Per-function point support, interval support, and advertised interval modes are reported by `api.list_public_function_metadata()` and `public_metadata.build_public_metadata_registry(...)`.
- The public metadata layer is the source of truth for whether a function is point-only, interval-capable, or experimental.

## Batch contract

- Batch helpers preserve the same mode vocabulary as their scalar counterparts.
- Fixed and padded batch helpers may use different execution paths internally, but they must preserve the same public mode meaning.

## Non-goals

- This contract does not freeze every implementation path in `src/arbplusjax/`.
- This contract does not claim that every public function has identical numerical maturity across all four modes.

## Source of truth

- `src/arbplusjax/api.py`
- `src/arbplusjax/public_metadata.py`
- `docs/standards/precision_standard.md`
- `tests/test_stable_kernels.py`
