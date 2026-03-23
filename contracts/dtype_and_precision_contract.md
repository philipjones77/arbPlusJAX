Last updated: 2026-03-17T00:00:00Z

# Dtype And Precision Contract

## Scope

This contract covers the repo-wide dtype and precision policy implemented by `arbplusjax.precision` and used by the interval and box runtime.

## Default runtime dtype policy

- Package import enables JAX x64 through `arbplusjax.precision.enable_jax_x64()`.
- The repository default numeric target is `float64` for real work and `complex128` for complex work.
- Integer parameters are expected to use JAX integer dtypes, with `int64` as the default engineering target where supported by JAX.

## Canonical enclosure layouts

- real interval scalar: shape `(2,)` storing `[lo, hi]`
- complex box scalar: shape `(4,)` storing `[re_lo, re_hi, im_lo, im_hi]`
- real interval matrix/vector layers and complex box matrix/vector layers preserve the same trailing layout convention

## Precision semantics

- `prec_bits` is an enclosure-widening control, not an arbitrary-precision arithmetic switch.
- `dps` and `prec_bits` helpers in `arbplusjax.precision` map user intent onto widening policy.
- Lower `prec_bits` may yield wider enclosures.
- Higher `prec_bits` may yield tighter enclosures.
- Runtime kernels still execute in standard JAX dtype precision rather than true Arb-style arbitrary precision.

## Centralization contract

- Repo code must use `arbplusjax.precision` for x64 enablement and precision-context helpers.
- Repo code should not scatter direct `jax.config.update("jax_enable_x64", ...)` calls across many modules.
- `precision.jax_x64_context(...)`, `precision.workdps(...)`, and `precision.workprec(...)` are the stable repo-owned context helpers.

## Error-handling contract

- Invalid or non-finite interval computations may widen to full intervals or full complex boxes instead of pretending to produce a valid enclosure.
- Precision controls must not silently claim arbitrary-precision semantics that the runtime does not implement.

## Source of truth

- `src/arbplusjax/__init__.py`
- `src/arbplusjax/precision.py`
- `docs/standards/precision_standard.md`
- `tests/conftest.py`
