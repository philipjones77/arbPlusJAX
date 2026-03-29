Last updated: 2026-03-26T00:00:00Z

# API Cold Path Inventory

This report records the observed module set for package import and `api` import after the current lazy-loading refactor.

Budgets:
- package import budget: `<= 1` `arbplusjax.*` modules
- `api` import budget: `<= 11` `arbplusjax.*` modules

## Package Import

- observed module count: `1`

```json
[
  "arbplusjax.precision"
]
```

## API Import

- observed module count: `12`
- `public_metadata` loaded: `True`
- `point_wrappers_core` loaded: `False`

```json
[
  "arbplusjax.acb_core",
  "arbplusjax.api",
  "arbplusjax.arb_core",
  "arbplusjax.checks",
  "arbplusjax.double_interval",
  "arbplusjax.elementary",
  "arbplusjax.jax_precision",
  "arbplusjax.kernel_helpers",
  "arbplusjax.lazy_imports",
  "arbplusjax.lazy_jit",
  "arbplusjax.precision",
  "arbplusjax.public_metadata"
]
```

## Notes

- `point_wrappers_core` staying absent from this report means the point-family split is still holding.
- `public_metadata` staying absent from this report means metadata access is no longer part of the `api` cold path.
- Remaining cold-path bulk is currently dominated by the tail-acceleration runtime surface.
