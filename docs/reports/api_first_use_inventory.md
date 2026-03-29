Last updated: 2026-03-26T00:00:00Z

# API First Use Inventory

This report records the observed `arbplusjax.*` module set after representative first-use calls on top of `from arbplusjax import api`.

Budgets:
- `api` import budget: `<= 11`
- core point first-use budget: `<= 12`
- matrix dense first-use budget: `<= 13`
- matrix plan-prepare first-use budget: `<= 14`
- matrix plan-apply first-use budget: `<= 14`
- tail first-use budget: `<= 20`

## API Import

- observed module count: `12`

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

## Core Point First Use (`eval_point("exp", ...)`)

- observed module count: `13`
- `arbplusjax.point_wrappers_core` loaded: `True`

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
  "arbplusjax.point_wrappers_core",
  "arbplusjax.precision",
  "arbplusjax.public_metadata"
]
```

## Matrix Dense First Use (`eval_point("arb_mat_zero", ...)`)

- observed module count: `14`
- `arbplusjax.point_wrappers_matrix_dense` loaded: `True`
- `arbplusjax.mat_common` loaded: `True`

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
  "arbplusjax.mat_common",
  "arbplusjax.point_wrappers_matrix_dense",
  "arbplusjax.precision",
  "arbplusjax.public_metadata"
]
```

## Matrix Plan Prepare First Use (`eval_point("arb_mat_matvec_cached_prepare", ...)`)

- observed module count: `14`
- `arbplusjax.point_wrappers_matrix_plans` loaded: `True`
- `arbplusjax.mat_common` loaded: `True`

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
  "arbplusjax.mat_common",
  "arbplusjax.point_wrappers_matrix_plans",
  "arbplusjax.precision",
  "arbplusjax.public_metadata"
]
```

## Matrix Plan Apply First Use (`eval_point("arb_mat_matvec_cached_apply", ...)`)

- observed module count: `14`
- `arbplusjax.point_wrappers_matrix_plans` loaded: `True`
- `arbplusjax.mat_common` loaded: `True`

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
  "arbplusjax.mat_common",
  "arbplusjax.point_wrappers_matrix_plans",
  "arbplusjax.precision",
  "arbplusjax.public_metadata"
]
```

## Tail First Use (`tail_integral(...)`)

- observed module count: `21`
- `arbplusjax.special.tail_acceleration` loaded: `True`

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
  "arbplusjax.public_metadata",
  "arbplusjax.special",
  "arbplusjax.special.tail_acceleration",
  "arbplusjax.special.tail_acceleration.core",
  "arbplusjax.special.tail_acceleration.diagnostics",
  "arbplusjax.special.tail_acceleration.fallback_mp",
  "arbplusjax.special.tail_acceleration.quadrature",
  "arbplusjax.special.tail_acceleration.recurrence",
  "arbplusjax.special.tail_acceleration.regions",
  "arbplusjax.special.tail_acceleration.sequence"
]
```

## Notes

- The core point path should load `point_wrappers_core` and nothing matrix-specific.
- The matrix dense point path should load `point_wrappers_matrix_dense` plus `mat_common` and remain off plan-only helpers.
- The matrix plan paths should load `point_wrappers_matrix_plans` and stay off interval wrappers.
- The tail path is intentionally larger because it brings in the tail-acceleration subsystem only on demand.
