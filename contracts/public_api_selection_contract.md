Last updated: 2026-03-29T00:00:00Z

# Public API Selection Contract

## Scope

This contract covers the selection axes used by the public API metadata layer, capability registry, and routed evaluation surface in `arbplusjax.api`.

## Selection axes

The public API may expose the following explicit selection axes per function:

- `value_kind`: real, complex, interval, or matrix-family value layout selection
- `implementation`: canonical or alternative implementation selection by public name
- `implementation_version`: implementation-version selector when multiple maintained versions exist
- `method`: algorithmic method selection such as series, asymptotic, quadrature, recurrence, or transform
- `strategy`: execution strategy such as dense, cached, matvec, rmatvec, factorized, or operator-plan reuse
- `method_params`: method- or strategy-specific parameterization dictionary
- `mode`: point/basic/adaptive/rigorous dispatch mode where supported

For matrix-family APIs, the routed public surface should also keep the following
conceptually separate even when they are not all carried as one metadata field:

- matrix kind:
  - dense
  - sparse
  - block-sparse / variable-block sparse
  - matrix-free / operator
- structure subtype:
  - symmetric / Hermitian
  - SPD / HPD
  - triangular / banded / related structure flags
- execution route:
  - direct
  - cached / prepared
  - factorized / plan-backed
  - operator-plan / compiled batch route

## Selection semantics

- Explicit user selections must be honored exactly or rejected clearly.
- Unsupported combinations must raise rather than silently downgrade to a different implementation, method, or strategy.
- Unsupported matrix-kind / structure / execution-route combinations must also
  raise rather than silently crossing into a different family or route.
- Default selection may choose canonical implementations or default methods, but that default behavior must remain inspectable through metadata and registry surfaces.

## Registry contract

For each public function, the metadata and capability registry should report the currently advertised:

- `value_kinds`
- `implementation_options`
- `implementation_versions`
- `default_implementation`
- `method_tags`
- `default_method`
- `method_parameter_names`
- `execution_strategies`

When a public function belongs to a matrix family, the metadata layer should
remain sufficient for a caller to determine at least:

- which matrix family owns the surface
- whether a cached / prepared / compiled repeated-call route exists
- whether point/basic coverage is advertised
- whether diagnostics-bearing or fallback-aware routes exist

## JAX-first expectation

- The routed API should preserve JAX-friendly calling patterns.
- Repeated-execution strategies should favor shape stability, plan reuse, and batch-friendly entry points.
- Selection metadata is meant to help users choose AD-safe and low-recompile routes, not to hide runtime recompilation issues.
- Tests, benchmarks, and notebooks should execute against the source tree under `src/arbplusjax`, not rely on a separately installed package copy.
- Canonical notebooks and practical docs should teach the efficient repeated-call
  route for matrix families rather than leaving callers to infer it from the
  selection metadata alone.

## Source of truth

- `src/arbplusjax/api.py`
- `src/arbplusjax/public_metadata.py`
- `src/arbplusjax/capability_registry.py`
- `contracts/public_api_mode_contract.md`
