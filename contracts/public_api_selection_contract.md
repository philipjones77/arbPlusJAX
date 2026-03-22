Last updated: 2026-03-20T00:00:00Z

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

## Selection semantics

- Explicit user selections must be honored exactly or rejected clearly.
- Unsupported combinations must raise rather than silently downgrade to a different implementation, method, or strategy.
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

## JAX-first expectation

- The routed API should preserve JAX-friendly calling patterns.
- Repeated-execution strategies should favor shape stability, plan reuse, and batch-friendly entry points.
- Selection metadata is meant to help users choose AD-safe and low-recompile routes, not to hide runtime recompilation issues.
- Tests, benchmarks, and notebooks should execute against the source tree under `src/arbplusjax`, not rely on a separately installed package copy.

## Source of truth

- `src/arbplusjax/api.py`
- `src/arbplusjax/public_metadata.py`
- `src/arbplusjax/capability_registry.py`
- `contracts/public_api_mode_contract.md`
