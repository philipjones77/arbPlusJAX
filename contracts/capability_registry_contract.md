Last updated: 2026-03-17T00:00:00Z

# Capability Registry Contract

## Scope

This contract covers the machine-readable capability registry exported by `arbplusjax.capability_registry` and rendered to JSON for downstream routing.

## Runtime entry points

The stable runtime entry points are:

- `build_capability_registry()`
- `render_capability_registry_json()`
- `lookup_capability(name)`

## Top-level registry shape

The registry object must contain:

- `generated_at`
- `source`
- `policy_refs`
- `downstream_kernels`
- `functions`

## Function entry contract

Each row in `functions` must describe a public function with the current metadata and engineering surface, including:

- identity fields such as `name`, `qualified_name`, `module`, `implementation_name`, and `family`
- stability and support fields such as `stability`, `point_support`, `interval_support`, and `interval_modes`
- selection fields such as `value_kinds`, `implementation_options`, `implementation_versions`, `default_implementation`, `default_method`, `method_parameter_names`, and `execution_strategies`
- method and regime descriptors such as `method_tags`, `regime_tags`, and `derivative_status`
- notes and engineering annotations
- downstream routing fields `downstream_supported` and `downstream_aliases`

## Downstream-kernel entry contract

Each row in `downstream_kernels` must contain:

- `alias`
- `public_name`
- `family`
- `notes`
- `capability`

When the alias distinguishes real and complex routing, the row may also contain:

- `complex_public_name`
- `complex_capability`

## Lookup contract

- `lookup_capability(name)` must accept either a downstream alias or a public function name.
- Unknown names must raise `KeyError`.

## JSON artifact contract

- `render_capability_registry_json()` must serialize the registry to JSON.
- The generated report in `docs/reports/function_capability_registry.json` is the current machine-readable artifact for documentation and downstream inspection.

## Source of truth

- `src/arbplusjax/capability_registry.py`
- `src/arbplusjax/public_metadata.py`
- `docs/reports/function_capability_registry.json`
