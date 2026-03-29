Last updated: 2026-03-26T00:00:00Z

# Startup Import Boundary Standard

## Purpose

Define what may and may not load during package import, `api` import, point
calls, and interval/mode calls.

This is the canonical import-boundary policy for startup-sensitive runtime
surfaces.

Companion documents:

- [startup_compile_standard.md](/docs/standards/startup_compile_standard.md)
- [lazy_loading_standard.md](/docs/standards/lazy_loading_standard.md)
- [operational_jax_standard.md](/docs/standards/operational_jax_standard.md)
- [metadata_registry_standard.md](/docs/standards/metadata_registry_standard.md)

## Import Tiers

### 1. Package Import

`import arbplusjax` must stay minimal.

Allowed:

- package namespace glue
- lightweight public exports
- static metadata helpers
- import-tier declarations

Disallowed:

- heavy numeric family modules
- provider or alternative backend modules
- interval/mode wrappers
- benchmark or docs helper modules

### 2. API Import

`from arbplusjax import api` may load only runtime routing and point-safe
binding machinery.

Allowed:

- routing tables
- dtype normalization helpers
- lazy import descriptors
- static metadata access
- point-safe core helpers

Disallowed:

- heavy family implementation modules unless required for the routing layer
- interval/mode wrappers
- provider/alternative backends
- benchmark/docs-only helpers

### 3. Point Call

A point call may load only the requested point family and its direct runtime
dependencies.

Allowed:

- the selected point-family module
- shared point-core helpers
- direct low-level math helpers required by that point family

Disallowed:

- unrelated point families
- interval/mode wrapper modules
- provider/alternative backends unless the chosen implementation is itself a
  provider backend

### 4. Interval/Mode Call

Interval, basic, adaptive, and rigorous calls may load their mode-specific
wrapper modules on demand.

Allowed:

- the selected interval/mode wrapper
- direct precise/adaptive dependencies required by that wrapper
- provider backends only when the selected mode or implementation needs them

Disallowed:

- eager load of all interval wrappers during `api` import
- eager load of unrelated families during one interval call

## Required Rules

1. Public runtime modules must implement explicit import tiers.
   Runtime code must distinguish package cold path, `api` cold path,
   point-on-demand, interval/mode-on-demand, and provider-on-demand modules.

2. Public routing layers must not hide heavy imports in module top level.
   If a family is not required on the cold path, it must be resolved lazily.

3. Import boundaries must be testable.
   The repo must keep subprocess tests that inspect `sys.modules` after:
   - package import
   - `api` import
   - representative point calls
   - representative interval/mode calls

4. Static metadata must respect the same boundaries.
   Metadata access must not widen the runtime import graph.

## Enforcement

The code-level owner for this policy is:

- [import_tiers.py](/src/arbplusjax/import_tiers.py)

Required test coverage includes:

- [test_import_tiers_policy.py](/tests/test_import_tiers_policy.py)
- [test_api_startup_lazy_loading.py](/tests/test_api_startup_lazy_loading.py)
- [test_family_import_boundaries.py](/tests/test_family_import_boundaries.py)
