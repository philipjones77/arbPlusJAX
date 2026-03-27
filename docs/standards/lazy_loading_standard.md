Last updated: 2026-03-26T00:00:00Z

# Lazy Loading Standard

Status: active

## Purpose

This document defines the repo policy for minimizing import-time load cost
while keeping the public API discoverable and stable.

It covers both:

- package-root and subpackage lazy import boundaries
- public routing/module lazy-loading boundaries inside the runtime surface

Public routing layers must not eagerly import heavy runtime families, provider
backends, or fallback implementations.

Companion documents:

- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)
- [startup_import_boundary_standard.md](/docs/standards/startup_import_boundary_standard.md)
- [startup_compile_standard.md](/docs/standards/startup_compile_standard.md)
- [repo_standards.md](/docs/standards/repo_standards.md)

## Required Policy

1. Package root and public entry modules must stay import-light.
   The package root may establish runtime defaults and lazy dispatch helpers,
   but it must not eagerly import broad runtime families only for convenience.

2. Public routing modules must stay import-light.
   Files such as package roots, `api.py`, metadata loaders, and public routing
   layers must not directly import heavy provider modules at module import
   time.

3. Heavy families must be resolved through shared lazy helpers.
   Preferred mechanisms:
   - `lazy_attr(...)`
   - `lazy_pair(...)`
   - `resolve_lazy_callable(...)`
   - `resolve_lazy_pair(...)`
   - module `__getattr__` only where namespace compatibility requires it

4. Lazy loading must be centralized.
   Do not create new ad hoc lazy-loader implementations in each module when
   [lazy_imports.py](/home/phili/projects/arbplusJAX/src/arbplusjax/lazy_imports.py)
   can express the same boundary.

5. Provider and fallback modules must load only when selected.
   This includes:
   - alternative implementation families
   - GPU/provider backends
   - interval fallback wrappers
   - optional docs/benchmark helpers

## Allowed Eager Work

At package import time, the following are acceptable:

- lightweight runtime-default setup such as shared dtype/x64 policy
- defining public export lists
- defining lazy dispatch helpers such as `__getattr__`
- importing small helper modules required to establish those defaults

The following are not acceptable as default eager work:

- importing broad function families only for convenience
- importing benchmark, test, plotting, comparison, or notebook tooling
- importing optional backends just to probe availability
- importing large implementation subtrees before a public symbol from them has
  been requested

## Package-Root Rule

- `import arbplusjax` should keep the eager `arbplusjax.*` module set minimal.
- Public submodules exported from the package root should resolve lazily via an
  explicit dispatch layer such as `__getattr__`.
- Accessing `arbplusjax.<module>` should load and cache that module on first
  access.

## Subpackage Facade Rule

- Large subpackages may use the same lazy-facade pattern when they expose a
  public surface broad enough to justify it.
- Subpackage lazy exports should map names to explicit owning modules rather
  than using implicit wildcard imports.

## Optional Dependency Rule

- Optional or comparison-only dependencies should never be imported eagerly
  from the package root or from startup-sensitive routing modules.
- Availability probes for optional stacks should remain local to the feature,
  benchmark, or comparison entrypoint that needs them.

## Banned Patterns

- `from . import <heavy family>` in public routing modules when that family is
  not required on the cold path
- eager package-root imports of broad runtime families only to make the root
  namespace convenient
- broad eager import blocks in `api.py` only to populate routing tables
- local one-off lazy proxy implementations when the shared lazy helper is
  sufficient
- using lazy loading only in tests/docs while production routing still imports
  eagerly

## Acceptable Compatibility Escape Hatch

If a public module historically exported many family-specific symbols, it may
use module `__getattr__` to preserve direct access while keeping those symbols
off the cold path.

That escape hatch must not reintroduce eager import through `dir(...)`-based
registration in a startup-sensitive module.

## Enforcement

Primary runtime helper:

- [lazy_imports.py](/home/phili/projects/arbplusJAX/src/arbplusjax/lazy_imports.py)

Representative tests:

- [test_lazy_import_policy.py](/home/phili/projects/arbplusJAX/tests/test_lazy_import_policy.py)
- [test_startup_compile_policy.py](/home/phili/projects/arbplusJAX/tests/test_startup_compile_policy.py)
- [test_api_startup_lazy_loading.py](/home/phili/projects/arbplusJAX/tests/test_api_startup_lazy_loading.py)
- [test_family_import_boundaries.py](/home/phili/projects/arbplusJAX/tests/test_family_import_boundaries.py)

## Consolidation Note

This document is the canonical lazy import/loading policy.

The older `lazy_import_standard.md` surface has been consolidated into this
document so package-root import policy and public routing lazy-loading policy
live in one place.
