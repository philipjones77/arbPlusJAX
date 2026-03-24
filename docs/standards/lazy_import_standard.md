Last updated: 2026-03-24T00:00:00Z

# Lazy Import Standard

Status: active

## Purpose

This document defines the repo policy for minimizing import-time load cost while
keeping the public API discoverable and stable.

This is a runtime/API companion to:

- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)
- [repo_standards.md](/docs/standards/repo_standards.md)

## Core Policy

- The package root should import only the minimal state required to establish
  repo-wide runtime defaults and lazy public-module access.
- Heavy numerical modules should not be imported eagerly just because the
  package root was imported.
- Lazy loading should happen at clear public boundaries such as the package root
  or intentionally grouped subpackage facades.
- Lazy loading should not be hidden deep inside hot numeric kernels.

## Allowed Eager Work

At package import time, the following are acceptable:

- lightweight runtime-default setup such as the shared dtype/x64 default policy
- defining the public export list
- defining lazy import dispatch helpers such as `__getattr__`
- importing small helper modules required to establish those defaults

The following are not acceptable as default eager package-root work:

- importing broad function families only for convenience
- importing benchmark, test, comparison, plotting, or notebook tooling
- importing optional backends just to probe availability
- importing large implementation subtrees when no public symbol from them has
  been requested yet

## Package-Root Rule

- `import arbplusjax` should keep the eager `arbplusjax.*` module set minimal.
- Public submodules exported from the package root should be resolved lazily via
  an explicit dispatch layer such as `__getattr__`.
- Accessing `arbplusjax.<module>` should load and cache that module on first
  access.

## Subpackage Facade Rule

- Large subpackages may use the same lazy-facade pattern when they expose a
  public surface broad enough to justify it.
- Subpackage lazy exports should map names to explicit owning modules rather
  than using implicit wildcard imports.

## Optional Dependency Rule

- Optional or comparison-only dependencies should never be imported eagerly from
  the package root.
- Availability probes for optional stacks should remain local to the feature,
  benchmark, or comparison entrypoint that needs them.

## Validation Rule

The repo should keep pytest-owned contracts that verify:

- package-root imports do not eagerly import broad `arbplusjax.*` subtrees
- lazy public-module access loads a requested module on demand
- repeated access reuses the loaded module instead of re-importing it

The current contract test is:

- [test_lazy_import_policy.py](/tests/test_lazy_import_policy.py)

## Practical Guidance

- Prefer `import arbplusjax as apj` followed by explicit attribute access when a
  caller wants the public surface without paying to import every family up
  front.
- If a script uses only one large family, importing that concrete submodule
  directly is still acceptable when clarity is improved.
- When adding a new public root export, decide whether it belongs in the eager
  package root or in the lazy export table. Default to the lazy table.
