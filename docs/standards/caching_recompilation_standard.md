Last updated: 2026-03-24T00:00:00Z

# Caching And Recompilation Standard

Status: active

## Purpose

This document defines the repo-wide policy for:

- cache-aware public calling patterns
- JIT cache and recompilation discipline
- prepared-plan reuse
- example and benchmark evidence for compliant reuse behavior

This is a specialized companion to:

- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)
- [engineering_standard.md](/docs/standards/engineering_standard.md)

## Scope

Apply this standard to:

- public bound API helpers such as `api.bind_point_batch()` and related compiled binders
- cached prepare/apply matrix, sparse, transform, and operator-plan surfaces
- repeated-call service or notebook patterns where stable shapes and compile-relevant controls matter
- benchmarks and canonical examples that claim production-quality JAX calling behavior

## Core Policy

- Cache-aware calling patterns must be explicit, not left implicit in examples or benchmarks.
- Reuse-oriented surfaces should keep compile-relevant controls stable across repeated calls.
- Diagnostics, tracing, and software-comparison work must stay outside the mandatory numeric hot path.
- Recompilation minimization is a governed runtime concern, not just a benchmark concern.
- Cache-aware surfaces should be discoverable in generated reports and auditable through pytest-owned contracts.

## JIT Cache Policy

- Stable dtype, shape, method, mode, and backend controls should reuse compiled callables when values change but compile-relevant controls do not.
- Changes to shape, dtype, device/backend selection, method family, or other static controls may legitimately trigger recompilation.
- Public APIs should keep compile-relevant controls explicit rather than burying them in hidden globals.
- Bound or cached compiled callables should prefer stable static kwargs and stable batch shapes.

## Prepared-Plan Policy

- Families with meaningful symbolic or structural preparation should expose prepare/apply or equivalent reusable plan surfaces.
- Prepared objects should be safe to reuse across repeated value calls when the structural inputs they depend on remain unchanged.
- Examples and benchmarks should prefer plan reuse over repeated ad hoc rebuilding when the surface is intended for repeated calls.
- If a family does not benefit from prepared reuse, the docs should say so explicitly.

## Invalidation Policy

The docs and examples should treat these as ordinary cache-invalidation boundaries:

- dtype change
- shape or padding target change
- compile-relevant method or mode change
- precision-policy change when it changes the compiled kernel
- backend or device change
- structural operator or matrix change for prepared-plan surfaces

## Diagnostics Policy

- Cache behavior may be reported through structured metadata or diagnostics, but cache reporting must remain optional.
- Compile or recompile observations belong at the outer API/diagnostics layer, not inside numeric kernels.
- Report-facing cache metadata should describe whether a surface is binder-based, prepare/apply-based, padded-batch aware, or otherwise reuse-oriented.

## Example Notebook Requirements

Canonical example notebooks for cache-aware families should:

- show the reusable public entrypoint directly
- demonstrate binder reuse, prepared-plan reuse, or both where relevant
- keep dtype and compile-relevant controls stable across repeated calls
- show optional `pad_to` or chunking when variable-size traffic would otherwise cause avoidable recompiles
- make CPU-current but GPU-portable calling patterns explicit

## Benchmark Requirements

Canonical benchmarks for cache-aware families should:

- separate cold, warm, and recompile-sensitive behavior when JAX compile cost is part of the practical calling contract
- use stable repeated-call patterns by default
- call out intentional recompiles separately from accidental shape churn
- exercise cached prepare/apply surfaces where those are the production-recommended path

## Generated Inventory Rule

The repo should maintain a generated report that records where cache-aware public surfaces exist and which canonical examples or benchmarks are expected to demonstrate those reuse patterns.

The current generated inventory is:

- [cache_aware_surface_inventory.md](/docs/reports/cache_aware_surface_inventory.md)

Refresh through:

- `python tools/cache_aware_surface_report.py`
- `python tools/check_generated_reports.py`
