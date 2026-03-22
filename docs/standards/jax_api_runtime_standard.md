# JAX API and Runtime Standard

Status: active
Version: v1.1
Date: 2026-03-21

## Purpose

This document records the common API, runtime, dtype, diagnostics, logging, and
recompilation discipline for JAX-first numerical libraries.

It is intended to be reusable across related JAX-first libraries so they expose
numerical runtimes in a consistent way.

This is a standards document, not a contract. Binding runtime/API guarantees
belong in library-specific contract documents.

## Scope

Use this standard for libraries that:

- expose public JAX evaluation APIs;
- support JIT and automatic differentiation;
- need consistent dtype handling across subsystems;
- return diagnostics or profiling metadata;
- want low import overhead without hiding the public API.

## Core Principles

- Keep the public surface JAX-first.
- Keep import-time cost low through clear lazy boundaries.
- Keep continuous numerical inputs dynamic when possible.
- Keep discrete algorithm choices explicit and compile-relevant.
- Keep ordinary parameter-value changes cheap for callers by avoiding unnecessary recompilation where shapes and compile-relevant controls stay fixed.
- Keep diagnostics opt-in and structured.
- Keep logging opt-in and outside the numerical kernel.
- Keep runtime policy shared across subsystems instead of re-implementing it in each module.
- Keep validation, benchmarking, and software-comparison layers outside the production numerical hot path.
- Do not let observability or validation features slow normal numerical evaluation.

## Public API Shape

### Package boundary

- The package root is the authoritative public API boundary.
- A symbol is public only if exported from the package root.
- Internal module layout is not part of the contract unless re-exported.

### Unified facade

- Expose a grouped unified facade for major capability areas.
- Keep direct concrete exports available for callers who need explicit control.
- Use the facade for discoverability, not to hide variant implementations.

Recommended groups:

- `runtime`: shared runtime policy and config helpers
- `diagnostics`: optional tracing/profiling helpers
- domain-specific groups such as `foxh`, `mb`, `quad`, `viz`

### Variant exposure

- Expose materially different execution paths separately.
- Do not collapse direct, batched, adaptive, raw, tuned, and policy-driven paths into one opaque selector.
- High-level auto-routing is allowed, but it does not replace explicit variants.

## Runtime Configuration Standard

### Shared runtime layer

- Provide one shared runtime module for dtype policy, config dataclasses, and cross-subsystem normalization.
- Reuse shared config types across subsystems when controls are conceptually the same.
- Keep config objects lightweight and side-effect free at import time.

Recommended shared config types:

- `ContourConfig`
- `BatchConfig`
- `QuadratureConfig`
- any additional config types only when they represent a genuinely distinct control family

### Recommended config builders

- Provide recommended config builders for common settings.
- Builders should return plain config containers, not initialized runtime objects.
- Builders should surface compile-relevant settings explicitly.

## Dtype Policy Standard

### Process-wide default

- Dtype policy is global by default.
- Callers should choose `float32` or `float64` once near process startup.
- Changing dtype policy after JIT compilation is allowed but expected to trigger recompilation.
- Public APIs may support explicit optional dtype overrides when there is a concrete numerical, backend, or performance reason to do so.

### Allowed overrides

- Local dtype overrides are exception paths, not the default model.
- Only permit local overrides through explicit public parameters or config.
- Do not hide dtype changes behind undocumented internal casting.
- When a local override is supported, document whether it is value-only, kernel-affecting, or expected to trigger recompilation.

### Practical rules

- Provide shared helpers for real and complex dtype resolution.
- Promote related outputs consistently so subsystem boundaries do not silently change precision.
- Tests should verify output dtypes on major public paths.

## AD and JIT Standard

### Continuous versus discrete inputs

- Keep continuous numerical inputs dynamic where possible.
- Treat structural or discrete controls as compile-relevant inputs.

Examples of usually dynamic inputs:

- values, coordinates, parameters, contour shifts, tolerances when shape-stable

Expected behavior:

- callers should be able to change ordinary parameter values freely without avoidable recompilation when shapes, dtypes, and compile-relevant controls are unchanged

Examples of usually compile-relevant inputs:

- method names
- window families
- rule families
- grid sizes
- step counts
- batch shapes
- control-flow families

### Minimal recompilation rule

- Make compile-relevant knobs explicit to the caller.
- Avoid hidden module-global mutation for algorithm control.
- Prefer fixed-shape kernels and batch-friendly outputs.
- Keep symbolic planning and canonicalization outside traced numerical kernels unless explicitly documented otherwise.

## Diagnostics Standard

### Numerical diagnostics

- Diagnostics are opt-in.
- Return diagnostics through structured payloads, not printing.
- Use explicit flags such as `return_diag=True`, `return_report=True`, or `full_output=True`.
- Keep returned metadata shape-stable and JIT-safe where it is part of a JAX-facing path.
- Diagnostics must not add mandatory overhead to normal production calls when they are disabled.

### Profiling and tracing

- Keep profiling/tracing in a separate diagnostics layer.
- Attach tracing at outer call boundaries, not inside hot kernels.
- Environment-driven enablement is acceptable for the diagnostics layer, but production kernels should not depend on it.
- Profiling and tracing must remain explicit outer wrappers so normal numerical execution does not pay for them.

Recommended diagnostics capabilities:

- compile timing
- steady-state timing
- recompile timing on changed shape
- peak memory delta
- optional JAXPR capture
- optional HLO capture
- optional JAX profiler traces

## Logging Standard

- Logging is opt-in.
- Logging should be hook-based or callback-based, not implicit global printing from numerical kernels.
- Logging payloads should be structured records, not formatted prose strings.
- Logging should layer on top of explicit metadata rather than replacing returned diagnostics.
- If a subsystem needs runtime logs, accept hooks such as `log_hook` and optional tags such as `log_tag`.
- Logging must not become a default side effect on performance-sensitive numerical paths.

## Lazy Loading Standard

- Use lazy loading at clear public boundaries when it reduces import cost or avoids optional heavyweight dependencies.
- Do not scatter lazy indirection unpredictably across runtime kernels.
- Favor lazy package-root exports and lazy unified facades.

## Optional Dependency Standard

- Optional integrations may exist, but they should not define the default runtime contract.
- Reference backends and validation stacks belong outside the hot path.
- Optional exports may resolve to `None` only when that behavior is documented and tested.

Examples:

- symbolic tooling
- plotting
- reference precision backends
- external numerical engines

## Tests, Benchmarks, and Software Comparison Standard

- Tests, benchmarks, and external-software comparisons are required validation layers, but they are not part of the default numerical execution path.
- Validation against reference backends such as `mpmath`, Mathematica, Boost, or other external software must be explicit and opt-in where it is expensive.
- Benchmark harnesses, guardrails, and comparison runners should live in dedicated test, benchmark, or experiment surfaces rather than inside production kernels.
- Production APIs may expose explicit validation or comparison flags only when the extra cost is obvious to callers.
- External comparison code must not silently run during ordinary numerical evaluation.
- Artifact capture from tests and benchmarks belongs in benchmark or experiment artifact roots, not in normal runtime code paths.
- The success criterion is two-layered:
  - production calls stay fast and focused on numerical evaluation;
  - validation layers remain reproducible and strong enough to catch regressions against trusted references.

## Diagnostics and Logging Naming

Use these names consistently where they fit:

- `return_diag` for compact evaluator diagnostics
- `return_report` for richer workflow or orchestration metadata
- `full_output` for quadrature-style value-plus-metadata outputs
- `log_hook` for explicit logging callbacks
- `log_tag` for caller-supplied log record labeling

## What Other Libraries Need To Know

- Choose and document one public API boundary.
- Choose and document one shared runtime layer.
- Choose and document one dtype policy story.
- Make compile-relevant controls explicit.
- Keep diagnostics and logging separate from numerical kernels.
- Decide which outputs are guaranteed shape-stable.
- Decide which optional dependencies are reference-only versus runtime-capable.
- Test dtype stability, batch behavior, and no-surprise recompilation scenarios.
- Document when recompilation is expected instead of pretending it does not happen.
- Keep tests, benchmarks, and cross-software comparisons strong, but isolate them so they do not slow standard runtime evaluation.
- If optional dtype overrides are supported, document them clearly and test their interaction with recompilation behavior.

## Related Local Documents

Each adopting library should pair this standard with local documents such as:

- binding API/runtime contracts
- architecture and project overview documents
- subsystem implementation-alignment notes
- repository-specific API/runtime mapping reports
