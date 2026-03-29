# Logging Standard

Status: active
Version: v1.0
Date: 2026-03-29

## Purpose

This document defines the repo-wide logging policy for JAX-first numerical
libraries.

Its goal is to make runtime and mathematical logging consistent, structured, and
performance-safe across subsystems.

Interpret this document in two layers:

- general JAX-library rule:
  the reusable rule for structured logging in JAX-facing numerical libraries
- arbPlusJAX specialization:
  the concrete logging levels, diagnostics sources, and documentation/report
  usage in this repo

This standard is a detailed companion to:

- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)
- [error_handling_standard.md](/docs/standards/error_handling_standard.md)
- [backend_realized_performance_standard.md](/docs/standards/backend_realized_performance_standard.md)
- [example_notebook_standard.md](/docs/standards/example_notebook_standard.md)

## Scope

Apply this standard to:

- diagnostics-bearing public APIs
- bound-service and prepared-plan surfaces
- benchmark and notebook helper surfaces that emit structured runtime summaries
- mathematical-method and numerical-status reporting across families

This standard governs emitted logs. It does not replace returned diagnostics or
binding contracts.

## Core Distinction

The repo must distinguish three different things:

1. returned diagnostics
2. emitted logs
3. raised errors or warnings

They are related but not interchangeable.

Required interpretation:

- diagnostics are structured data returned from public surfaces
- logs are host-side emitted records derived from diagnostics and metadata
- errors are control-flow outcomes governed by
  [error_handling_standard.md](/docs/standards/error_handling_standard.md)

The repo must not use logging as a substitute for diagnostics or status fields.

## Performance Invariant

Logging must not slow the mandatory numerical hot path when disabled.

This is the controlling rule for the whole standard.

Required consequences:

- no string formatting inside JIT-compiled kernels
- no logging-side host callbacks inside hot paths
- no mandatory per-iteration Python emission on production throughput paths
- no expensive mathematical narration assembled when `log_level="none"`

Logging is a host-side interpretation layer on top of structured diagnostics and
metadata.

## Required Logging Levels

Public surfaces or host-side runners that support logging should use one of
these levels:

- `none`
- `sparse`
- `full`
- `verbose`

### `none`

- emit no logs
- return values and diagnostics only
- required default for performance-sensitive production paths unless a more
  explicit repo-local default is documented

### `sparse`

Major events only.

Examples:

- selected backend
- chosen method family
- fallback used
- convergence or failure summary
- high-level mathematical regime note

### `full`

Structured execution and numerical summary.

Examples:

- setup and routing choice
- prepare/apply or binder reuse story
- compile/prewarm/cache behavior
- iteration counts and residual summaries
- estimator standard error and validation summaries
- mathematical regime classification
- approximation order, steps, poles, probes, or truncation settings

### `verbose`

Everything in `full` plus step-level or investigation-grade detail.

Examples:

- per-iteration solver traces
- restart, recycle, or deflation events
- probe-block progression
- contour, quadrature, or pole metadata
- regime-switch detail
- containment inflation causes
- derivation-adjacent mathematical notes

Step-level verbose logging should be available only on explicit debug surfaces
or dedicated debug runs. It must not be silently enabled on normal high-level
APIs.

## Required Log Content Families

Logging should separate two content families.

### 1. Runtime / execution logging

Typical fields:

- `backend_requested`
- `backend_used`
- `compile_event`
- `cache_hit`
- `cache_miss`
- `prewarm_used`
- `plan_reused`
- `shape_bucket`
- `pad_to`
- `chunk_size`
- `timing_class`

### 2. Mathematical / numerical logging

Typical fields:

- `method_requested`
- `method_used`
- `regime`
- `structure_assumption`
- `residual`
- `relative_residual`
- `stderr`
- `iterations`
- `probe_count`
- `pole_count`
- `step_count`
- `guarantee_level`
- `validity`
- `fallback_used`

Logs may combine these families in one structured record, but both should
remain conceptually visible.

## Required Emission Model

Logging should be hook-based or logger-based, not implicit global printing from
numerical kernels.

Preferred public controls:

- `log_level`
- `log_hook`
- `log_tag`
- `log_io`

Equivalent structured interfaces are acceptable if they remain public,
documented, and shared across subsystems.

Recommended behavior:

- `log_hook(record)` receives structured dict-like records
- formatting into prose happens outside the kernel and preferably outside the
  lowest-level public numerical surface
- `log_tag` may carry caller identity such as notebook section, benchmark row,
  or service route

## Required I/O Policy

Logging I/O policy must be explicit whenever a public surface, benchmark, or
notebook runner can emit logs.

Preferred values:

- `log_io="none" | "inline" | "file"`

Meaning:

- `none`: no emitted log output
- `inline`: host-side emitted output during the run, such as notebook cells,
  CLI summaries, or CI output
- `file`: structured records written to an explicit artifact file or output path

Required rules:

- numerical kernels must never write files directly
- file logging belongs to outer runners, notebooks, benchmark harnesses,
  service wrappers, or diagnostics adapters
- `file` logging must be opt-in and target an explicit path or managed artifact
  sink
- ordinary API calls must not create retained log files by default

If a surface does not expose `log_io` directly, its surrounding runner or
documented usage pattern must still obey this model.

## Required Record Shape

When a surface emits structured logs, the records should include:

- `level`
- `event`
- `family`
- `surface`
- `status_code` when status is relevant
- `log_tag` when supplied

Recommended additional fields:

- `backend_requested`
- `backend_used`
- `method_requested`
- `method_used`
- `mode`
- `timing_s`
- `compile_event`
- `cache_hit`
- `fallback_used`
- `note`

If logs describe a diagnostics-bearing call, they should be derivable from the
returned diagnostics rather than rebuilt from hidden state.

For `file` logging, the serialized form should be deterministic and structured.

Preferred formats:

- JSONL for event streams
- JSON for compact summaries
- Markdown only for postprocessed human-facing reports, not the primary machine
  log stream

## Retention and Artifact Rule

The repo must distinguish transient inline logs from retained artifact logs.

Required rules:

- transient local debug output may use `inline`
- retained benchmark, notebook, and report logs should use explicit artifact
  paths
- file logging must respect the repo's governed artifact layout rather than
  writing ad hoc files into source directories
- rich rendered reports may be derived from file logs, but should not replace
  them as the primary machine-readable record

For arbPlusJAX, retained file logs should follow the existing benchmark/output
artifact placement standards rather than inventing family-local directories.

## Relation To Diagnostics

Logging must layer on top of diagnostics rather than replacing them.

Required rules:

- a caller must be able to inspect important runtime/numerical status without
  enabling log emission
- logs may summarize diagnostics but must not be the only place where fallback,
  convergence, or validity changes are recorded
- `log_level="none"` must still allow diagnostics-bearing surfaces to function
  normally

## Relation To Error Handling

Logging is not an error policy.

Required rules:

- do not use logs as the only signal for numerical failure
- do not depend on log output for fallback visibility
- emitted warnings must still be grounded in structured status/error semantics
- strict failure behavior is controlled by
  [error_handling_standard.md](/docs/standards/error_handling_standard.md), not
  by log verbosity
- `log_io="file"` must not be the only way to recover runtime status

## Benchmark and Notebook Rule

Benchmarks and notebooks may emit richer logs than low-level APIs, but they
must still obey the off-hot-path rule.

Required practice:

- collect diagnostics first
- render logs or narrative summaries second
- benchmark timing must exclude rich formatting and verbose explanatory output
  unless that overhead is the explicit subject of measurement
- inline and file logging should be measured separately from the core numeric
  timing when performance claims are made

Notebooks may present prose explanations of the mathematics, but those should be
derived from diagnostics and metadata rather than requiring the numerical
function itself to emit prose.

## Mathematical Narration Rule

This repo explicitly allows mathematical logging, but only outside the hot path.

That includes:

- explaining which asymptotic regime or approximation family was selected
- reporting convergence theory proxies such as residual or quadrature gap
- reporting why a guarantee was weakened or preserved
- naming the solver, estimator, or approximation family used

This mathematical narration must be:

- structured first
- prose second
- optional always

## API Surface Naming Rule

Preferred public names:

- `log_level`
- `log_hook`
- `log_tag`
- `log_io`

Equivalent structured controls are acceptable, but local aliases should not
fragment across families without a documented reason.

## Anti-patterns

- printing from inside JIT-compiled kernels
- building verbose prose strings before checking `log_level`
- using logging instead of diagnostics/status fields
- hiding fallback or guarantee downgrade only in emitted logs
- enabling per-iteration logs on standard throughput APIs by default
- using notebook-local ad hoc log schemas that do not match runtime semantics

## Enforceability

A family is considered compliant only if all of the following are true:

1. `log_level="none"` is supported directly or via a clear default behavior
2. logging-disabled execution does not perform rich formatting in the hot path
3. emitted logs are structured rather than print-only prose
4. runtime and mathematical logging are both representable when relevant
5. fallback and numerical status remain available through diagnostics even when
   logging is disabled
6. at least one test, benchmark, or notebook demonstrates the intended logging
   path outside the numerical kernel

## Required Evidence

Each substantial family that exposes logging should have:

- at least one diagnostics-bearing test or contract showing log-compatible
  structured metadata
- at least one notebook, practical doc, or benchmark that demonstrates the
  intended logging/readout path
- explicit evidence that the standard throughput path remains logging-free or
  logging-light by default

## arbPlusJAX Specialization

For this repo:

- emitted mathematical narration should usually live in notebooks, practical
  guides, benchmark reports, and diagnostics formatters
- the low-level numerical surfaces should prefer structured diagnostics-bearing
  APIs and host-side interpretation
- the canonical logging levels are `none`, `sparse`, `full`, and `verbose`
- runtime and mathematical logging should stay compatible with generated
  reports, retained benchmark artifacts, and notebook teaching surfaces
