# Error Handling Standard

Status: active
Version: v1.0
Date: 2026-03-29

## Purpose

This document defines the repo-wide error, status, fallback, and diagnostics
policy for JAX-first numerical libraries.

It is intended to be enforceable across subsystems so that numerical families do
not invent incompatible behavior for hard contract errors, numerical-status
failures, or fallback events.

Interpret this document in two layers:

- general JAX-library rule:
  the reusable rule for numerical libraries that expose JAX-facing APIs
- arbPlusJAX specialization:
  the concrete mode naming, diagnostics surfaces, and generated reports used in
  this repo

This standard is a detailed companion to:

- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)
- [api_surface_kinds_standard.md](/docs/standards/api_surface_kinds_standard.md)
- [contract_and_provider_boundary_standard.md](/docs/standards/contract_and_provider_boundary_standard.md)
- [backend_realized_performance_standard.md](/docs/standards/backend_realized_performance_standard.md)

## Scope

Apply this standard to:

- scalar, interval, dense, sparse, matrix-free, special-function, and curvature
  public APIs
- diagnostics-bearing surfaces
- prepared-plan and bound-service surfaces
- benchmark/example-facing public helpers when they expose runtime status or
  fallback behavior

This standard governs runtime behavior and public semantics. It does not define
the mathematical formulas themselves.

## Core Model

Every public numerical call must classify failure or degraded execution into one
of four categories:

1. hard contract error
2. numerical-status failure
3. containment or validity degradation
4. fallback or execution-policy change

The repo must not collapse all four into one undifferentiated exception path.

## Required Distinction

### 1. Hard contract errors

These are caller-contract failures and must raise immediately.

Typical examples:

- invalid shape or rank
- impossible dtype or mode combination
- unsupported required backend with no documented fallback
- impossible structure claim such as `hermitian=True` on a clearly
  incompatible payload
- invalid static control values such as negative iteration count or malformed
  solver family name

Hard contract errors must not be silently downgraded into diagnostics-only
status.

### 2. Numerical-status failures

These are runtime numerical outcomes and must not automatically be treated as
programmer errors.

Typical examples:

- solver non-convergence
- residual above tolerance
- iteration or probe budget exhausted
- estimator standard error above requested threshold
- loss of definiteness or violated structural assumption discovered at runtime
- method-specific stability warning

These must surface through a structured status object and may optionally raise
according to the caller-selected error policy.

### 3. Containment or validity degradation

This category is required when the library exposes interval, containment, or
validity-bearing outputs.

Typical examples:

- interval inflation
- enclosure invalidation
- downgrade from stronger guarantee to weaker guarantee
- adaptive method unable to certify the requested condition

The default behavior may differ by mode:

- `point`: usually return value plus status/diagnostics
- `basic`: usually preserve containment semantics and mark degradation
- `adaptive`: usually return the achieved certification level and note why a
  stronger target was not met
- `rigorous`: usually fail closed, invalidate explicitly, or raise when a
  claimed guarantee cannot be delivered

### 4. Fallback or execution-policy change

The repo must distinguish “result obtained by a different path” from both hard
failure and ordinary success.

Typical examples:

- GPU request routed to CPU fallback
- specialized structured solver routed to a general solver
- exact route routed to an estimator route
- prepared/cached path unavailable and replaced by uncached execution

Fallback must be visible in returned status or diagnostics even when the result
is otherwise successful.

## Required Public Controls

### Error policy

Diagnostics-bearing public APIs should expose an explicit error policy when
runtime numerical failure is relevant.

Preferred policy names:

- `error_policy="raise" | "diagnostics" | "warn"`

Equivalent structured policies are acceptable if they are public, documented,
and shared across subsystems.

Meaning:

- `raise`: numerical-status failures and validity failures that are marked
  actionable should raise
- `diagnostics`: return status/diagnostics without raising
- `warn`: return status/diagnostics and emit a host-side warning outside the
  numerical hot path

If a public surface does not expose an explicit policy, its default behavior
must still match this standard and be documented.

### Fallback policy

Public surfaces that may switch backend or method should expose fallback policy
explicitly when the behavior materially affects reproducibility or guarantees.

Preferred controls:

- `allow_fallback=True/False`
- `backend="cpu" | "gpu" | "auto"`
- explicit method family selectors

Silent fallback that changes semantics or guarantees is not allowed.

### Diagnostics level

Diagnostics detail should be a separate concern from error policy.

Preferred public values:

- `diagnostics_level="none" | "summary" | "full"`

Meaning:

- `none`: no extra diagnostics payload beyond ordinary return value
- `summary`: cheap status fields only
- `full`: richer structured metadata

The library may expose these controls through dedicated diagnostics-bearing
surfaces instead of inline kwargs, but the semantic distinction must remain
visible.

## Required Status Semantics

Diagnostics-bearing surfaces must expose stable status fields when runtime
status matters.

Minimum required fields:

- `status_code`
- `success`
- `converged` when convergence is relevant
- `fallback_used`
- `failure_kind` when not fully successful
- `message` or `note`

Recommended additional fields when applicable:

- `residual`
- `relative_residual`
- `stderr`
- `iterations`
- `budget_exhausted`
- `validity`
- `guarantee_level`
- `backend_requested`
- `backend_used`
- `method_requested`
- `method_used`

### Shared status codes

Use these codes unless a narrower family-specific code is genuinely necessary:

- `success`
- `nonconverged`
- `budget_exhausted`
- `fallback_used`
- `invalidated`
- `unsupported`
- `backend_unavailable`
- `structure_violation`
- `numerical_instability`
- `tolerance_not_met`
- `stderr_too_large`

Family-specific codes may extend this list but must not replace the generic
codes for common cases.

## Status Payload Serialization Rule

Status and diagnostics payloads should be transportable and serializable at the
outer API, benchmark, report, and notebook layers.

Required rules:

- status payloads should prefer plain Python/JAX-safe scalars, arrays, and
  lightweight structured containers
- diagnostics-bearing public surfaces should not require callers to inspect
  exception text to recover status semantics
- rich objects that are not directly serializable are acceptable internally, but
  the public/report-facing layer must provide a deterministic serialized view

Recommended serialized fields:

- `status_code`
- `success`
- `converged`
- `fallback_used`
- `failure_kind`
- `message`
- relevant numeric metrics such as `residual`, `stderr`, or `iterations`

The serialized view belongs at the host/report layer, not in the compiled hot
path.

## Mode-Specific Rules

### Point mode

- hard contract errors must raise
- numerical-status failures should default to diagnostics-bearing return or a
  documented diagnostics-bearing companion surface
- callers may opt into strict raising through error policy

### Basic mode

- hard contract errors must raise
- containment degradation must be represented explicitly in the result or
  diagnostics
- basic mode must not quietly return an apparently ordinary result when the
  containment story changed materially

### Adaptive mode

- report achieved versus requested target when adaptive stopping or validation
  is part of the method
- budget exhaustion or tolerance miss must be explicit

### Rigorous mode

- if the claimed rigorous guarantee cannot be established, the surface must
  either:
  - return an explicit invalidated/failed-rigorous status, or
  - raise under a documented strict policy
- rigorous mode must not silently downgrade to a weaker guarantee while still
  presenting the result as rigorous

## Diagnostics and Hot-Path Rule

Status and diagnostics must be designed so they do not slow the mandatory
numerical hot path when disabled.

Required rules:

- no exception-like control flow inside the steady-state compiled kernel for
  ordinary diagnostics-disabled success cases
- no string formatting inside compiled kernels
- no host callbacks for routine status reporting
- no per-iteration Python logging on standard throughput paths

Diagnostics should be returned as compact structured payloads and interpreted on
the host side.

## Warning Rule

Warnings are allowed only as a host-side interpretation layer.

That means:

- warnings may be emitted from outer wrappers, benchmark harnesses, notebook
  helpers, or diagnostics formatters
- warnings must not be emitted directly from the hot numerical kernel
- warning text should be derived from status codes and structured diagnostics,
  not reimplemented ad hoc in every subsystem

Preferred mapping:

- hard contract errors: raise
- soft numerical-status failures under `error_policy="warn"`: return status and
  emit host-side warning
- fallback events: do not warn by default unless they materially weaken a
  guarantee or violate an explicit caller expectation

## Exceptions and Messages

When raising:

- use concrete exception types where available
- keep messages factual and action-oriented
- include the relevant public control, mode, backend, or structure term
- do not encode the whole diagnostics payload into the exception string

Recommended message content:

- what contract was violated
- what control or input caused it
- what the valid alternatives are when obvious

## Fallback Reporting Rule

Fallback must always be visible when it changes execution semantics, backend, or
guarantee level.

Minimum requirements:

- set `fallback_used=True`
- record requested and used backend/method when relevant
- note whether the fallback preserves or weakens the original guarantee

Fallback must not be discoverable only through emitted logs.

## Reproducibility Rule

When stochastic estimators are involved, status and error handling must preserve
reproducibility information.

Required when applicable:

- RNG seed or seed policy visibility
- probe or sample count
- whether probes were reused across gradients or repeated calls
- whether fallback changed the stochastic method family

## API Surface Naming Rule

When a family exposes explicit diagnostics-bearing or report-bearing variants,
the names should remain consistent with the runtime standard.

Preferred names:

- `return_diag`
- `return_report`
- `error_policy`
- `diagnostics_level`

Equivalent structured controls are acceptable, but the repo should not invent
incompatible local names family by family without a documented reason.

## Enforceability

A public family is considered compliant only if all of the following are true:

1. hard contract errors raise on invalid input
2. numerical-status failures are distinguishable from hard errors
3. fallback is visible in status or diagnostics
4. mode-specific behavior is documented for `point/basic/adaptive/rigorous`
5. diagnostics-disabled hot paths do not pay for warning/log formatting
6. tests cover at least one hard-failure case and one diagnostics/status case

## Required Evidence

Each substantial family should have:

- at least one test that verifies hard contract failure
- at least one test that verifies status or diagnostics for a soft numerical
  failure or fallback
- at least one practical/example/documentation surface that shows how callers
  inspect status instead of relying only on exceptions

## Anti-patterns

- raising for every numerical imperfection
- silently swallowing non-convergence
- silently downgrading rigorous/basic guarantees
- hiding backend or method fallback
- returning plain booleans instead of structured status
- building warning or logging strings inside the compiled hot path
- making fallback discoverable only through notebook prose

## arbPlusJAX Specialization

For this repo:

- the four canonical evaluation modes are `point`, `basic`, `adaptive`, and
  `rigorous`
- diagnostics-bearing public surfaces are governed by
  [api_surface_kinds_standard.md](/docs/standards/api_surface_kinds_standard.md)
- generated reports, notebooks, and practical docs should teach status-driven
  inspection for matrix-free, sparse, and estimator-heavy families
- status and fallback visibility must remain compatible with the repo's
  generated report and benchmark surfaces
