Last updated: 2026-03-28T00:00:00Z

# API Usability Standard

Status: active

## Purpose

This document defines the practical usability contract for public APIs in this
repo.

The runtime and JAX standards define what an API is allowed to do. This
standard defines how the API should actually be used, taught, and evidenced in
practice.

It is the bridge between:

- public API design
- example notebooks
- benchmark harnesses
- practical calling guidance

The canonical API-kind taxonomy used here is defined in:

- [api_surface_kinds_standard.md](/docs/standards/api_surface_kinds_standard.md)

This standard exists so that production usage is not left implicit or spread
across notebook-local conventions.

Use the same two-layer reading model as the other owner standards:

- reusable JAX-library rule:
  how a production-facing JAX numerical API should be taught and evidenced
- arbPlusJAX specialization:
  the concrete binders, prepared surfaces, notebooks, and diagnostics this repo
  uses to satisfy that rule

## Scope

Apply this standard to:

- public batch binders
- public prepare/apply surfaces
- public diagnostics-bearing surfaces
- canonical example notebooks
- benchmark harnesses that claim production relevance
- practical run guidance under `docs/practical/`

## Core Rule

Every production-facing public API should have a documented intended calling
pattern.

This is a general JAX-library rule. arbPlusJAX then specializes it through its
named public surfaces and retained example/benchmark artifacts.

That pattern should make clear:

- what to bind or prepare once
- what to reuse across repeated calls
- which controls are expected to stay stable
- how CPU versus GPU should be chosen
- where diagnostics are read
- how AD is intended to be applied on the real surface

If the repo cannot state the intended calling pattern clearly, the API is not
yet considered usable enough.

## Required Usability Elements

### 1. One canonical repeated-call pattern

Each major family should have one preferred repeated-call usage pattern.

Examples:

- bind once and reuse for point batches
- prepare once and apply repeatedly for matrix/operator workloads
- warm once and then execute stable-shape calls

Avoid teaching multiple competing patterns without naming one as the default.

### 2. Stable-shape usage must be explicit

If a surface benefits from padding, bucketing, chunking, or stable prepared
shapes, the practical docs and examples must show that explicitly.

Do not require users to infer the stable-shape rule from benchmark code.

### 3. Backend choice must be taught, not assumed

Examples and practical guidance must state when:

- CPU is the preferred backend
- GPU is expected to help
- batch size or shape stability changes that recommendation

The repo should not teach GPU as a universal default for tiny workloads.

### 4. Diagnostics must be part of the taught workflow

When an API exposes diagnostics, the canonical usage should show:

- what diagnostics exist
- how to inspect them
- which fields correspond to execution strategy or reuse boundaries

Diagnostics are part of usability, not just debugging.

### 5. AD must be demonstrated on the real public surface

Canonical usage must demonstrate AD on the actual public or prepared API
surface, not only on a toy helper.

For parameterized families, the practical usage must distinguish:

- variable-direction AD
- continuous parameter-direction AD

Discrete selectors or indices should be called out explicitly as non-AD
controls where that is the policy.

### 6. Notebooks are the executable teaching layer

Canonical example notebooks are the primary executable teaching surface for API
usage.

They should show:

- setup and environment
- stable-shape repeated-call pattern
- backend choice
- diagnostics
- AD usage
- validation and benchmark summaries

The notebook should teach the API a user is supposed to copy, not a hidden
implementation shortcut.

## Relationship To Other Standards

- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)
  owns the broader runtime/API contract
- [backend_realized_performance_standard.md](/docs/standards/backend_realized_performance_standard.md)
  owns backend-realized performance policy
- [example_notebook_standard.md](/docs/standards/example_notebook_standard.md)
  owns notebook structure and retained execution rules
- [point_fast_jax_standard.md](/docs/standards/point_fast_jax_standard.md)
  owns structural fast-JAX readiness, not practical API teaching

## Repo Mapping

In arbPlusJAX, this standard applies most directly to:

- `api.bind_point_batch(...)`
- `api.bind_point_batch_jit(...)`
- interval/basic binders where present
- matrix, sparse, and matrix-free prepare/apply surfaces
- the canonical `example_*.ipynb` notebooks
- practical run guidance and retained benchmark artifacts
