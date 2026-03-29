Last updated: 2026-03-29T00:00:00Z

# Operational JAX Standard

Status: active

## Purpose

This document defines what the repo means by a compliant operational JAX
surface.

Operational JAX is broader than structural `fast JAX`. It covers:

- minimal-load behavior where relevant
- stable-shape compiled paths
- repeated-call teaching surfaces
- routing policy that prefers JAX-native paths when available

Interpret this document in two layers:

- general JAX-library rule:
  hot public JAX surfaces should be minimal-load, stable-shape, and taught
  through explicit repeated-call paths
- arbPlusJAX specialization:
  the current strongest application is the public `point` surface, because that
  is where the repo has pushed the operational JAX rollout furthest

## Required Policy

### 1. The primary repeated-evaluation path should be JAX-native

If a public family supports a direct JAX path, that path should be the default
teaching and runtime path for repeated bulk evaluation.

### 2. Operational surfaces must be minimal-load

A hot-path call must not import unrelated families or wrapper stacks unless
explicitly required by the selected implementation.

### 3. Operational surfaces must expose a stable-shape compiled path

Required public forms include one or more of:

- fixed batch
- `pad_to`
- shape buckets
- prepare/apply for matrix-like workloads
- bound compiled repeated-call surfaces such as `bind_*_jit(...)`

### 4. Routing must be JAX-first where a direct JAX kernel exists

Do not route ordinary repeated traffic through precise/adaptive machinery when
a direct JAX kernel exists and satisfies the family contract.

### 5. Docs and examples must teach the stable-shape path first

Public examples should prefer padded, fixed-batch, bucketed, or prepared-plan
paths over ad hoc `jax.jit` on arbitrary caller arrays.

## Banned Patterns

- teaching fallback wrappers as the primary repeated-call workload path
- importing hot families eagerly into `api` startup when they can be loaded on
  demand
- treating `pad_to` or shape bucketing as obscure escape hatches rather than
  canonical compile contracts
- scattering family-specific compile policy across many leaf helpers

## Required Evidence

Each hot operational family should provide:

- a boundary test proving minimal-load behavior where startup sensitivity
  matters
- a compile/startup probe or first-use report
- at least one contract or engineering test using the stable-shape path

Representative evidence in this repo includes:

- [test_family_import_boundaries.py](/tests/test_family_import_boundaries.py)
- [test_point_fast_jax_categories.py](/tests/test_point_fast_jax_categories.py)
- [dirichlet_point_startup_probe.py](/benchmarks/dirichlet_point_startup_probe.py)
- [hypgeom_point_startup_probe.py](/benchmarks/hypgeom_point_startup_probe.py)

## Relationship To Other Standards

- [fast_jax_standard.md](/docs/standards/fast_jax_standard.md)
  owns structural fast-JAX readiness
- [startup_import_boundary_standard.md](/docs/standards/startup_import_boundary_standard.md)
  owns import-boundary policy
- [startup_compile_standard.md](/docs/standards/startup_compile_standard.md)
  owns compile/startup discipline
- [api_usability_standard.md](/docs/standards/api_usability_standard.md)
  owns how these surfaces should be taught and used

## arbPlusJAX Specialization

In this repo, the operational JAX rollout is currently strongest on the public
`point` surface. That is why older docs referred to a point-surface standard.
This document generalizes that policy so it applies beyond point as interval,
dense, sparse, matrix-free, and other repeated-call JAX surfaces are hardened.
