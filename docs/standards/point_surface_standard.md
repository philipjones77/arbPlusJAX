Last updated: 2026-03-26T00:00:00Z

# Point Surface Standard

## Purpose

Point APIs must be minimal-load, stable-shape, and fast-JAX first.

This standard defines what the repo means by a compliant public point surface.

Companion documents:

- [point_fast_jax_standard.md](/docs/standards/point_fast_jax_standard.md)
- [startup_import_boundary_standard.md](/docs/standards/startup_import_boundary_standard.md)
- [startup_compile_standard.md](/docs/standards/startup_compile_standard.md)

## Required Policy

1. Point is the default repeated-evaluation path.
   If a public family supports a fast JAX point path, that path should be the
   default teaching and runtime path for repeated bulk evaluation.

2. Point surfaces must be minimal-load.
   A point call must not import interval/mode wrappers or unrelated families
   unless explicitly required by the selected implementation.

3. Point surfaces must expose a stable-shape compiled path.
   Required public forms include one or more of:
   - fixed batch
   - `pad_to`
   - prepare/apply for matrix-like workloads
   - bound compiled point batch via `bind_point_batch_jit(...)`

4. Point routing must be fast-JAX first.
   Do not route ordinary repeated point traffic through precise/adaptive
   interval machinery when a direct JAX point kernel exists.

5. Point docs and examples must teach the stable-shape path first.
   Public examples should prefer padded or fixed batch paths over ad hoc
   `jax.jit` on arbitrary caller arrays.

## Banned Patterns

- teaching interval/basic wrappers as the primary point workload path
- importing point families eagerly into `api` startup when they can be loaded
  on demand
- treating `pad_to` as an obscure escape hatch rather than a canonical compile
  contract
- scattering family-specific compile policy across many leaf helpers

## Required Evidence

Each hot point family should provide:

- a point boundary test proving minimal-load behavior
- a point compile/startup probe
- at least one contract or engineering test using the stable-shape point path

Representative evidence in this repo:

- [test_family_import_boundaries.py](/home/phili/projects/arbplusJAX/tests/test_family_import_boundaries.py)
- [test_point_fast_jax_categories.py](/home/phili/projects/arbplusJAX/tests/test_point_fast_jax_categories.py)
- [dirichlet_point_startup_probe.py](/home/phili/projects/arbplusJAX/benchmarks/dirichlet_point_startup_probe.py)
- [hypgeom_point_startup_probe.py](/home/phili/projects/arbplusJAX/benchmarks/hypgeom_point_startup_probe.py)

