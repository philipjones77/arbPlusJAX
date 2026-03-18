Last updated: 2026-03-17T00:00:00Z

# JAX Surface Policy

This repository is allowed to use official public JAX implementations, including official JAX linear algebra.

It is not allowed to depend on SciPy-derived implementation paths in runtime code.

## Allowed in runtime code

- `jax.numpy`
- `jax.numpy.linalg`
- `jax.lax`
- `jax.lax.linalg`
- public JAX autodiff and control-flow APIs such as `jit`, `vmap`, `scan`, `while_loop`, `custom_vjp`, `custom_jvp`, and `lax.custom_linear_solve`

## Discouraged in runtime code

- `jax.experimental.pallas`

Use Pallas only when:
- a measured performance bottleneck exists
- the kernel cannot be expressed cleanly with the standard public JAX surface
- the kernel is covered by dedicated tests and benchmarks

## Forbidden in runtime code

- `jax.scipy`
- `scipy`
- direct `jaxlib` internals
- private JAX internals
- handwritten custom-call / FFI lowering paths unless explicitly approved as an exception

## Benchmark and reference exception

External comparison surfaces may use `scipy` or `jax.scipy` when they are clearly acting as:

- benchmarks
- reference backends
- comparison harnesses
- audit or migration tooling

That exception does not apply to:

- library runtime code under `src/arbplusjax/`
- user-facing examples intended to demonstrate the canonical implementation path

## Implementation rule

When a SciPy-derived implementation appears in runtime code, replace it with one of:

- a public JAX primitive or linear algebra routine
- a repo-owned JAX implementation
- a benchmark-only or reference-only comparison path outside runtime

## Lower-level guidance

- `jax.lax` and `jax.lax.linalg` are preferred lower-level building blocks
- Pallas comes before FFI/custom-call work
- FFI/custom-call work is a last resort because it increases AD, portability, and upgrade risk

## `jax.experimental` guidance

The locally installed JAX surface currently exposes experimental modules such as `checkify`, `compilation_cache`, `jet`, `pallas`, `sparse`, `x64_context`, `shard_map`, and `serialize_executable`.

Repo policy is:

- do not add runtime dependencies on `jax.experimental.*`
- evaluate experimental ideas case-by-case and reimplement the narrow part we actually need in repo-owned code when the behavior is stable enough to justify it

Current adopted example:

- x64 policy is now centralized in `arbplusjax.precision`, including a local `jax_x64_context(...)` helper, instead of scattering direct `jax.config.update(...)` calls or depending on `jax.experimental.x64_context`

Current non-adoptions:

- `jax.experimental.pallas`: still discouraged unless a measured kernel bottleneck forces it
- `jax.experimental.checkify`: useful conceptually, but current validation needs are still handled by repo-owned checks and contracts
- `jax.experimental.sparse`, `jax.experimental.ode`, `jax.experimental.shard_map`, and related sharding/lowering helpers: not part of the canonical runtime surface for this repo today
