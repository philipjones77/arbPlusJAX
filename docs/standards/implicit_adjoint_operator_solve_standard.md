# Implicit Adjoint Operator Solve Standard

This standard defines the production contract for differentiable matrix-free solves that are exposed through arbPlusJAX.

## Purpose

Matrix-free solve surfaces must not differentiate through solver iterations or adaptive restart logic when an implicit-adjoint formulation is available. The canonical JAX boundary for these solves is `jax.lax.custom_linear_solve`.

## Required Pattern

- expose operator-first solve surfaces rather than dense-only fallback APIs
- define the primal solve in terms of `A(v)` or an `OperatorPlan`
- define a transpose or adjoint solve policy explicitly
- use `custom_linear_solve` when the surface claims implicit-adjoint gradient support
- keep cached solve metadata small, pytree-safe, and reusable across logdet/Laplace corrections

## Operator Metadata

Implicit-adjoint solve surfaces should retain compact cached metadata that is sufficient to reconstruct the backward solve without replaying Python control flow:

- forward operator plan
- transpose or adjoint operator plan when needed
- preconditioner plan and transpose preconditioner plan when needed
- solver family
- structural tag such as `symmetric`, `spd`, `hermitian`, or `hpd`

This metadata belongs in runtime-returned auxiliary objects, not in ambient hidden caches.

## Safety Requirements

- SPD / Hermitian assumptions must be explicit in the surface contract
- apply damping, jitter, or nugget terms outside the implicit solve boundary
- do not differentiate through restart policy, rank selection, probe-budget adaptation, or similar heuristics
- if a non-symmetric surface lacks a trustworthy transpose solve, it must not claim implicit-adjoint support

## Diagnostics

Diagnostics should distinguish:

- primal convergence state
- whether implicit adjoints were used
- whether gradients are supported under the current operator/transpose policy

Diagnostics and logging must remain outside the mandatory numeric hot path.

## Related Standards

- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)
- [jax_surface_policy_standard.md](/docs/standards/jax_surface_policy_standard.md)
- [caching_recompilation_standard.md](/docs/standards/caching_recompilation_standard.md)
