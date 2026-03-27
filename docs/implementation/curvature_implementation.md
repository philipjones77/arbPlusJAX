Last updated: 2026-03-25T00:00:00Z

# Curvature Implementation

## Purpose

This note records the intended curvature layer for arbPlusJAX and how it fits
the existing six-category runtime structure.

Curvature is not a separate public function category. It is a cross-cutting
helper layer that should live under:

- `src/arbplusjax/curvature/`

and support dense, sparse, matrix-free, and downstream inference workflows.

## What Curvature Means Here

In arbPlusJAX, curvature means:

- any first-class representation or approximation of the local second-order
  structure of an objective, likelihood, posterior, operator, or map
- together with the linear algebra needed to apply, solve, factorize,
  approximate, differentiate, and diagnose that structure

The layer should cover:

- exact Hessians
- Hessian-vector products
- generalized Gauss-Newton
- Fisher information
- prior-precision plus likelihood-curvature composition
- posterior-precision operators
- diagonal, block, low-rank, and low-rank-plus-diagonal approximations
- Lanczos-based spectral approximations
- inverse-diagonal and marginal-variance extraction
- curvature-aware preconditioners
- implicit adjoints through curvature solves

## Placement In The Runtime Tree

The architectural rule is:

- keep the public package API stable at the root
- keep the six public runtime categories:
  - `core_scalar`
  - `special`
  - `dense_matrix`
  - `sparse_matrix`
  - `matrix_free`
  - `transforms`
- place reusable curvature substrate under `arbplusjax/curvature/`

This means:

- dense/sparse/matrix-free modules may expose category-owned public surfaces
- reusable curvature operators, approximations, diagnostics, and implicit solve
  rules should not be buried in `jrb_mat`, `jcb_mat`, `arb_mat`, or `acb_mat`

## Recommended Module Layout

Recommended internal layout:

```text
arbplusjax/
  curvature/
    __init__.py
    base.py
    types.py
    configs.py
    hvp.py
    hessian.py
    ggn.py
    fisher.py
    composition.py
    posterior_precision.py
    approximations/
      diagonal.py
      block_diagonal.py
      low_rank.py
      low_rank_plus_diag.py
      lanczos.py
      nystrom.py
    inverse/
      inverse_diag.py
      selected_inverse.py
      marginals.py
    solvers/
      newton.py
      trust_region.py
      damping.py
      line_search.py
    preconditioners/
      diagonal.py
      block.py
      incomplete_factor.py
      low_rank.py
    diagnostics/
      definiteness.py
      conditioning.py
      spectral_checks.py
      dot_tests.py
    adjoints/
      implicit_solve.py
      transpose_solve.py
      custom_vjp.py
```

## Core Abstractions

### `CurvatureOperator`

The primary object should be operator-first and matrix-optional.

Required surface:

- `shape()`
- `dtype()`
- `matvec(v)`
- `rmatvec(v)`
- `to_dense()`
- `diagonal()`
- `trace()`
- `is_symmetric()`
- `is_psd()`

Useful extensions:

- `solve(b, ...)`
- `logdet(...)`
- `inverse_diagonal(...)`
- `block_diagonal(...)`

### `CurvatureSpec`

The layer should also expose a declarative config/spec describing:

- curvature kind
- representation
- differentiation mode
- damping / jitter
- symmetrization policy
- PSD-enforcement policy

## Must-Have Functional Families

Base constructors:

- `make_curvature_operator`
- `make_hvp_operator`
- `make_hessian_operator`
- `make_ggn_operator`
- `make_fisher_operator`
- `make_posterior_precision_operator`

Core actions:

- `matvec`
- `solve`
- `diagonal`
- `trace`
- `logdet`
- `inverse_diagonal`
- `to_dense`

Curvature builders:

- `hvp`
- `batched_hvp`
- `hessian_dense`
- `hessian_sparse`
- `ggn_matvec`
- `fisher_matvec`

Approximations:

- `diagonal_approximation`
- `block_diagonal_approximation`
- `low_rank_approximation`
- `low_rank_plus_diag_approximation`
- `lanczos_approximation`

Posterior summaries:

- `inverse_diagonal_estimate`
- `selected_inverse`
- `posterior_marginal_variances`
- `covariance_pushforward`

Step builders:

- `newton_step`
- `damped_newton_step`
- `trust_region_step`

Safety and stabilization:

- `symmetrize_operator`
- `ensure_psd`
- `add_jitter`
- `detect_negative_curvature`

Diagnostics:

- `estimate_extreme_eigenvalues`
- `estimate_condition_number`
- `dot_test_curvature`
- `curvature_regime_report`

## Integration With Current Matrix Work

Current integration target:

- keep dense, sparse, and matrix-free public matrix APIs in their current
  category-owned modules
- route reusable curvature substrate through the governed helper layer
- use `matrix_free_core` as the current operator-first bridge while the
  dedicated curvature package is introduced in tranches

That means near-term curvature work should align with:

- operator-first `logdet_solve`
- SLQ / Hutch++ / Lanczos approximations
- implicit-adjoint solve boundaries
- posterior-precision composition
- inverse-diagonal and selected-inverse estimation

## Development Order

### Phase 1

- `CurvatureOperator`
- `hvp`
- `make_hvp_operator`
- `make_posterior_precision_operator`
- `solve`
- `newton_step`
- `ensure_psd`

### Phase 2

- `ggn`
- `fisher`
- `inverse_diagonal_estimate`
- `selected_inverse`
- `logdet`

### Phase 3

- `low_rank_approximation`
- `lanczos_approximation`
- `posterior_marginal_variances`
- custom VJP/JVP support

### Phase 4

- trust-region and advanced damping
- regime detection
- automated approximation selection

## Downstream Priority

RandomFields77-style consumers are expected to rely primarily on:

- `make_posterior_precision_operator`
- `solve`
- `logdet`
- `inverse_diagonal`
- `selected_inverse`
- `newton_step`
- `ensure_psd`
- `estimate_condition_number`

That is why the curvature layer belongs in arbPlusJAX rather than being
re-implemented downstream.
