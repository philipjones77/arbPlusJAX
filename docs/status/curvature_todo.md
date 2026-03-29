Last updated: 2026-03-29T00:00:00Z

# Curvature TODO

This file tracks the dedicated curvature-layer backlog.

Current implementation note:
- [curvature_implementation.md](/docs/implementation/curvature_implementation.md)

Status legend:
- `done`: landed in code and covered at least by targeted tests
- `in_progress`: partially implemented or exposed, but still needs hardening
- `planned`: accepted roadmap item, not yet at implementation level

## Shared Helper Placement

Status: `in_progress`

- curvature is now being treated as a shared helper layer rather than being
  folded into one runtime category
- target placement is [curvature/](/src/arbplusjax/curvature/)
- root package and architecture docs now recognize curvature as a cross-cutting
  layer used by dense, sparse, and matrix-free stacks

## Phase 1 Core Operator Surface

Status: `in_progress`

- harden the newly introduced shared operator surface:
  - `CurvatureOperator`
  - `CurvatureSpec`
  - HVP builders
  - dense Hessian builders
  - posterior-precision composition
  - generic solve and Newton-step helpers
  - PSD/symmetrization helpers
- keep the implementation operator-first and matrix-optional
- minimize duplication by delegating to current dense and Jones
  matrix-free/sparse surfaces where they already exist
- explicit fast-JAX / operational-JAX contract coverage now exists for:
  - dense curvature `matvec` / `solve` / `logdet` / inverse-diagonal under
    `jit` and `vmap`
  - posterior-precision parameter AD through `damping` and `jitter`
  - variable- and parameter-direction AD on the HVP-facing curvature surface
  - matrix-free `jrb` curvature apply / solve / `logdet` under repeated `jit`
    reuse
- the dedicated runtime proof slice is
  [test_curvature_operational_contracts.py](/tests/test_curvature_operational_contracts.py)

## Phase 2 Bayesian And Approximation Surfaces

Status: `planned`

- generalized Gauss-Newton and Fisher operators
- inverse-diagonal estimation
- selected-inverse extraction
- operator-first `logdet` integration

## Phase 3 Spectral And Posterior Summaries

Status: `planned`

- low-rank curvature approximations
- Lanczos curvature approximations
- posterior marginal-variance extraction
- custom VJP/JVP support for scalable evidence terms

## Phase 4 Advanced Stabilization

Status: `planned`

- trust-region and advanced damping policy
- curvature regime detection
- automated approximation selection
- FWHT and QR probe-block parity checks for unbiased trace estimation on
  small reference problems
- low-rank-deflated estimator variance scans versus undeflated estimators at
  fixed probe budgets
- cached-deflation forward/VJP consistency checks across nearby parameter
  values
- eigen-interval and nugget sensitivity sweeps for logdet stability
- JIT/cache stability checks so value changes do not trigger recompiles for
  fixed operator shapes/configuration
- pytree contract checks so cached aux metadata survives `jit`, `vmap`, and
  repeated calls without hidden Python-side state

## Longer-Horizon Additions

Status: `planned`

- add contour-integral matrix functions plus reusable dense/operator-first
  `logm`, `sqrtm`, `rootm`, and `signm` infrastructure
- add broader operator-parameter adjoints beyond the now-landed
  parameter-differentiable operator-plan tranche
- keep PETSc/SLEPc as benchmark and design references only, not governed
  runtime backends
