Last updated: 2026-03-22T00:00:00Z

# Matrix-Free Completion Plan

## Scope

This plan covers the matrix-free/operator stack implemented in:

- [matrix_free_core.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matrix_free_core.py)
- [matrix_free_basic.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matrix_free_basic.py)
- [jrb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jrb_mat.py)
- [jcb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jcb_mat.py)

The goal is dense-style functional parity where matrix-free semantics make sense, while keeping point mode as the optimized execution engine and `basic` as a separate semantic layer.

## Status Summary

### Implemented

- shared operator-plan substrate in [matrix_free_core.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matrix_free_core.py)
  - includes reusable shell-orientation and parametric sparse-plan helpers used by the Jones wrappers
- shared `basic` semantic wrapper layer in [matrix_free_basic.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matrix_free_basic.py)
- dense operator plans
- shell operator plans
- finite-difference operator plans with reusable base-point updates
- sparse operator plans through the common sparse canonicalization path
- block-sparse and vblock-sparse operator plans
- thin storage-to-operator adapter wrappers now exist in the dense, sparse, block-sparse, and vblock sparse storage modules
- `matvec`, `rmatvec`, and adjoint operator application
- cached/operator-plan apply
- polynomial and `expm` actions
- named action wrappers currently surfaced:
  - `log`
  - `sqrt`
  - `root`
  - `sign`
  - selected trigonometric / hyperbolic wrappers
  - integer powers
- solve-action and inverse-action APIs
- operator-owned symmetric / Hermitian indefinite solve surface via `minres`
- reusable preconditioner-plan substrate:
  - identity
  - dense
  - diagonal
  - Jacobi-derived dense and sparse diagonal inverse plans
- shared multi-shift solve substrate and matrix-free front doors:
  - `jrb_mat_multi_shift_solve_*`
  - `jcb_mat_multi_shift_solve_*`
- unified `logdet_solve` point/basic result surfaces:
  - `jrb_mat_logdet_solve_*`
  - `jcb_mat_logdet_solve_*`
- `logdet` and determinant wrappers on top of matrix-free estimators
- real structured aliases:
  - symmetric
  - SPD
- complex structured aliases:
  - Hermitian
  - HPD
- operator-owned partial symmetric / Hermitian spectral solve surface:
  - `jrb_mat_eigsh_*`
  - `jcb_mat_eigsh_*`
  - restarted and block variants now exist on the same operator substrate
  - Davidson, Jacobi-Davidson, shift-invert, and contour-filter front doors now exist on the same operator substrate
  - diagnostics-bearing front doors now exist across the main eigensolver family
  - sparse `srb_mat` / `scb_mat` front doors now delegate into operator plans rather than rebuilding dense midpoint matrices
- parameter-differentiable sparse `BCOO` and dense operator-plan constructors now exist in `jrb_mat` / `jcb_mat`
- diagnostics-aware `basic` wrappers now exist for the main matrix-free solve/inverse/logdet families and the public log-action Krylov wrappers
- sparse interval/box storage wrappers now exist in `srb_mat` / `scb_mat`
- dedicated `basic` matrix-free tests in [test_matrix_free_basic.py](/home/phili/projects/arbplusJAX/tests/test_matrix_free_basic.py)
- point chassis coverage in:
  - [test_jrb_mat_chassis.py](/home/phili/projects/arbplusJAX/tests/test_jrb_mat_chassis.py)
  - [test_jcb_mat_chassis.py](/home/phili/projects/arbplusJAX/tests/test_jcb_mat_chassis.py)
- benchmark coverage in [benchmark_matrix_free_krylov.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_matrix_free_krylov.py)
  - includes compile-vs-execute slices for plan-backed solve, inverse, `minres`, multi-shift, restarted `eigsh`, and selected gradient paths

### Partial

- `basic` matrix-free semantics now include diagnostics-aware invalidation for the main solve/inverse/logdet families plus the public named Krylov action wrappers across the current Lanczos/Arnoldi surface, but broader family-by-family enclosure policy is still not exhaustive
- sparse operator-plan support exists for `COO`, `CSR`, and `BCOO` at the API edge, but internal plan canonicalization is still centered on the common sparse substrate rather than fully distinct format-specialized plans
- structured complex Hermitian/HPD paths now route the main projected action/integrand families through Hermitian Lanczos-backed kernels, but broader Hermitian-specialized coverage is still incomplete
- point-mode Jones public chassis is now complete enough to pass the dedicated `jrb_mat` and `jcb_mat` chassis suites from source-tree execution, including solve/inverse, `minres`, action/logdet SLQ, operator-plan `eigsh`, and plan-native JIT coverage for the main repeated-use paths
- shared midpoint Krylov solve scaffolding now lives in [matrix_free_core.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matrix_free_core.py), reducing duplication between [jrb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jrb_mat.py) and [jcb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jcb_mat.py)
- SLQ estimator handling now includes exact orthogonal-probe scaling for complete probe families while keeping probe gradients stable
- `minres` now exists on both the unpreconditioned and current shared preconditioned point-mode path for symmetric / Hermitian indefinite operator solves, including Jones plan/JIT entry points and `basic` wrappers
- benchmark coverage now includes compile-vs-execute and selected gradient-cost slices, but it is not yet exhaustive across sparse formats and all structured variants
- matrix-free eigensolvers now exist for partial spectra, including restarted and block variants, and diagnostics/report surfaces now exist across the family, but they are still early operator-subspace implementations:
  - residual-history and deflation-count metadata now exist across the main diagnostics surface
  - restarted, Davidson, and Jacobi-Davidson paths now preserve locked Ritz vectors more consistently across iterations
  - richer locking / restart / correction-equation policy still needs hardening
  - correction-equation quality and preconditioned outer-loop policy still need hardening
  - the newer Davidson, Jacobi-Davidson, shift-invert, and contour paths now pass the dedicated chassis suite but are not yet mature solver products

### Remaining

## 1. Basic Semantic Completion

- finish `basic` matrix-free as a genuinely separate semantic layer, not only a point wrapper with outward boxing
- add interval/box lifting policy for every exposed matrix-free family
- add enclosure inflation and uncertainty policy for stochastic estimators
- add residual-based validation for solve-action and inverse-action families
- document exact `basic` contracts separately from point contracts

Priority:
- operator apply
- named function actions
- solve-action / inverse-action
- `logdet` / `det`
- diagnostics and estimator metadata

## 2. Dense-Function Parity

- audit dense matrix functions in [arb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/arb_mat.py) and [acb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/acb_mat.py)
- add matrix-free action analogues where they are numerically appropriate
- explicitly mark dense-only or tiny-matrix-only functions where matrix-free parity is not the right target

Required matrix-free policy categories:
- exact operator action
- Krylov or projected approximation
- stochastic estimator
- dense-only / tiny-matrix-only non-goal

## 3. Structured Frontends

- route every eligible real symmetric / SPD function through the structured path explicitly
- route every eligible complex Hermitian / HPD function through the structured path explicitly
- tighten complex Hermitian/HPD kernels so they prefer Hermitian projected solvers instead of generic Arnoldi where possible
- make structured aliases visible as first-class entrypoints rather than relying on user convention
- refine the current preconditioned `minres` policy into an explicitly documented structure/preconditioner contract rather than a single shared midpoint path
- keep `eigsh` in the structured matrix-free layer:
  - symmetric real should use the Lanczos family
  - Hermitian complex should use Hermitian Lanczos rather than generic Arnoldi for partial spectra

## 4. Sparse Operator-Plan Coverage

- broaden sparse matrix-free plan coverage beyond the current common canonicalization path
- add more direct sparse operator-plan support for `COO` and `CSR` at the plan layer
- add sparse structured plan variants for:
  - symmetric sparse real
  - SPD sparse real
  - Hermitian sparse complex
  - HPD sparse complex
- keep plan reuse optimized without fragmenting the shared operator substrate

## 5. Cached And Plan-Native JIT Coverage

- make sure the following all have explicit plan-native JIT entrypoints:
  - named function actions not already covered by the current plan-native JIT tranche
  - newer eigensolver and structured helper front doors where repeated plan reuse is a core claim
- expand reusable-plan support beyond current apply and determinant-heavy paths
- audit complex plan-JIT wrappers so plan payloads stay dynamic and only true compile-shape/configuration arguments are static

## 6. AD And Gradient Coverage

- add gradient tests for plan-based:
  - `logdet`
  - `det`
  - solve-action
  - inverse-action
  - named function actions
- cover both real and complex paths
- add structured gradient tests for:
  - SPD
  - HPD
  - sparse SPD
  - sparse HPD
- document where AD is exact versus estimator-based versus approximate

## 7. Benchmarks

- expand [benchmark_matrix_free_krylov.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_matrix_free_krylov.py) with explicit slices for:
  - complex structured Hermitian / HPD
  - sparse `COO` vs `CSR` vs `BCOO`
  - callable vs plan reuse under repeated loops
  - gradient cost for solve/inverse/determinant APIs
- keep compile-time vs execute-time reporting everywhere that repeated plan reuse is a core claim
- keep benchmark artifacts under:
  - `benchmarks/results/`
  - `experiments/benchmarks/outputs/`

## 8. Preconditioners, Multi-Shift, And Recycling

- harden and broaden the current reusable preconditioner-plan abstraction in [matrix_free_core.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matrix_free_core.py)
- extend the current shared multi-shift solve substrate toward rational matrix functions and shifted trace/logdet estimators
- add recycled Krylov basis support for closely related solves and hypergradient-style repeated adjoints
- add block multi-RHS Krylov support as a standard path, not only as a special-case wrapper
- keep approximation policy separate from operator infrastructure:
  - node/pole placement belongs to rational-approximation policy
  - basis reuse, shift reuse, and preconditioner reuse belong to operator infrastructure
- structured preference:
  - real symmetric / SPD should prefer block CG, multi-shift CG, and recycled Lanczos
  - complex Hermitian / HPD should prefer Hermitian-specialized block/multi-shift solvers where possible
  - general nonsymmetric paths should use shared-basis GMRES / Arnoldi-style recycling
- design AD around plan objects and implicit adjoint solves, not one-off callable closures

## PETSc Reference Points

PETSc is a design reference for matrix-free operator infrastructure, not a runtime backend for the JAX path.

Useful PETSc concepts to mirror:

- shell matrix model
  - `MatCreateShell()` / `MATSHELL` for user-owned operator payloads and dimensions
  - `MatShellSetOperation()` for attaching `matvec`-style operations
  - `MatShellSetMatProductOperation()` for optional matrix-product hooks
- shell preconditioner model
  - `PCSHELL`
  - `PCShellSetApply()` and related shell hooks for reusable custom preconditioners
- finite-difference Jacobian operator model
  - `MatCreateMFFD()`
  - `MatCreateSNESMF()`
  - `MatMFFDSetBase()` and related `MATMFFD` controls for base-point reuse

Implications for `arbplusjax`:

- our `OperatorPlan` is conceptually closest to `MATSHELL`
- our future preconditioner-plan abstraction should follow the same separation PETSc uses between operator shell and preconditioner shell
- multi-shift, recycled Krylov, and structured solver policies belong in the operator infrastructure, not in storage-specific sparse wrappers
- finite-difference Jacobian-vector products are a possible future matrix-free family, but they should be added as an explicit JAX-native operator-plan type rather than by wrapping PETSc `MATMFFD`

## SLEPc-Inspired JAX Feature Matrix

The repo should continue treating SLEPc as an algorithm and interface reference, not as the governed runtime backend.

Current external-boundary rule:

- `petsc4py` / `slepc4py` remain optional benchmark-oracle tooling only
- no governed runtime path in `src/arbplusjax` should depend on PETSc or SLEPc object models
- benchmark probing for PETSc/SLEPc remains isolated in [benchmark_matrix_backend_candidates.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_matrix_backend_candidates.py)

Implemented or now clearly owned in JAX:

- shell/operator abstraction
  - owner: [matrix_free_core.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matrix_free_core.py)
  - status: landed via `OperatorPlan`, shell plans, sparse plans, and preconditioner plans
- spectral-transformation substrate
  - owner: [matrix_free_core.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matrix_free_core.py)
  - status: landed for shift-invert, contour quadrature/filtering, restarted subspace utilities, and shared shifted solves
- public partial-spectrum eigensolver families
  - owner: [jrb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jrb_mat.py), [jcb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jcb_mat.py)
  - status: landed for Lanczos/Arnoldi, restarted, block, Krylov-Schur-style, Davidson, Jacobi-Davidson, generalized Hermitian-definite problems, standard and generalized shift-invert, contour front doors, and first Hermitian polynomial/nonlinear point fronts
- convergence diagnostics
  - owner: [matrix_free_core.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matrix_free_core.py), [jrb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jrb_mat.py), [jcb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jcb_mat.py)
  - status: landed for residual-history, convergence flags, locked count, and deflation count
- restart and locking policy
  - owner: [matrix_free_core.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matrix_free_core.py) for shared policy, [jrb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jrb_mat.py) and [jcb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jcb_mat.py) for public eigensolver use
  - status: landed broader native tranche with locked-first restart selection, retained restart windows larger than `k`, refill from the current subspace rather than fresh random seeds, and shared correction-column prioritization for Davidson/Jacobi-Davidson expansion

Still to deepen:

- richer restart-window truncation and wanted/unwanted partition policy
  - owner: [matrix_free_core.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matrix_free_core.py)
- stronger Davidson/Jacobi-Davidson correction-equation policy and preconditioned outer-loop strategy
  - owner: [jrb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jrb_mat.py), [jcb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jcb_mat.py) with reusable helpers promoted into [matrix_free_core.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matrix_free_core.py) when stable
- generalized, polynomial, and nonlinear eigensolver abstractions
  - owner: [matrix_free_core.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matrix_free_core.py) for substrate, [jrb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jrb_mat.py) / [jcb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jcb_mat.py) for public surfaces
  - status: generalized Hermitian-definite, generalized shift-invert, and first native Hermitian polynomial/nonlinear point surfaces are landed; broader nonlinear solver-product depth remains open beyond the current Newton-on-shift-invert tranche
- mature solver-product behavior
  - owner: [jrb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jrb_mat.py), [jcb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jcb_mat.py)
  - status: still open for more sophisticated restart selection, convergence safeguards, and stronger preconditioned correction solves

## 9. Documentation

- add a practical `basic` matrix-free guide once the semantic layer is fully specified
- add a matrix-free capability/status table covering:
  - dense real
  - dense complex
  - sparse real
  - sparse complex
  - symmetric / SPD
  - Hermitian / HPD
  - point / basic
- document which functions are:
  - exact actions
  - stochastic estimators
  - approximations
- document explicit non-goals so the repo does not imply false parity for functions that should remain dense-only

## Recommended Execution Order

1. Finish `basic` semantics for existing point families.
2. Complete plan-native JIT coverage for the current public matrix-free surface.
3. Tighten structured Hermitian/HPD specialization.
4. Add reusable preconditioner, multi-shift, and recycled-Krylov infrastructure.
5. Expand sparse plan coverage at the plan layer.
6. Add AD tests for plan-based and structured paths.
7. Expand benchmarks with compile/recompile and gradient slices.
8. Finish the capability matrix and non-goal documentation.
