Last updated: 2026-03-20T00:00:00Z

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
- shared `basic` semantic wrapper layer in [matrix_free_basic.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matrix_free_basic.py)
- dense operator plans
- sparse operator plans through the common sparse canonicalization path
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
- `logdet` and determinant wrappers on top of matrix-free estimators
- real structured aliases:
  - symmetric
  - SPD
- complex structured aliases:
  - Hermitian
  - HPD
- dedicated `basic` matrix-free tests in [test_matrix_free_basic.py](/home/phili/projects/arbplusJAX/tests/test_matrix_free_basic.py)
- point chassis coverage in:
  - [test_jrb_mat_chassis.py](/home/phili/projects/arbplusJAX/tests/test_jrb_mat_chassis.py)
  - [test_jcb_mat_chassis.py](/home/phili/projects/arbplusJAX/tests/test_jcb_mat_chassis.py)
- benchmark coverage in [benchmark_matrix_free_krylov.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_matrix_free_krylov.py)

### Partial

- `basic` matrix-free semantics exist, but they still need stronger enclosure and residual-validation policy for more families
- sparse operator-plan support exists for `COO`, `CSR`, and `BCOO` at the API edge, but internal plan canonicalization is still centered on the common sparse substrate rather than fully distinct format-specialized plans
- structured complex Hermitian/HPD paths exist, but some kernels still share more of the generic Arnoldi-style machinery than is ideal
- plan-native JIT coverage exists for the main repeated-use paths, but not every public matrix-free API has its own explicit plan-native JIT entrypoint yet
- benchmark coverage exists for core action/logdet/solve flows, but it is not yet exhaustive on compile-vs-execute and gradient-cost slices

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
  - `rmatvec`
  - adjoint apply
  - solve-action
  - inverse-action
  - `logdet`
  - `det`
  - named function actions
- expand reusable-plan support beyond current apply and determinant-heavy paths

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
- add compile-time vs execute-time reporting everywhere that repeated plan reuse is a core claim
- keep benchmark artifacts under:
  - `experiments/benchmarks/results/`
  - `experiments/benchmarks/outputs/`

## 8. Documentation

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
4. Expand sparse plan coverage at the plan layer.
5. Add AD tests for plan-based and structured paths.
6. Expand benchmarks with compile/recompile and gradient slices.
7. Finish the capability matrix and non-goal documentation.

