Last updated: 2026-03-27T00:00:00Z

# Matrix-Free Estimator Review (2026)

## Purpose

This note records the repo disposition for several recent matrix-free and sparse
estimator patterns that are relevant to arbPlusJAX.

The goal is not to accept every external idea as a public API immediately. The
goal is to record whether the idea:

- belongs in this repo
- belongs in the public surface or only the internal substrate
- should be implemented now, later, or only benchmarked
- should be governed by existing standards and diagnostics contracts

## Reviewed Techniques

### 1. Reusable Krylov Bases For Multi-Shift Batched Solves

Status:

- reviewed
- should be added
- belongs under the matrix-free operator/solve substrate, not as a point-only
  convenience trick

Why it belongs:

- the repo already has shared multi-shift solve substrate
- the repo already treats basis reuse and shift reuse as core matrix-free
  infrastructure
- this pattern matches the current operator-plan architecture and the remaining
  recycling / multi-shift backlog

Where it belongs:

- `src/arbplusjax/matrix_free_krylov.py`
- `src/arbplusjax/matrix_free_core.py`
- `src/arbplusjax/jrb_mat.py`
- `src/arbplusjax/jcb_mat.py`

Disposition:

- add as a matrix-free substrate feature
- expose through prepared multi-shift solve plans and repeated-use public
  surfaces
- design AD around `custom_linear_solve` and transpose-plan reuse

Not a first step:

- do not add a separate public top-level API family just for this
- do not couple it to Lineax adoption

### 2. Transpose-Correctness And Mixed-Precision Test Toolkit

Status:

- reviewed
- should be added
- belongs in tests and implementation guidance immediately

Why it belongs:

- this repo already depends on transpose correctness for implicit adjoints,
  `custom_linear_solve`, sparse preconditioners, and matrix-free solve AD
- dot tests, linearity tests, and JVP/VJP consistency checks are cheap and fit
  existing repo standards

Where it belongs:

- matrix-free regression tests
- curvature regression tests
- implementation guidance, not public API docs

Required checks:

- dot-product test for forward operator vs transpose/adjoint path
- linearity test for transpose solve and preconditioner transpose
- JVP/VJP identity smoke test around scalar objectives
- batched RHS parity for transpose solves

Disposition:

- add as a required testing pattern for operator-solve surfaces
- keep fp64 checks as the correctness reference even when kernels run in lower
  precision

### 3. GPU-Safe Accuracy Toolkit For Reductions And Krylov

Status:

- reviewed
- partially belongs
- should be treated as an internal numerical policy, not as a standalone public
  feature

What belongs:

- fp64 accumulation for sensitive reductions
- selective re-orthogonalization for Krylov bases
- stable reduction patterns for estimator statistics and orthogonalization

What does not need to be added immediately:

- heavyweight reproducible BLAS / superaccumulator machinery as a default path

Where it belongs:

- `src/arbplusjax/matrix_free_krylov.py`
- `src/arbplusjax/matrix_free_estimators.py`
- implementation notes and benchmarks

Disposition:

- adopt as numerical implementation guidance
- benchmark the cost of fp64 accumulation and re-orthogonalization
- do not expand the public API surface just to expose reduction schemes

### 4. Léja-Point Hutch++ Logdet

Status:

- reviewed
- should be documented as an experimental estimator candidate
- not ready to become the default logdet path yet

Why it belongs:

- the repo already has Leja and Hutch++ estimator surfaces
- this is directly on the matrix-free stochastic logdet roadmap

Why it should stay experimental first:

- current repo work still needs broader benchmark and diagnostics parity across
  SLQ, Hutch++, rational-Hutch++, contour, and sparse prepared plans
- this is an estimator policy extension, not missing substrate

Where it belongs:

- `src/arbplusjax/matrix_free_estimators.py`
- Jones real/complex wrappers
- matrix-free benchmark matrix

Disposition:

- keep on the estimator roadmap
- require benchmark and diagnostics comparison against current SLQ and
  Hutch++ paths before promoting it

### 5. Contour / Pole-Expansion Logdet Via PEXSI-Like Rational Evaluation

Status:

- reviewed
- conceptually belongs
- should be treated as a specialized contour/rational extension, not a core
  near-term replacement

Why it belongs:

- the repo already has contour and rational matrix-function surfaces
- the repo already has multi-shift solve and implicit-adjoint infrastructure

Why it is not a near-term default:

- it adds heavy approximation-policy surface area
- the repo does not currently have a selected-inversion backend comparable to
  PEXSI
- it is best treated as a higher-order contour/rational estimator family

Where it belongs:

- matrix-free contour/rational implementation notes
- future specialized sparse logdet experiments

Disposition:

- document as reviewed and relevant
- do not promote to the main public logdet recommendation yet

### 6. SLQ With Gauss-Radau / Kronrod-Like Error Bars

Status:

- reviewed
- should be added
- belongs in diagnostics and stopping policy

Why it belongs:

- the repo already has SLQ metadata, probe-statistics stopping helpers, and
  estimator diagnostics
- this is a natural upgrade to the current estimator stopping story

Where it belongs:

- `src/arbplusjax/matrix_free_estimators.py`
- `src/arbplusjax/matrix_free_krylov.py`
- matrix-free diagnostics and benchmark reports

Required outputs:

- nested-steps quadrature estimate
- truncation-gap error indicator
- stochastic standard error
- combined stopping criterion
- tail / Ritz safety diagnostics

Disposition:

- add as the preferred next SLQ stopping/diagnostics upgrade

## Repo Decision Summary

### Add

- reusable Krylov bases for multi-shift batched solves
- transpose-correctness and mixed-precision validation checks
- SLQ Gauss-Radau / Kronrod-style error bars and stopping diagnostics

### Add As Internal Numerical Policy

- GPU-safe accuracy toolkit for reductions and Krylov

### Keep Experimental / Benchmark First

- Léja-point Hutch++ logdet
- contour / pole-expansion logdet in the PEXSI style

## Placement Rules

These reviewed items belong in:

- matrix-free substrate
- sparse prepared-plan estimator paths
- curvature posterior-summary tooling
- implementation and theory notes

They do not belong as:

- point-only public APIs
- top-level convenience wrappers without operator-plan ownership
- undocumented one-off experimental helpers in `jrb_mat.py` / `jcb_mat.py`

## Standards Alignment

Any accepted follow-up implementation must obey existing repo standards:

- lazy loading and cold-path boundaries
- centralized JIT ownership
- startup/import/first-use measurement
- diagnostics-bearing public surfaces where estimators are stochastic or
  approximate
- explicit AD policy for solve and parameterized estimator surfaces
- benchmark artifacts and retained example evidence where a public claim is made

## Immediate Follow-On Work

The next concrete additions implied by this review are:

1. add transpose-correctness / JVP-VJP identity regression helpers for
   matrix-free solve surfaces
2. extend multi-shift substrate toward shared-basis reuse with explicit plan
   objects
3. add nested-SLQ truncation-gap diagnostics and stopping metadata
4. benchmark Léja-Hutch++ and contour/rational logdet variants only after the
   stopping/diagnostics story is stronger
