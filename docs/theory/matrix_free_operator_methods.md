Last updated: 2026-03-25T16:20:00Z

# Matrix-Free Operator Methods

## Scope

This note describes the current matrix-free layer implemented in:

- [matrix_free_core.py](/src/arbplusjax/matrix_free_core.py)
- [jrb_mat.py](/src/arbplusjax/jrb_mat.py)
- [jcb_mat.py](/src/arbplusjax/jcb_mat.py)

The design target is a reusable operator-first surface for:

- dense operators
- dense symmetric / Hermitian operators
- sparse operators
- sparse symmetric / Hermitian operators

The point implementation remains the optimized execution substrate. `basic` matrix-free now exists as a separate shared semantic layer in [matrix_free_basic.py](/src/arbplusjax/matrix_free_basic.py), but its enclosure and validation semantics are still only partially completed.

## 1. Operator-First Model

The common abstraction is an operator

$$
v \mapsto A v
$$

with optional structural variants:

- right action: `rmatvec`, meaning `x^T A`
- adjoint action: `A^H x`
- structured aliases:
  - real symmetric / SPD
  - complex Hermitian / HPD

The shared runtime form is `OperatorPlan`, which packages:

- storage kind
- numeric payload
- orientation
- algebra

This separates operator preparation from operator application, which is the main mechanism for reducing JAX recompilation pressure on repeated runs.

## 2. Krylov Families

The matrix-free stack uses two Krylov families.

### 2.1 Lanczos

For real symmetric operators,

$$
A = A^\top,
$$

Lanczos builds an orthonormal basis `Q_k` and tridiagonal projection `T_k` such that

$$
A Q_k \approx Q_k T_k.
$$

This is used for:

- `funm(A) x`
- quadratic-form integrands `x^T f(A) x`
- SLQ log-determinant estimation
- structured SPD paths

Lanczos is the preferred route whenever symmetry is available because the projected dense problem is smaller and cheaper than a general Hessenberg reduction.

### 2.2 Arnoldi

For general or complex operators,

$$
A Q_k \approx Q_k H_k,
$$

with `H_k` upper Hessenberg. This supports:

- general matrix-function actions
- complex log-determinant estimation
- general nonsymmetric / non-Hermitian operator actions

When the operator is explicitly Hermitian, the implementation should prefer Hermitian-specialized frontends rather than the generic route.

## 3. Matrix Functions As Projected Dense Functions

Given a Krylov projection, the matrix-free action is approximated through a dense function of the reduced matrix:

$$
f(A) x \approx \beta Q_k f(T_k) e_1
$$

for Lanczos, and analogously with `H_k` for Arnoldi.

Dense projected kernels are used for:

- `exp`
- `log`
- `sqrt`
- `root`
- `sign`
- `sin`
- `cos`
- `sinh`
- `cosh`
- `tanh`
- integer powers

The current operator layer also includes contour-integral action helpers in the
shared core. These now back public real/complex matrix-free wrappers for:

- `log`
- `sqrt`
- `root`
- `sign`
- `sin`
- `cos`

This gives a unified story:

- the outer algorithm stays matrix-free
- only the projected matrix function is dense

## 4. Determinants And Log-Determinants

For structured positive problems, the determinant path is based on

$$
\log \det(A) = \operatorname{tr}(\log A).
$$

The implementation uses stochastic Lanczos quadrature in the real symmetric case and Arnoldi-style trace integrands in the complex case. Determinant wrappers are then computed as:

$$
\det(A) = \exp(\log \det(A)).
$$

Sparse SPD log-determinant also has a Leja plus Hutch++ path documented separately in [sparse_symmetric_leja_hutchpp_logdet.md](/docs/theory/sparse_symmetric_leja_hutchpp_logdet.md).

The current shared postprocessing layer also exposes:

- reusable SLQ quadrature metadata
- heat-trace evaluation from cached SLQ nodes and weights
- spectral-density histograms from cached SLQ nodes and weights
- Hutch++ metadata carrying low-rank and residual contributions together with
  probe-statistics summaries
- reusable low-rank deflation metadata carrying a compact basis, its operator
  image, and the exact low-rank trace contribution for residual estimators

## 5. Solve And Inverse Actions

Matrix-free inverse functionality is exposed as solve actions:

$$
A x = b,
\qquad
x = A^{-1} b.
$$

The point implementation uses iterative solvers:

- `cg` for symmetric / Hermitian paths
- `gmres` for general paths

The inverse-action API is intentionally just a solve-action alias with a different name because matrix-free inversion should not materialize `A^{-1}`.

## 6. AD Design

The matrix-free layer uses `custom_vjp` for callable operator paths in the Krylov action and trace-integrand machinery, and now uses `jax.lax.custom_linear_solve` on the operator-first solve surfaces. This avoids differentiating naively through every Krylov loop iteration.

There is an important JAX constraint:

- Python-callable operators work well as `nondiff_argnums`
- dynamic `OperatorPlan` payloads do not

So the current design splits these paths:

- callable-oriented `custom_vjp` kernels for operator callables
- plan-safe kernels for prepared `OperatorPlan` reuse under `jit` and `vmap`

This split is necessary to avoid tracer errors and to keep prepared operators reusable without forcing closure-based JIT wrappers.

For solve and Laplace-style correction paths the current contract is:

- keep the forward solve operator-first and matrix-free
- cache only compact operator metadata needed for the transpose solve
- use transpose or adjoint solves in the backward pass rather than replaying the primal Krylov history
- keep damping, jitter, and structural safety checks outside the differentiated linear-solve boundary
- when shell preconditioners are used, explicit transpose/adjoint callbacks can
  now be carried through the same shared preconditioner policy rather than
  assuming the forward callback is valid in every backward or transpose solve

The same design rule now applies to the estimator stack:

- cache projected SLQ or Hutch++ metadata, not full primal iteration histories
- keep restart, locking, correction-equation, and adaptive probe-budget policy
  outside the differentiated estimator boundary
- prefer projected-state reuse and implicit adjoints over replaying primal
  Lanczos/Arnoldi loops

## 7. What Is Optimized Today

The strongest current matrix-free optimizations are:

- prepared `OperatorPlan` reuse
- common dense / sparse operator application machinery
- explicit right-action and adjoint plans
- structured aliases for symmetric / Hermitian / SPD / HPD paths
- plan-safe JIT wrappers for repeated solve / inverse / logdet / det use
- shared SLQ/Hutch++ metadata preparation and postprocessing
- shared correction-expansion helpers for restarted Davidson/JD-style subspace
  updates
- orthogonal probe-block and adaptive probe-budget helpers

## 8. What Still Needs A Separate Basic Layer

Point matrix-free is now the execution substrate. The `basic` matrix-free layer still needs fuller semantics for:

- interval / box lifting
- enclosure inflation policy
- residual-based validation for solve / inverse actions
- uncertainty policy for stochastic estimators

That should sit above the point operator engine, not inside it.
