Last updated: 2026-03-25T16:20:00Z

# Matrix-Free Practical Guide

## Scope

This page is the practical companion to [matrix_free_operator_methods.md](/docs/theory/matrix_free_operator_methods.md). It explains how to use the current point-mode matrix-free surface effectively, where JAX bottlenecks show up, and how to avoid unnecessary recompilation.

## 1. Choose The Right Operator Surface

Prefer prepared operator plans when you will reuse the same operator more than once.

Use:

- dense: `*_dense_operator_plan_prepare`
- sparse BCOO: `*_bcoo_operator_plan_prepare`
- generic sparse API edge: `*_sparse_operator_plan_prepare`
- structured real: `*_symmetric_operator_plan_prepare`, `*_spd_operator_plan_prepare`
- structured complex: `*_hermitian_operator_plan_prepare`, `*_hpd_operator_plan_prepare`
- right action: `*_operator_rmatvec_plan_prepare`
- adjoint action: `*_operator_adjoint_plan_prepare`

Use raw callables when:

- the operator is genuinely dynamic each call
- you need the most flexible AD story through a Python closure

## 2. Recompilation Guidance

The main JAX compile trap in matrix-free code is changing Python-callable structure between runs.

Prefer:

```python
plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)
out = jrb_mat.jrb_mat_logdet_slq_point_jit(plan, probes, steps=16)
```

over:

```python
out = jax.jit(lambda ps: jrb_mat.jrb_mat_logdet_slq_point(op, ps, 16))(probes)
```

Reasons:

- the plan path keeps the operator payload in a stable pytree
- the dedicated `*_point_jit` wrappers avoid ad hoc closure churn
- the callable path can trigger extra recompiles if the closure shape changes

Keep these arguments stable across repeated calls:

- `steps`
- `terms`
- `restarts`
- `symmetric` / `hermitian`
- `tol`, `atol`, `maxiter` when using solve paths

## 3. AD Guidance

Current point-mode AD is strongest on:

- operator apply
- matrix-function actions
- trace / logdet estimators on callable operators
- iterative solve / inverse actions

Important caveat:

- callable-based `custom_vjp` and plan-based reuse are not identical in JAX

The repo now uses plan-safe JIT kernels for repeated `logdet` and `det` runs because `OperatorPlan` payloads cannot be treated like static nondifferentiable Python callables.

Practical rule:

- if you need repeated throughput, prepare a plan and use the dedicated `*_point_jit` entrypoint
- if you are experimenting with custom differentiation through a bespoke operator closure, start with the callable surface

## 4. Structured Path Advice

Always use the structured alias when the matrix really has that structure.

Use real structured aliases for:

- symmetric
- SPD

Use complex structured aliases for:

- Hermitian
- HPD

Benefits:

- better Krylov family choice
- cheaper dense projected kernels
- more stable log / sqrt / root behavior
- cheaper solve-action path via `cg` instead of `gmres`

Do not route an HPD problem through the general complex interface unless you are deliberately testing the general path.

## 5. Benchmark Findings

`python benchmarks/benchmark_matrix_free_krylov.py` on 2026-03-20 under CPU-only JAX reported:

- real dense action: plan `0.352s` vs callable `0.728s`
- real dense logdet: plan `0.412s` vs callable `0.506s`
- sparse real apply: plan `0.157s` vs callable `0.295s`
- sparse real solve: plan `0.259s` vs callable `0.423s`
- sparse real logdet: plan `0.544s` vs callable `0.638s`
- complex dense logdet: plan `0.527s` vs callable `0.738s`
- sparse complex logdet: plan `0.529s` vs callable `0.594s`

Interpretation:

- plan reuse helps most when the operator application itself is a meaningful part of the total cost
- sparse apply and sparse solve benefit strongly from avoiding repeated operator wrapping
- restarted actions and some iterative solves show smaller gains because solver iterations dominate
- complex AD and complex logdet remain materially more expensive than the real symmetric path

## 6. Likely Bottlenecks

The main current matrix-free cost centers are:

- Krylov basis construction
- repeated dense projected eigendecompositions
- stochastic logdet probe batching
- custom-VJP boundary overhead on callable paths
- GMRES paths on general complex operators
- inverse-diagonal correction paths for sparse real selected inverse estimation

The benchmark also shows expensive paths here:

- sparse real Leja plus Hutch++ logdet
- sparse real corrected inverse diagonal
- complex logdet gradient

Those are expected to remain heavier than plain operator apply or plain solve-action.

## 7. Recommended Usage Patterns

For repeated action evaluation:

- prepare an operator plan once
- keep `steps` and `restarts` fixed across calls
- use the structured alias where valid

For repeated logdet or determinant estimation:

- prefer the dedicated `*_logdet_slq_point_jit` and `*_det_slq_point_jit` wrappers
- reuse the same probe shape
- keep the same number of Lanczos / Arnoldi steps across the loop
- if you also need heat-trace or spectral-density summaries, prepare the SLQ
  metadata once and reuse it instead of recomputing separate probe reductions
- if you also need Hutch++ residual statistics or adaptive probe-count advice,
  reuse the returned metadata rather than recomputing pilot variance passes

For repeated contour-integral matrix-function actions:

- prefer the dedicated contour wrappers on `jrb_mat` / `jcb_mat` for `log`,
  `sqrt`, `root`, and `sign`
- keep `center`, `radius`, and `quadrature_order` fixed across repeated calls
  to avoid unnecessary recompilation and policy churn
- use the structured real/complex operator aliases whenever the spectrum really
  satisfies the symmetric / Hermitian assumptions used to pick the shifted
  solve path

For repeated solve-plus-logdet workloads:

- prefer the shared operator-first `*_logdet_solve_*` bundles rather than
  calling solve and logdet separately
- these surfaces now retain compact transpose-operator metadata so the implicit
  adjoint path does not need to replay the full primal iteration history
- keep solver choice, structure flags, and probe shapes fixed if you want
  stable cache reuse

For repeated solves:

- use `*_solve_action_point_jit` or `*_inverse_action_point_jit`
- set `symmetric=True` or `hermitian=True` whenever valid
- keep solver tolerances stable

## 8. When Not To Use Matrix-Free

Prefer direct dense methods when:

- the matrix is already small
- you need the full matrix inverse
- you need the full dense matrix function, not just an action
- exact dense eigensystems are the actual target

Matrix-free is the right tool when you want:

- repeated `A v`
- repeated `f(A) v`
- trace / logdet / determinant estimators
- solve / inverse actions
- sparse or implicit operator access

## 9. Current Practical Limits

The landed surface is broader than the original matrix-free tranche:

- SLQ metadata can now be reused for heat-trace and spectral-density summaries
- Hutch++ metadata now carries reusable low-rank and residual statistics
- low-rank deflation metadata can now be prepared once and reused across
  residual trace-estimation passes on the real and complex Jones surfaces
- cached rational Hutch++ metadata can now be prepared once and reused for
  rational trace and rational-logdet estimators on the real and complex Jones
  surfaces
- adaptive probe-budget helpers exist in shared `matrix_free_core`
- contour-integral `log`, `sqrt`, `root`, and `sign` action wrappers now exist
  on the real and complex Jones matrix-free surfaces
- contour-integral `sin` and `cos` action wrappers now also exist on the real
  and complex Jones matrix-free surfaces
- shell preconditioners can now carry explicit transpose/adjoint callbacks
  through the shared implicit-solve path instead of silently falling back to
  the forward callback

The remaining heavy work is still advanced-method hardening:

- flexible preconditioner policy beyond the current shared MINRES-centric path
- AD-safe cached rational-Krylov trace/logdet
- low-rank deflation and recycling
- fuller variance/stopping contracts around probe blocks
- broader public contour-integral matrix-function coverage
