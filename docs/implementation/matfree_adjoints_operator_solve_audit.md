## Matrix-Free Adjoint Audit

This note records which `matfree_adjoints.py` paths should remain estimator-style
`custom_vjp` surfaces and which should migrate to true operator-solve surfaces.

### Keep as estimator-specific `custom_vjp`

- Lanczos and Arnoldi decomposition adjoints.
  These are decomposition/backpropagation kernels, not solve surfaces.
- Hutchinson-style trace estimators and related sample-coupled estimators.
  These depend on estimator-state reuse and sample coupling rather than a linear
  solve contract.
- Integrand-style logdet/trace estimator helpers where the backward pass reuses
  projected state instead of solving a transpose system.

### Prefer true operator-solve surfaces

- Matrix-free solve and inverse-action APIs.
  These should use `jax.lax.custom_linear_solve` through the operator-plan
  substrate instead of tracing iterative steps or relying on one-off
  `custom_vjp`.
- Prepared sparse and structured preconditioned solve paths.
  Once a surface has explicit operator, transpose-operator, preconditioner, and
  transpose-preconditioner ownership, it belongs on the implicit-adjoint solve
  path.
- Shifted solve bundles where a transpose/adjoint shifted operator can be
  prepared explicitly.

### Current repo decision

- `matrix_free_adjoint.py` is the canonical implicit-adjoint solve layer.
- `matfree_adjoints.py` remains the canonical efficient-adjoint estimator and
  decomposition layer.
- Do not migrate estimator-specific `custom_vjp` paths into
  `custom_linear_solve` unless the public surface is truly a linear solve with a
  well-defined transpose solve contract.

### Lineax evaluation

- Lineax is conceptually aligned with the repo's operator-first solve direction.
- Do not adopt Lineax as a blanket internal backend right now.
- Consider it only for solver families where it would remove local ownership
  code rather than add another abstraction boundary.
- The current repo-specific operator, preconditioner, interval/box lifting, and
  diagnostics contracts are still richer than a straight solver backend swap.
