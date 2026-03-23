Last updated: 2026-03-22T00:00:00Z

# SLEPc-Inspired JAX Status

## Purpose

This note records what parts of the SLEPc-style eigensolver model are now implemented natively in `arbplusjax`, what remains external-only, and what is still open.

The repo position is:

- SLEPc is an algorithm and interface reference
- SLEPc is not the governed runtime backend
- PETSc/SLEPc Python bindings remain optional benchmark-oracle tooling only

## Landed Native JAX Surface

Shared substrate in [matrix_free_core.py](/src/arbplusjax/matrix_free_core.py):

- operator-plan shell model
- preconditioner-plan shell model
- shift-invert and contour-style spectral-transform substrate
- shared restart and locking helpers
- correction-column prioritization for Davidson/Jacobi-Davidson expansion

Public eigensolver families in [jrb_mat.py](/src/arbplusjax/jrb_mat.py) and [jcb_mat.py](/src/arbplusjax/jcb_mat.py):

- Hermitian / symmetric `eigsh`
- restarted and block eigensolvers
- Krylov-Schur-style restart
- Davidson
- Jacobi-Davidson
- standard shift-invert
- generalized Hermitian-definite eigensolvers
- generalized shift-invert
- contour eigensolver front doors
- first Hermitian polynomial eigenproblem point fronts
- first Hermitian nonlinear eigenproblem point fronts

Current polynomial/nonlinear implementation style:

- Newton refinement on the smallest-magnitude shift-invert eigenpair
- automatic polynomial operator and derivative builders for the polynomial case
- diagnostics-bearing point surfaces in both real and complex Jones layers

## External-Only Boundary

The following remain external-only and are not part of the governed runtime:

- `petsc4py` object-model execution
- `slepc4py` object-model execution
- PETSc `KSP` as a library runtime dependency
- SLEPc `EPS` as a library runtime dependency

Current optional probing remains isolated to [benchmark_matrix_backend_candidates.py](/benchmarks/benchmark_matrix_backend_candidates.py).

That benchmark currently:

- detects `petsc4py` and `slepc4py` if installed
- times PETSc sparse `Mat.mult` and `KSP.solve`
- times SLEPc `EPS.solve`
- does not define the repo runtime contract

## Remaining Open Work

What is still open is solver-product depth, not missing baseline surface:

- richer restart-window truncation and wanted/unwanted partition policy
- stronger Davidson/Jacobi-Davidson correction-equation policy
- stronger preconditioned outer-loop behavior
- broader nonlinear eigensolver policy beyond the current Newton-on-shift-invert tranche
- broader polynomial eigenproblem policy beyond the current Hermitian point tranche

## Cross References

- status matrix: [matrix_free_completion_plan.md](/docs/status/matrix_free_completion_plan.md)
- placement rules: [matrix_stack.md](/docs/implementation/matrix_stack.md)
- real module surface: [jrb_mat.md](/docs/implementation/modules/jrb_mat.md)
- complex module surface: [jcb_mat.md](/docs/implementation/modules/jcb_mat.md)
