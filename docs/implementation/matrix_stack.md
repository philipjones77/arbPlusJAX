Last updated: 2026-03-22T00:00:00Z

# Matrix Stack

This page describes the intended division of responsibility across the matrix modules.

## Current Status

The current matrix stack is centralized enough to be usable, but it is not uniformly "fully complete" across every layer.

- dense, sparse, block-sparse, and variable-block-sparse storage surfaces now have a common storage-to-operator bridge through thin `*_operator_plan_prepare(...)` wrappers
- reusable operator infrastructure is centralized in `matrix_free_core`
- public operator execution, solve, logdet/det, and eigensolver workflows are centralized in `jrb_mat` / `jcb_mat`
- dense, sparse, block-sparse, and variable-block-sparse point-mode chassis are functionally present and covered by targeted chassis and regression tests
- matrix-free point-mode public chassis is functionally present and covered by targeted chassis and regression tests
- matrix-free `basic` semantics, dense/sparse parameter-differentiable operator plans, and the first residual-history/deflation diagnostics tranche are now landed
- deeper eigensolver policy hardening beyond the new diagnostics metadata is still not fully complete

So the honest answer is:

- `common place`: mostly yes
- `fully functional` for point-mode public chassis: yes
- `fully complete` across all modes, AD, and advanced eigensolver maturity: no

## Closure Summary

- Dense point chassis: complete for the main public surface
- Sparse point chassis: complete for the main public surface
- Block-sparse and variable-block-sparse point chassis: complete for the main public surface they currently advertise
- Matrix-free point chassis: complete for the main public surface
- Common storage-to-operator placement: complete
- Shared operator substrate placement: complete
- Matrix-free `basic` semantics: main public wrappers complete for the current chassis
- Broader operator-parameter AD: dense, sparse, and shell/operator-plan parameter differentiation present for the main shared plan layer
- Advanced eigensolver hardening: diagnostics metadata tranche landed; deeper policy hardening still open
- Sparse interval/box storage modes: landed for sparse storage/public wrappers

Interpretation rule:

- If the question is "can callers use dense, sparse, block-sparse, vblock-sparse, and matrix-free point-mode functionality through a coherent shared placement?", the answer is yes.
- If the question is "is every related mode and advanced capability finished?", the answer is no.

## Layers

- `arb_mat`
  Real dense interval matrices.
  Canonical dense real matrix layer for direct matrix kernels and precision modes. It may adapt dense matrices into the operator layer when matrix-free execution is the right backend, but it is not the owner of matrix-free algorithms.

- `acb_mat`
  Complex dense box matrices.
  Canonical dense complex matrix layer for direct matrix kernels and precision modes. It may adapt dense matrices into the operator layer when matrix-free execution is the right backend, but it is not the owner of matrix-free algorithms.

- `srb_mat`
  Real sparse matrices.
  Sparse-storage and sparse-kernel surface over `COO`, `CSR`, and `BCOO`. It may adapt sparse storage into operator plans, but it is not the owner of matrix-free Krylov algorithms.

- `scb_mat`
  Complex sparse matrices.
  Complex sparse-storage and sparse-kernel surface over `COO`, `CSR`, and `BCOO`. It may adapt sparse storage into operator plans, but it is not the owner of matrix-free Krylov algorithms.

- `jrb_mat`
  Real matrix-free operators.
  Canonical real matrix-free execution layer. Owns operator plans, shell/preconditioner plans, finite-difference Jacobian-vector plans, and Krylov action/solve/eigensolver machinery for large-scale real workflows.

- `jcb_mat`
  Complex matrix-free operators.
  Canonical complex matrix-free execution layer. Owns operator plans, shell/preconditioner plans, finite-difference Jacobian-vector plans, and Krylov action/solve/eigensolver machinery for large-scale complex workflows.

## Adapter Layer

The matrix stack now makes the storage-to-operator bridge explicit.

- storage modules own representation-native kernels and caches:
  - `arb_mat`
  - `acb_mat`
  - `srb_mat`
  - `scb_mat`
  - `srb_block_mat`
  - `scb_block_mat`
  - `srb_vblock_mat`
  - `scb_vblock_mat`
- storage modules may expose thin `*_operator_plan_prepare(...)` wrappers so dense, sparse, block-sparse, and variable-block-sparse payloads can enter the Jones operator layer without duplicating Krylov policy
- `matrix_free_core` owns reusable operator-plan payload builders, orientation helpers, preconditioner-plan substrate, shifted-solve substrate, and shared restart/subspace utilities
- `jrb_mat` / `jcb_mat` own the public operator execution surface on top of those plans:
  - action/apply
  - solve and inverse action
  - `logdet` / `det`
  - multi-shift solves
  - restarted and block Krylov families
  - eigensolver families

This means dense and sparse modules should connect to matrix-free execution through adapters, but should not absorb matrix-free solver logic.

## Contracts

Dense real:
- matrix: `(..., n, n, 2)`
- vector / rhs: `(..., n, 2)`
- helper entrypoints:
  - `arb_mat_as_matrix`
  - `arb_mat_as_vector`
  - `arb_mat_as_rhs`
  - `arb_mat_operator_plan_prepare`
  - `arb_mat_operator_rmatvec_plan_prepare`
  - `arb_mat_operator_adjoint_plan_prepare`

Dense complex:
- matrix: `(..., n, n, 4)`
- vector / rhs: `(..., n, 4)`
- helper entrypoints:
  - `acb_mat_as_matrix`
  - `acb_mat_as_vector`
  - `acb_mat_as_rhs`
  - `acb_mat_operator_plan_prepare`
  - `acb_mat_operator_rmatvec_plan_prepare`
  - `acb_mat_operator_adjoint_plan_prepare`

Matrix-free real:
- matrix-like inputs use the same interval layout as `arb_mat`
- operator vectors use `(..., n, 2)`
- helper entrypoints:
  - `jrb_mat_as_interval_matrix`
  - `jrb_mat_as_interval_vector`

Matrix-free complex:
- matrix-like inputs use the same box layout as `acb_mat`
- operator vectors use `(..., n, 4)`
- helper entrypoints:
  - `jcb_mat_as_box_matrix`
  - `jcb_mat_as_box_vector`

Sparse real / complex:
- storage is structural, not boxed in the same way as dense
- canonical sparse families:
  - `SparseCOO`
  - `SparseCSR`
  - `SparseBCOO`
- dense-to-sparse bridges:
  - `srb_mat_from_dense_coo/csr/bcoo`
  - `scb_mat_from_dense_coo/csr/bcoo`
- storage-to-operator wrappers:
  - `srb_mat_operator_plan_prepare`
  - `srb_mat_operator_rmatvec_plan_prepare`
  - `srb_mat_operator_adjoint_plan_prepare`
  - `scb_mat_operator_plan_prepare`
  - `scb_mat_operator_rmatvec_plan_prepare`
  - `scb_mat_operator_adjoint_plan_prepare`
  - `srb_block_mat_operator_plan_prepare`
  - `srb_block_mat_operator_rmatvec_plan_prepare`
  - `srb_block_mat_operator_adjoint_plan_prepare`
  - `scb_block_mat_operator_plan_prepare`
  - `scb_block_mat_operator_rmatvec_plan_prepare`
  - `scb_block_mat_operator_adjoint_plan_prepare`
  - `srb_vblock_mat_operator_plan_prepare`
  - `srb_vblock_mat_operator_rmatvec_plan_prepare`
  - `srb_vblock_mat_operator_adjoint_plan_prepare`
  - `scb_vblock_mat_operator_plan_prepare`
  - `scb_vblock_mat_operator_rmatvec_plan_prepare`
  - `scb_vblock_mat_operator_adjoint_plan_prepare`

## Consistency Rules

- Dense and matrix-free layers should share the same value layouts within each algebra.
- Dense and sparse families adapt into the matrix-free layer; `jrb_mat` / `jcb_mat` are the owners of reusable matrix-free plan and Krylov infrastructure.
- storage modules may expose adapter wrappers, but Krylov/logdet/eigensolver policy stays out of storage modules and inside `jrb_mat` / `jcb_mat`.
- PETSc/SLEPc-inspired reusable substrate belongs in `matrix_free_core`; the governed implementation should remain in the Jones operator layer rather than optional external backends.
- `point` is the optimized execution engine.
- `basic` should be a semantic wrapper layer and must not perturb `point` hot paths.
- `rmatvec` and cached `rmatvec` are part of the expected surface for dense, sparse, and matrix-free layers.
- block and variable-block sparse families should follow the same rule when they advertise operator-style execution paths, and should adapt into `jrb_mat` / `jcb_mat` through block-native `matvec` / `rmatvec` callbacks instead of dense reconstruction where that is avoidable.
- Matrix-free plans should use `OperatorPlan`; sparse cached plans should use `SparseMatvecPlan`.
- Matrix-free reusable solver hints should use `PreconditionerPlan`, `ShiftedSolvePlan`, and recycled Krylov pytrees rather than ad hoc callable-only closures.
- JAX diagnostics must remain opt-in so normal execution has no profiling overhead.

## Execution Strategies

The matrix stack is expected to make the execution-strategy split explicit:

- `dense`: direct dense kernels for explicit matrix payloads
- `cached`: prepare/apply reuse paths for repeated shape-stable calls
- `matvec` and `rmatvec`: operator-style application paths
- `factorized`: decomposition-backed solve reuse paths
- `operator_plan`: matrix-free prepared operators for repeated Krylov-style work

Within `operator_plan`, the repeated-use substrate now includes:

- reusable preconditioner plans, including Jacobi-style diagonal inverses for dense and sparse operators
- shell operator and shell preconditioner plans for user-owned matrix-free callbacks
- finite-difference Jacobian-vector operator plans with explicit base-point update
- shared-operator multi-shift solve plans for rational and shifted solve workloads
- symmetric / Hermitian indefinite `minres` solve paths on the operator substrate
- restarted and block partial-spectrum `eigsh` variants built on the same operator substrate
- shared subspace/restart, Ritz extraction, and contour-filter helpers in `matrix_free_core`
- shared locked-first restart-basis selection for Jones eigensolver families, with retained restart windows larger than `k`
- first public Jones/operator eigensolver families for Krylov-Schur-style restart, Davidson/Jacobi-Davidson-style block subspace iteration, shift-invert, and contour filtering
- generalized Hermitian-definite Jones/operator eigensolver families, including generalized shift-invert spectral transforms, now sit on the same shared operator substrate
- first Jones/operator Hermitian polynomial and nonlinear eigenproblem point fronts now sit on the same shared operator substrate through Newton refinement on shift-invert kernels

SLEPc-style placement rule:

- spectral-transformation substrate such as shift-invert, Cayley-like, polynomial/rational transforms, correction equations, and contour quadrature belongs in [matrix_free_core.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matrix_free_core.py)
- public JAX-native operator eigensolver families belong in [jrb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jrb_mat.py) and [jcb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jcb_mat.py)
- Krylov-Schur, Davidson, Jacobi-Davidson, contour eigensolvers, and spectral-transform eigensolver families should be treated as Jones/operator-stack work, not dense/sparse-storage work and not external-backend substitutions
- PETSc/SLEPc remain optional benchmark-oracle layers only; they are not part of the governed runtime path
For JAX-first usage, repeated calls should prefer cached or prepared-plan entry points when they exist, and repeated batches should prefer padded or fixed-shape helpers so recompiles stay under control.

For matrix-free solve and eigensolver work in particular:
- prefer plan-backed preconditioners over rebuilding equivalent closures inside each call
- keep compile-time and steady-state execution cost separate in benchmarks
- keep structured Hermitian / symmetric routes on the operator substrate instead of introducing dense fallback helpers

For sparse real and complex matrices in particular:
- `matvec`, cached `matvec`, `rmatvec`, and cached `rmatvec` are part of the optimized core surface and should remain benchmarked directly.
- symmetric/Hermitian structural point paths should stay sparse for structure checks and structured factorizations instead of rebuilding dense midpoint matrices as an internal fallback.
- partial sparse spectral routines such as `eigsh` should delegate into `jrb_mat` / `jcb_mat` operator plans instead of owning separate storage-specific Krylov implementations.

## Current Shared Infrastructure

- shared dense/sparse helpers:
  - [mat_common.py](/home/phili/projects/arbplusJAX/src/arbplusjax/mat_common.py)
  - [sparse_common.py](/home/phili/projects/arbplusJAX/src/arbplusjax/sparse_common.py)
  - [sparse_core.py](/home/phili/projects/arbplusJAX/src/arbplusjax/sparse_core.py)

- shared matrix-free helpers:
  - [matrix_free_core.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matrix_free_core.py)
  - [matrix_free_basic.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matrix_free_basic.py)
  - [matfree_adjoints.py](/home/phili/projects/arbplusJAX/src/arbplusjax/matfree_adjoints.py)

- optional diagnostics:
  - [jax_diagnostics.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jax_diagnostics.py)

## Audit Scope

The repo-level consistency harness for this stack currently lives in:
- [test_matrix_stack_contracts.py](/home/phili/projects/arbplusJAX/tests/test_matrix_stack_contracts.py)
- [test_matrix_free_basic.py](/home/phili/projects/arbplusJAX/tests/test_matrix_free_basic.py)
- [test_jax_diagnostics.py](/home/phili/projects/arbplusJAX/tests/test_jax_diagnostics.py)

The matrix-oriented benchmark and diagnostics runners currently live in:
- [benchmark_dense_matrix_surface.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_dense_matrix_surface.py)
- [benchmark_sparse_matrix_surface.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_sparse_matrix_surface.py)
- [benchmark_matrix_free_krylov.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_matrix_free_krylov.py)
- [benchmark_matrix_stack_diagnostics.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_matrix_stack_diagnostics.py)

The report/workbook view for this stack currently lives in:
- [matrix_surface_workbook.md](/home/phili/projects/arbplusJAX/docs/reports/matrix_surface_workbook.md)
