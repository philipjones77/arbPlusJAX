Last updated: 2026-03-20T00:00:00Z

# Matrix Stack

This page describes the intended division of responsibility across the matrix modules.

## Layers

- `arb_mat`
  Real dense interval matrices.
  Canonical dense real matrix layer for direct matrix kernels and precision modes.

- `acb_mat`
  Complex dense box matrices.
  Canonical dense complex matrix layer for direct matrix kernels and precision modes.

- `srb_mat`
  Real sparse matrices.
  Sparse-storage and sparse-kernel surface over `COO`, `CSR`, and `BCOO`.

- `scb_mat`
  Complex sparse matrices.
  Complex sparse-storage and sparse-kernel surface over `COO`, `CSR`, and `BCOO`.

- `jrb_mat`
  Real matrix-free operators.
  Operator-plan and Krylov action layer for large-scale real workflows.

- `jcb_mat`
  Complex matrix-free operators.
  Operator-plan and Krylov action layer for large-scale complex workflows.

## Contracts

Dense real:
- matrix: `(..., n, n, 2)`
- vector / rhs: `(..., n, 2)`
- helper entrypoints:
  - `arb_mat_as_matrix`
  - `arb_mat_as_vector`
  - `arb_mat_as_rhs`

Dense complex:
- matrix: `(..., n, n, 4)`
- vector / rhs: `(..., n, 4)`
- helper entrypoints:
  - `acb_mat_as_matrix`
  - `acb_mat_as_vector`
  - `acb_mat_as_rhs`

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

## Consistency Rules

- Dense and matrix-free layers should share the same value layouts within each algebra.
- `point` is the optimized execution engine.
- `basic` should be a semantic wrapper layer and must not perturb `point` hot paths.
- `rmatvec` and cached `rmatvec` are part of the expected surface for dense, sparse, and matrix-free layers.
- Matrix-free plans should use `OperatorPlan`; sparse cached plans should use `SparseMatvecPlan`.
- JAX diagnostics must remain opt-in so normal execution has no profiling overhead.

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
