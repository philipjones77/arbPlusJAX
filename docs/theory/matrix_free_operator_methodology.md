Last updated: 2026-03-23T00:00:00Z

# Matrix Free Operator Methodology

## Purpose

This note records the current production methodology for the matrix-free and
operator-plan stack in arbPlusJAX.

## Scope

This note covers:

- operator-plan preparation and reuse
- matrix-function actions
- solve, inverse-action, determinant, and log-determinant paths
- AD-safe and JIT-safe calling expectations
- diagnostics and metadata expectations for matrix-free routing

## Operator-First Interpretation

The matrix-free stack is built around the idea that the linear map is primary
and explicit materialization is optional.

The public contract is therefore:

- callers may provide operator semantics without forming a dense matrix
- prepared operator plans are first-class runtime objects
- repeated application, solve, and trace-style workflows should reuse those
  plans instead of rebuilding structure inside every call

## Prepare Once, Reuse Often

Production matrix-free usage should prefer:

1. operator-plan preparation outside the service loop
2. stable dtype and structural controls
3. repeated apply/solve/logdet calls through the prepared plan
4. optional adjoint or preconditioner preparation as separate reusable assets

This is the main practical rule for minimizing recompilation and setup churn.

## Krylov Interpretation

Current matrix-free matrix-function and solve layers use Krylov projections as
the core execution strategy.

Production interpretation:

- symmetry or Hermitian structure should be surfaced explicitly when known
- restarted or structured variants are execution strategies, not separate
  mathematical APIs
- cached projected-state metadata is part of the orchestration layer, not part
  of the mandatory numeric kernel path

## AD Interpretation

The matrix-free production contract should be read conservatively:

- AD support should operate on the real public operator-facing surface
- plan reuse and AD should coexist without hidden Python state
- diagnostics and auxiliary projected metadata must remain pytree-safe when they
  are returned through a JAX-facing path

Examples, tests, and benchmarks should therefore validate:

- primal reuse behavior
- gradient behavior on scalarized objectives
- no avoidable recompilation from plan rebuilding

## Diagnostics Interpretation

Matrix-free diagnostics should expose enough information for routing and
debugging:

- operator kind
- structural assumptions
- convergence or termination state
- cache/plan reuse status
- projected-dimension or probe metadata where relevant

Those diagnostics should remain opt-in and should not slow the mandatory
numeric path when disabled.

## Relation To Dense And Sparse Surfaces

Matrix-free is not a replacement for dense or sparse matrix families.

Production interpretation:

- use dense when materialization and direct kernels are appropriate
- use sparse when structural sparsity is the main advantage
- use matrix-free when operator application, Krylov workflows, or large implicit
  systems are primary

The notebooks and workbook should make this comparison explicit rather than
presenting matrix-free as a hidden backend detail.

## Current Limits

- contour-integral matrix functions are still a backlog item
- broader rational/polynomial spectral transforms still need hardening
- deflation, recycling, and variance-control layers are only partly complete
- GPU validation remains a later tranche

For algorithm-specific detail on the current operator-first implementation, see
[matrix_free_operator_methods.md](/docs/theory/matrix_free_operator_methods.md).
