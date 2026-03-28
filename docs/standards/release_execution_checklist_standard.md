Last updated: 2026-03-27T00:00:00Z

# Release Execution Checklist Standard

This standard defines the minimum execution evidence required before calling a
change release-quality in this repository.

## Purpose

The checklist exists so release readiness is not inferred from ad hoc local
judgment. A release-quality change must have an explicit execution story across
correctness, startup/import behavior, examples, and benchmark surfaces.

## Required Checklist Categories

Every release-quality change must classify itself against these categories:

- source scope:
  - scalar
  - special-function
  - matrix dense
  - sparse/block/vblock
  - matrix-free/operator
  - curvature
  - docs/tooling/process only
- execution scope:
  - targeted owner tests
  - broader chassis or profile tests
  - benchmark or probe reruns where runtime behavior changed
  - example notebook execution where public teaching surfaces changed
  - generated report or registry refresh where checked-in artifacts changed

## Minimum Required Slices

At minimum, the checklist must name whether each of the following was required
and whether it passed:

- CPU owner-test slice
- broader parity or chassis slice when runtime behavior changed
- startup or first-use probe slice when loading behavior changed
- benchmark slice when compile/runtime policy or numerical performance changed
- example notebook execution when public API teaching surfaces changed
- generated artifact refresh and drift-check slice

## Startup And Compile Evidence

When a change touches startup, lazy loading, JIT ownership, or cached/prepared
surfaces, the checklist must explicitly record:

- import boundary tests run
- startup or first-use report refresh
- compile/probe evidence refreshed where applicable
- whether cold-start cost changed because of import loading or because of JAX
  backend/compile behavior

## Matrix And Matrix-Free Evidence

When a change touches dense, sparse, or matrix-free surfaces, the checklist must
explicitly record:

- prepared-plan or cached-plan coverage run
- implicit-adjoint or `custom_linear_solve` coverage run where solve AD changed
- sparse structured/operator-plan coverage run where sparse dispatch changed
- estimator or contour coverage run where matrix-free estimator families changed

## Special-Function Evidence

When a change touches special functions, the checklist must explicitly record:

- argument-direction AD coverage where relevant
- parameter-direction AD coverage where relevant
- hardening or provider benchmarks when numerical/provider behavior changed
- canonical notebook execution when public examples changed

## Generated Artifact Rule

If a checked-in report, registry, inventory, notebook, or index changes, the
release checklist must include:

- refresh command used
- drift-check command used
- confirmation that checked-in artifacts match the generator output

## Canonical Runbook

The concrete step-by-step checklist belongs in
[release_execution_checklist.md](/docs/implementation/release_execution_checklist.md).

This document is the policy surface. The implementation runbook defines the
actual command sequence.
