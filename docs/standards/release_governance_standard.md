Last updated: 2026-03-29T00:00:00Z

# Release Governance Standard

Status: active

## Purpose

This document defines the non-code governance surfaces required for stable
public releases.

## Required Governance Surfaces

The repo should maintain:

- changelog or release-notes policy
- deprecation/removal policy
- API stability levels
- canonical support matrix
- migration-guide pattern for breaking changes

These should be documented as governed repo surfaces rather than implied by
maintainer habit.

## Stability Levels

The repo should distinguish at least:

- `stable`
- `experimental`
- `benchmark-first`
- `debug-only`

These levels should be visible in public metadata, status docs, or generated
reports where practical.

## Changelog Rule

The repo should keep a governed release-history surface, typically
`CHANGELOG.md`, with predictable release-note structure.

## Migration Rule

Breaking or materially changed public behavior should have a governed migration
pattern under docs or release notes rather than only appearing in commit
messages.

## Release-Quality Execution Checklist

Release readiness must not be inferred from ad hoc local judgment.

Every release-quality change should classify itself against these categories:

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
  - generated artifact refresh where checked-in artifacts changed

At minimum, the checklist should record whether each of the following was
required and whether it passed:

- CPU owner-test slice
- broader parity or chassis slice when runtime behavior changed
- startup or first-use probe slice when loading behavior changed
- benchmark slice when compile/runtime policy or numerical performance changed
- example notebook execution when public API teaching surfaces changed
- generated artifact refresh and drift-check slice

### Startup and compile evidence

When a change touches startup, lazy loading, JIT ownership, or cached/prepared
surfaces, the checklist must explicitly record:

- import boundary tests run
- startup or first-use report refresh
- compile/probe evidence refreshed where applicable
- whether cold-start cost changed because of import loading or because of JAX
  backend/compile behavior

### Matrix and matrix-free evidence

When a change touches dense, sparse, or matrix-free surfaces, the checklist must
explicitly record:

- prepared-plan or cached-plan coverage run
- implicit-adjoint or `custom_linear_solve` coverage run where solve AD changed
- sparse structured/operator-plan coverage run where sparse dispatch changed
- estimator or contour coverage run where matrix-free estimator families changed

### Special-function evidence

When a change touches special functions, the checklist must explicitly record:

- argument-direction AD coverage where relevant
- parameter-direction AD coverage where relevant
- hardening or provider benchmarks when numerical/provider behavior changed
- canonical notebook execution when public examples changed

### Generated artifact rule

If a checked-in report, registry, inventory, notebook, or index changes, the
release checklist must include:

- refresh command used
- drift-check command used
- confirmation that checked-in artifacts match the generator output

The concrete runbook remains in:

- [release_execution_checklist.md](/docs/implementation/release_execution_checklist.md)

## Required Evidence

The repo is compliant only if:

- the governed file structure exists
- the release-governance policy is documented
- changelog, support, and security entrypoints exist
- the release-quality execution checklist policy is documented
