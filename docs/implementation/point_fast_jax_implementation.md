Last updated: 2026-03-23T00:00:00Z

# Point Fast JAX Implementation

This note turns the repo-wide `point fast JAX` standard into an implementation
program for arbPlusJAX.

The priority is:

1. make `point` mode truly fast and JAX-native
2. keep the precise/adaptive layers as truth and fallback engines
3. prove the work tranche-by-tranche with category-owned tests

## Repo-Wide Architectural Split

Each family should separate into three layers:

- `*_precise`
  - Arb, mpmath, adaptive logic, variable precision, hard rescue paths
- `*_point_fast`
  - JAX-only arrays, fixed-shape kernels, vectorized reductions,
    approximants/recurrences/direct formulas
- dispatcher / diagnostics
  - decides whether `point_fast` is safe enough and whether precise fallback is
    needed

The immediate target is the middle layer.

For public API-facing families, the default compiled service entrypoint should
be `api.bind_point_batch_jit(...)` unless there is a narrower family-owned JIT
surface that is materially better.

## Common Fast-Point Infrastructure

The shared infrastructure that point-fast families should converge onto is:

- shared dtype and complex-promotion helpers
- log-domain helpers
- recurrence helpers built on `lax.scan` / `lax.fori_loop`
- Chebyshev and rational approximant evaluators
- JAX-safe region selectors
- explicit diagnostics outside the hot path

These should be shared instead of reimplemented family-by-family.

## Six Category Program

### 1. Core numeric scalars

Target:

- direct JAX formulas or stable elementary kernels
- no Python loops
- direct `jit` and `vmap` ownership tests

Proof surface:

- scalar chassis tests
- scalar API/service tests
- representative compiled batch proof via `bind_point_batch_jit`

### 2. Interval / box / precision modes

Target:

- point wrappers do not fall back through interval kernels just to reuse
  plumbing
- mode dispatch stays outside the point-fast hot kernel
- batch padding and dtype routing stay shape-stable

Proof surface:

- wrapper and precision-routing tests
- shape and dtype propagation tests
- representative compiled point-wrapper proof via `bind_point_batch_jit`

### 3. Dense matrix functionality

Target:

- point-mode dense kernels for cached `matvec`, `rmatvec`, direct apply, and
  structured helpers stay JAX-native
- no Python orchestration in hot matrix application paths

Proof surface:

- dense chassis tests
- dense plan/mode/structured tests
- representative compiled dense point-batch proof via `bind_point_batch_jit`

### 4. Sparse / block-sparse / vblock functionality

Target:

- point-mode sparse apply and cached apply paths remain shape-stable and JAX
  friendly
- format conversion and planning stay outside the hot apply kernel

Proof surface:

- sparse chassis tests
- sparse format/mode/structured tests
- representative compiled sparse point-batch proof via `bind_point_batch_jit`

### 5. Matrix-free / operator functionality

Target:

- operator application, cached actions, and selected point estimators use fixed
  shape JAX kernels
- shell/preconditioner/diagnostic selection remains outside the hot loop

Proof surface:

- `jrb_mat` / `jcb_mat` chassis tests
- matrix-free core/basic/logdet/adjoint tests
- representative compiled point-estimator proof on `*_point_jit` surfaces

### 6. Special functions

Target:

- point-mode special-function kernels use direct JAX formulas, fixed
  recurrences, or approximant-backed JAX paths
- difficult precise rescue logic stays outside the hot kernel

Proof surface:

- hypgeom/gamma/Bessel/incomplete-tail tests
- service-contract tests for special families
- representative compiled special-function point-batch proof via
  `bind_point_batch_jit`

## Conversion Order

Use this order for each family:

1. audit current point path
2. classify as `direct_fast`, `recurrence_fast`, `approx_fast`, or
   `precise_only_for_now`
3. refactor the point kernel into JAX-only array code
4. add `jit` and `vmap` tests
5. add safe-region parity against the precise path
6. only then widen box coverage or add approximant complexity

## What Must Change

Across all six categories, remove these from point-fast hot paths:

- Python loops over nodes or work items
- Python region switching on array values
- object-based parameter handling inside the kernel
- runtime shape changes
- Arb or mpmath objects
- host callbacks
- dynamic precision escalation
- fallback switching inside the compiled numeric kernel

Replace them with:

- vectorized array evaluation
- `lax.cond` / `lax.switch`
- `lax.scan` only where fixed sequential structure is necessary
- fixed-shape work arrays
- offline coefficient generation and runtime JAX array tables

## Acceptance Rule

A category tranche is not complete until its proof tests show:

- `jit` works on the point-fast path
- `vmap` works on the point-fast path
- the category hot path stays Python-free in the kernel
- the category precise fallback stays outside the hot path
- safe-box agreement with the precise path is within the documented tolerance
