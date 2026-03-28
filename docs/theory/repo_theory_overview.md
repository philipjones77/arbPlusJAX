Last updated: 2026-03-28T00:00:00Z

# Repo Theory Overview

## Purpose

This is the general theory note for arbPlusJAX.

Its job is to explain the mathematical and numerical philosophy of the repo as
a whole:

- what kinds of objects the repo works with
- why the public surface is split into `point`, `basic`, `adaptive`, and
  `rigorous`
- why dense, sparse, matrix-free, transform, special-function, and curvature
  layers are all present
- why the repo prefers JAX-first kernels together with explicit diagnostics,
  metadata, and repeated-call runtime policy

This note is not the place for family-specific derivations. Those belong in the
other methodology notes in `docs/theory/`.

## The Core Problem The Repo Solves

arbPlusJAX is a JAX-first numerical library that tries to combine four things
that are often separated in other systems:

1. direct fast numeric evaluation
2. baseline enclosure or uncertainty-aware evaluation
3. structured linear algebra over dense, sparse, and operator-backed objects
4. automatic differentiation through those same public surfaces

The central tension is that these goals do not naturally lead to one single
implementation style.

For example:

- high-throughput repeated evaluation wants fixed-shape compiled kernels
- enclosure-oriented evaluation wants explicit widening or conservative bounds
- sparse and matrix-free work wants prepared plans and iterative structure
- AD wants continuous parameters to remain dynamic while structural choices
  remain explicit

The repo therefore treats these as related but non-identical layers instead of
pretending that one implementation path solves every problem equally well.

## The Main Mathematical Object Classes

The repo works with six top-level runtime object classes:

1. scalar and elementary functions
2. interval and box enclosures
3. dense matrices
4. sparse, block-sparse, and variable-block sparse matrices
5. matrix-free operators
6. transforms and special-function families

Curvature is treated as a shared helper layer over these classes rather than as
a seventh unrelated category.

Mathematically, the repo moves between:

- scalar functions `f(x)` and parameterized families `f(x; theta)`
- enclosure objects such as intervals and complex boxes
- concrete matrices `A`
- operator plans `O_A`
- solve, inverse-action, trace, logdet, and posterior-summary functionals built
  from those matrices or operators

This is why a unified notation bridge exists in:

- [notation.md](/docs/notation/notation.md)

## Why The Mode Split Exists

The repo does not use one universal evaluation mode because the mathematical
goal changes between workloads.

### `point`

`point` is the direct repeated-evaluation surface.

Its role is:

- fast JAX execution
- stable-shape compiled batches
- ordinary service-style evaluation
- direct AD through the main public numerical path

Mathematically, `point` is the performance-first approximation or direct
floating-point evaluation surface.

It is not required to carry enclosure semantics.

### `basic`

`basic` is the baseline enclosure or uncertainty-aware surface.

Its role is:

- provide the first public containment or widening story
- expose baseline uncertainty-aware matrix or matrix-free semantics
- remain much cheaper than the strongest rigorous or analytic path

Mathematically, `basic` is the first conservative layer above `point`.

In scalar families it often means midpoint-style evaluation plus outward
rounding or widening. In matrix and matrix-free workloads it often means
diagnostics-aware inflation or a baseline interval interpretation rather than a
full exact enclosure calculus.

### `adaptive`

`adaptive` exists where the numerical regime can justify spending more work to
improve a baseline enclosure or estimate.

Its role is:

- increase work where the local regime is difficult
- use fixed but richer stencils, refinements, or strategy changes
- improve practical quality without claiming that every family is already at a
  strongest rigorous path

### `rigorous`

`rigorous` is the strongest currently exposed public containment or proof-like
path for a family.

It exists because some families can support stronger analytic or structure-aware
bounds than `basic`, while others cannot yet do so cheaply.

This split keeps the public API honest:

- `point` is not asked to be an enclosure path
- `basic` is not forced to claim the strongest rigorous status
- `rigorous` is reserved for the strongest currently supported public path

## Why JAX-First Rather Than Precision-First

The repo is JAX-first because repeated compiled numerical evaluation is the
default production target.

That has several consequences:

- fixed-shape kernels matter
- stable batching matters
- lazy loading matters
- prepared reuse surfaces matter
- AD through public kernels matters

This is different from a precision-first system where arbitrary precision and
fully rigorous ball arithmetic are the default internal substrate.

arbPlusJAX instead uses:

- JAX-native arrays and kernels as the main execution model
- explicit surface kinds for direct, bound, compiled, diagnostics-bearing, and
  prepared-plan calls
- diagnostics and metadata to explain strategy choice
- widening and regime logic as explicit public layers rather than hidden
  folklore

That is why the repo keeps separate standards for:

- fast-JAX structure
- backend-realized performance
- API surface kinds
- API usability

## Why Prepared Plans And Bound Surfaces Matter

Repeated numerical work is not only about mathematical formulas. It is also
about reuse boundaries.

The repo therefore treats these as mathematically meaningful public concepts:

- bind once and reuse for repeated point batches
- prepare once and apply repeatedly for matrix and operator workloads
- separate numeric values from structural preparation when the structure is the
  expensive reusable object

This matters because the same mathematical object may admit very different
runtime organizations:

- dense direct apply
- sparse prepared solve
- matrix-free operator plan
- iterative solve with implicit adjoint
- cached trace/logdet approximation state

Those are not only implementation details. They change what repeated numerical
work looks like in practice, so the theory layer has to acknowledge them.

## Why Diagnostics And Metadata Are First-Class

The repo treats diagnostics and metadata as part of the mathematical
interpretation of a public surface, not as mere debugging leftovers.

This is because many important repo surfaces are strategy-dependent:

- special-function regime selection
- sparse policy selection
- matrix-free Krylov convergence and uncertainty
- curvature approximations and posterior summaries

Without diagnostics, the user sees only a value and cannot tell:

- which strategy was used
- how conservative the widening was
- what uncertainty estimate drove the `basic` result
- whether a sparse path stayed sparse-native
- whether an iterative estimate is converged, heuristic, or widened

So the repo treats `D_f` and related metadata as part of the theory-to-runtime
bridge, not just as internal logging.

## Why AD Is Split Into Two Directions

For parameterized families, the repo distinguishes:

- argument-direction AD
- parameter-direction AD

This is a mathematical distinction, not only a test distinction.

For a parameterized family `f(x; theta)`:

- differentiating with respect to `x` studies the local response in the
  evaluation variable
- differentiating with respect to `theta` studies the response of the family
  itself

These can fail for different reasons:

- branch or regime logic may be continuous in `x` but not in `theta`
- a recurrence may be stable in one direction and not the other
- a solve or estimator helper may expose one direction cleanly but not the
  other

That is why the repo audits both directions explicitly instead of treating
“supports grad” as one undifferentiated claim.

## Why Dense, Sparse, And Matrix-Free All Exist

The repo is not only a scalar special-function library. It also targets
structured linear algebra and second-order workflows.

These three matrix regimes correspond to genuinely different mathematical
settings:

### Dense

Dense paths are appropriate when the matrix is explicit and moderate enough that
materialized linear algebra is still the right object.

### Sparse

Sparse paths are appropriate when the matrix has explicit sparsity structure
that should be preserved in storage, solves, and selected summaries.

### Matrix-free

Matrix-free paths are appropriate when the operator is known only through
application, or when a solve/trace/logdet computation should be driven by
operator actions rather than by explicit storage.

This split matters mathematically because the approximation and evidence model
changes:

- dense paths can rely on direct factorizations more often
- sparse paths care about storage-preserving policy and selected-inverse style
  summaries
- matrix-free paths care about Krylov methods, quadrature, probe variance,
  implicit adjoints, and reusable projected state

## Why Curvature Is A Shared Helper Layer

Curvature uses the same mathematical substrates as the matrix families:

- Hessian-like operators
- Fisher and GGN approximations
- posterior precision
- inverse diagonal and selected inverse summaries
- logdet, trace, and covariance pushforwards

So the repo treats curvature as a shared second-order helper layer rather than a
separate disjoint category.

This keeps the mathematical story coherent:

- dense, sparse, and matrix-free paths provide the operator substrate
- curvature provides reusable second-order constructions on top of it

## Why The Repo Has Both Theory Notes And Practical Docs

The repo deliberately separates:

- theory: what the method means and why it is structured this way
- standards: how the public/runtime surface must behave
- practical docs: how to use the API and when it is fast or stable in practice

This separation exists because numerical libraries fail in two different ways:

1. the method is not mathematically clear
2. the usage pattern is not operationally clear

arbPlusJAX tries to avoid both.

## Relationship To The Other Theory Notes

This file is the general overview.

The other theory notes should be read as specializations of this general model:

- scalar and enclosure semantics:
  [point_basic_surface_methodology.md](/docs/theory/point_basic_surface_methodology.md)
- matrix interval and mode semantics:
  [matrix_interval_and_modes.md](/docs/theory/matrix_interval_and_modes.md)
- matrix-free operator semantics:
  [matrix_free_operator_methodology.md](/docs/theory/matrix_free_operator_methodology.md)
- sparse/block/vblock structure:
  [sparse_block_vblock_methodology.md](/docs/theory/sparse_block_vblock_methodology.md)
- family-specific special-function methodology notes for gamma, Bessel,
  Barnes, and hypergeometric families

## Summary

The repo’s general theory can be summarized as:

- keep `point` as the performance-first repeated-evaluation surface
- keep `basic` as the baseline conservative or uncertainty-aware surface
- use `adaptive` and `rigorous` where the mathematics justifies stronger work
- keep dense, sparse, and matrix-free as distinct mathematical regimes
- treat curvature as a shared second-order layer over those regimes
- make diagnostics, metadata, reuse boundaries, and AD direction explicit
  because they are part of the public mathematical interpretation of the
  library, not only implementation detail
