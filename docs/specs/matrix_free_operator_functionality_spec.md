Last updated: 2026-03-29T00:00:00Z

# Matrix Free Operator Functionality Spec

## Purpose

This document defines the semantic scope of matrix-free and operator-first
functionality.

## Category

Coverage category:

5. Matrix-Free / Operator Functionality

## Scope

This category covers apply-first linear operators where the primary public
object is not a stored explicit matrix.

Representative families:

- `jrb_mat`
- `jcb_mat`
- operator-prepared plans and matrix-free helper layers

## Semantic Rule

Matrix-free/operator surfaces represent linear maps through apply or oracle
semantics first.

They are not:

- dense matrices with hidden storage
- sparse matrices with hidden storage

Even when an explicit matrix is used to build an operator, the operator category
remains distinct because the intended execution route is apply/plan-based.

## Required Axes

Operator APIs should remain legible across:

- operator kind:
  dense-backed / sparse-backed / shell / finite-difference / abstract operator
- structure subtype:
  symmetric / Hermitian / SPD / HPD / general
- execution route:
  direct apply / prepared operator plan / cached transpose or adjoint plan /
  solve plan / diagnostics-bearing route
- workload kind:
  apply / transpose apply / adjoint apply / solve / inverse / logdet /
  multi-shift / estimator

## Required Semantic Meaning

Matrix-free semantics should cover:

- `matvec`
- `rmatvec`
- adjoint or conjugate-transpose apply where relevant
- prepared operator-plan reuse
- operator-first AD expectations
- explicit distinction between direct and prepared execution routes

## Relationship To Other Categories

- matrix-free is not a sparse fallback
- dense and sparse families may feed matrix-free plans, but that does not erase
  the operator-first category

## Notebook Requirement

Matrix-free notebooks should show:

- direct operator construction
- prepared plan usage
- cached transpose/adjoint routes
- CPU/GPU crossover guidance
- AD on the real operator surface

