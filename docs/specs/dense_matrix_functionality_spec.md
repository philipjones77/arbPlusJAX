Last updated: 2026-03-29T00:00:00Z

# Dense Matrix Functionality Spec

## Purpose

This document defines the semantic scope of the dense matrix category.

## Category

Coverage category:

3. Dense Matrix Functionality

## Scope

This category covers explicit dense matrix objects and their dense-first
algorithms.

Representative families:

- `arb_mat`
- `acb_mat`

## Semantic Rule

Dense matrix surfaces represent explicit full matrices where the primary object
model is a concrete dense array-like matrix, not a sparse storage or operator
oracle.

## Required Axes

Dense APIs should remain legible across:

- numeric kind:
  real / complex
- structure subtype:
  symmetric / Hermitian / SPD / HPD / triangular / banded
- execution route:
  direct / cached apply / prepared solve / diagnostics-bearing / compiled batch
- mode:
  point / basic / adaptive / rigorous where supported

## Required Surface Meaning

Dense semantics should cover:

- explicit matrix construction
- matrix algebra
- cached dense apply
- dense factorization-backed solves
- dense determinant / inverse / decomposition semantics
- dense AD on the real public surface

## Relationship To Other Categories

- sparse matrices are not a dense subtype
- matrix-free/operator is not a dense implementation route
- interval/box modes apply across dense surfaces but do not redefine “dense”

## Notebook Requirement

The dense notebook should show:

- direct dense calls
- cached `matvec` / `rmatvec`
- prepared solve/factor usage
- structure-specialized routes where relevant
- CPU/GPU practical guidance

