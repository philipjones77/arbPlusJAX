Last updated: 2026-03-29T00:00:00Z

# Cross Cutting Fast JAX Operational Spec

## Purpose

This document defines the semantic scope of the fast-JAX and operational-JAX
cross-category layer.

## Cross Category

Cross-cutting point-fast JAX conversion and operational JAX

## Scope

This layer covers:

- structural fast-JAX readiness
- repeated-call operational usage
- stable-shape execution patterns
- cached / prepared / compiled repeated-call routes
- off-hot-path diagnostics

## Semantic Rule

Fast-JAX and operational-JAX are cross-category execution semantics.

They should be interpreted consistently across:

- scalar
- interval/basic
- dense
- sparse
- matrix-free
- special functions

## Required Meaning

This layer should define, semantically:

- what counts as structurally fast-JAX
- what counts as practical operational-JAX
- how prepared/bound/cached routes differ from direct routes
- how notebooks should teach the efficient repeated-call path

