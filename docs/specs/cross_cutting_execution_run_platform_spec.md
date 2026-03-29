Last updated: 2026-03-29T00:00:00Z

# Cross Cutting Execution Run Platform Spec

## Purpose

This document defines the semantic scope of the cross-cutting execution and run
platform layer.

## Cross Category

Cross-cutting execution and run-platform

## Scope

This layer covers:

- CPU vs GPU execution intent
- runtime environment parameterization
- startup / compile / import boundaries
- retained artifact layout
- execution portability across supported environments

## Semantic Rule

Execution platform is a cross-category concern.

It is not owned by any one function family and should be expressed consistently
across scalar, matrix, sparse, matrix-free, and special-function surfaces.

## Required Meaning

This layer should define, semantically:

- how backend choice is expressed
- what counts as structural fast-JAX vs practical operational JAX
- what evidence is required for CPU/GPU claims
- how retained test/benchmark/notebook artifacts relate to execution claims

