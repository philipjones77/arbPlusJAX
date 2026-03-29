Last updated: 2026-03-29T00:00:00Z

# Interval Box Precision Modes Spec

## Purpose

This document defines the semantic scope of interval/box/precision modes as a
cross-cutting function category.

## Category

Coverage category:

2. Interval / Box / Precision Modes

## Scope

This category owns the shared semantics behind:

- `point`
- `basic`
- `adaptive`
- `rigorous`

It also owns shared expectations for:

- interval and box lifting
- precision routing
- midpoint interpretation
- enclosure widening / invalidation semantics
- wrapper-mode dispatch

## Semantic Rule

Mode is an evaluation-semantic axis, not a family axis.

The same mathematical family may expose several modes, but the meaning of those
modes should stay consistent across scalar, matrix, sparse, and special-function
families unless a documented specialization says otherwise.

## Required Semantic Axes

Mode-bearing APIs should make clear:

- whether the result is point-valued or enclosure-valued
- whether precision is fixed or policy-routed
- whether containment is heuristic/basic or tighter/adaptive/rigorous
- whether diagnostics/error handling change by mode

## Relationship To Other Categories

This category is infrastructure for:

- scalars
- dense matrices
- sparse matrices
- matrix-free/operator surfaces where enclosure semantics exist
- special functions

It should not be treated as a separate mathematical family.

## Notebook Requirement

Canonical notebooks that use mode-bearing APIs should show:

- mode selection explicitly
- midpoint interpretation where relevant
- precision controls explicitly
- repeated-call binder usage for interval/basic surfaces when supported

