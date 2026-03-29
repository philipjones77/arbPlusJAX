Last updated: 2026-03-29T00:00:00Z

# Core Numeric Scalars Spec

## Purpose

This document defines the semantic scope of the core numeric scalar category.

## Category

Coverage category:

1. Core Numeric Scalars

## Scope

This category covers explicit scalar-valued numerical surfaces such as:

- real interval/ball-like scalars
- complex box/ball-like scalars
- point-only scalar helper families
- scalar arithmetic, transcendental, and core special-value support

Representative families:

- `arb_core`
- `acb_core`
- `arf`
- `acf`
- `fmpr`
- `fmpzi`
- `arb_fpwrap`
- scalar calc/helper surfaces that remain scalar in semantic scope

## Semantic Rule

Scalar surfaces represent single numeric quantities rather than matrices,
vectors, operators, or family-level batched abstractions.

The scalar category is the semantic base for:

- point mode
- interval/box lifting
- parameterized scalar special functions
- scalar AD and scalar backend/runtime policy

## Required Semantic Axes

Scalar APIs should be legible across:

- numeric kind:
  real / complex
- enclosure kind:
  point / interval / box
- precision kind:
  fixed hardware precision vs higher-precision routed evaluation
- execution route:
  direct / bound repeated-call / compiled repeated-call / diagnostics-bearing

## Relationship To Other Categories

- interval/box/precision infrastructure defines shared mode semantics
- special functions build on scalar semantics
- matrix categories should reuse scalar semantics where entries are scalar

## Notebook Requirement

Canonical scalar notebooks should show:

- direct scalar evaluation
- repeated-call binder usage
- backend choice
- diagnostics
- argument-direction and parameter-direction AD where applicable

